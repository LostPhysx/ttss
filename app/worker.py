"""Runs one backend in its own process / Python env.

Server mode (started by app/server.py):
    python -m app.worker --backend NAME --repo ORG/MODEL --port P --authkey HEX

Test mode (for development, no server needed):
    <env python> -m app.worker --backend NAME --repo ORG/MODEL --test "Some text" \
        --params '{"voice": "x"}' --out out.wav
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".tools" / "hf-home"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".tools" / "torch-home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402


def load_backend(name: str, repo: str, device: str):
    mod = importlib.import_module(f"app.backends.{name}")
    m = mod.BACKEND(repo, device)
    m.load()
    return m


def clean(audio) -> np.ndarray:
    from app.backends.base import to_mono_float32
    a = to_mono_float32(audio)
    return np.clip(np.nan_to_num(a), -1, 1).astype(np.float32)


def synth(m, text: str, params: dict):
    t = time.time()
    audio, sr = m.synthesize(text, **params)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    return clean(audio), int(sr), time.time() - t


def vram_mb() -> int | None:
    try:
        import torch
        return int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
    except ImportError:
        return None


def serve(a):
    from multiprocessing.connection import Client

    conn = Client(("127.0.0.1", a.port), authkey=bytes.fromhex(a.authkey))
    send = lambda d: conn.send_bytes(json.dumps(d).encode())
    try:
        m = load_backend(a.backend, a.repo, a.device)
        send({"ok": True, "params": [p.to_dict() for p in m.params()]})
    except BaseException as e:
        traceback.print_exc()
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    while True:
        try:
            msg = json.loads(conn.recv_bytes())
        except (EOFError, OSError):
            return
        if msg.get("cmd") == "quit":
            return
        try:
            audio, sr, elapsed = synth(m, msg["text"], msg.get("params") or {})
            send({"ok": True, "sr": sr, "elapsed": elapsed, "info": getattr(m, "last_info", None),
                  "params": [p.to_dict() for p in m.params()]})
            conn.send_bytes(audio.tobytes())
        except BaseException as e:
            traceback.print_exc()
            send({"ok": False, "error": f"{type(e).__name__}: {e}"})


def test(a):
    import soundfile as sf

    t = time.time()
    m = load_backend(a.backend, a.repo, a.device)
    print(f"[test] loaded in {time.time() - t:.1f}s, vram {vram_mb()} MB")
    print("[test] params:", json.dumps([p.to_dict() for p in m.params()], ensure_ascii=False)[:2000])
    audio, sr, el = synth(m, a.test, json.loads(a.params))
    dur = len(audio) / sr
    sf.write(a.out, audio, sr)
    print(f"[test] {dur:.2f}s audio in {el:.2f}s (RTF {el / max(dur, 1e-6):.2f}), sr {sr}, "
          f"peak {abs(audio).max():.2f}, max vram {vram_mb()} MB, info {getattr(m, 'last_info', None)!r} -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int)
    p.add_argument("--authkey")
    p.add_argument("--test")
    p.add_argument("--params", default="{}")
    p.add_argument("--out", default="test.wav")
    a = p.parse_args()
    test(a) if a.test else serve(a)
