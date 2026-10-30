"""TTSS - tiny local TTS server. Run via run.cmd.

The server itself never touches the GPU: each model runs in a worker process
(app/worker.py) inside the Python env its backend asks for (app/envs.py).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / ".tools"
# Keep every download inside the project (set before any HF import).
os.environ.setdefault("HF_HOME", str(TOOLS / "hf-home"))
os.environ.setdefault("TORCH_HOME", str(TOOLS / "torch-home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import io
import json
import logging
import secrets
import subprocess
import tempfile
import threading
import time
import traceback
from datetime import datetime
from multiprocessing.connection import Listener
from urllib.parse import quote

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app import backends as B
from app import envs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ttss")

MODELS_FILE = ROOT / "models.json"            # your list (git-ignored), created on first load
DEFAULT_MODELS_FILE = ROOT / "models.default.json"  # shipped list, used until models.json exists
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
envs.LOGS.mkdir(parents=True, exist_ok=True)


def _nvidia_smi(query: str) -> list[str] | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, creationflags=0x08000000)
        return [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")] if out.returncode == 0 else None
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


DEVICE = "cuda" if _nvidia_smi("name") else "cpu"
BACKENDS = B.discover()
log.info("backends: %s | device: %s", ", ".join(BACKENDS), DEVICE)


def _lan_ip() -> str | None:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the LAN interface
            return s.getsockname()[0]
    except OSError:
        return None


if _ip := _lan_ip():
    log.info("On your phone (same network): http://%s:%s", _ip, os.environ.get("PORT", "7860"))


# --- saved model list ------------------------------------------------------

def read_models() -> list[dict]:
    for f in (MODELS_FILE, DEFAULT_MODELS_FILE):
        try:
            return [{"repo": m["repo"], "backend": m["backend"]} for m in json.loads(f.read_text(encoding="utf-8"))]
        except FileNotFoundError:
            continue
    return []


def save_model_entry(repo: str, backend: str) -> None:
    models = [m for m in read_models() if m["repo"] != repo]
    models.insert(0, {"repo": repo, "backend": backend})
    MODELS_FILE.write_text(json.dumps(models, indent=2), encoding="utf-8")


# --- worker process ----------------------------------------------------------

class Worker:
    """A backend running in its own process; talks over a local authenticated socket."""

    def __init__(self, cls, repo: str, status):
        self.cls, self.repo = cls, repo
        key = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=key)
        self.log_path = envs.LOGS / f"worker-{cls.name}.log"
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
        env.pop("VIRTUAL_ENV", None)
        self.proc = subprocess.Popen(
            [str(envs.python_for(cls.env)), "-u", "-m", "app.worker", "--backend", cls.name, "--repo", repo,
             "--device", DEVICE, "--port", str(self.listener.address[1]), "--authkey", key.hex()],
            cwd=ROOT, env=env, stdout=open(self.log_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
            creationflags=0x08000000,  # no console window
        )
        self.conn = None
        acc = threading.Thread(target=self._accept, daemon=True)
        acc.start()
        while acc.is_alive():
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker exited during startup: {self._tail()}")
            acc.join(0.2)
        status(f"Downloading / loading with '{cls.name}' (first time can take a while)...")
        msg = self._recv(timeout=None)
        if not msg["ok"]:
            self.close()
            raise RuntimeError(msg["error"])
        self.params = msg["params"]

    def _accept(self):
        self.conn = self.listener.accept()

    def _tail(self, n: int = 6) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return " | ".join(lines[-n:])
        except OSError:
            return "(no log)"

    def _recv(self, timeout):
        while not self.conn.poll(1.0):
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker crashed: {self._tail()}")
        return json.loads(self.conn.recv_bytes())

    def synth(self, text: str, params: dict):
        self.conn.send_bytes(json.dumps({"cmd": "synth", "text": text, "params": params}).encode())
        msg = self._recv(timeout=None)
        if not msg["ok"]:
            raise RuntimeError(msg["error"])
        audio = np.frombuffer(self.conn.recv_bytes(), dtype=np.float32)
        self.params = msg.get("params", self.params)
        return audio, msg["sr"], msg["elapsed"], msg.get("info")

    def close(self):
        try:
            if self.conn:
                self.conn.send_bytes(b'{"cmd": "quit"}')
            self.proc.wait(5)
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        self.listener.close()


# --- loaded-model state ----------------------------------------------------

class State:
    def __init__(self):
        self.lock = threading.Lock()  # guards worker + inference
        self.worker: Worker | None = None
        self.status = "idle"  # idle | loading | ready | error
        self.repo: str | None = None
        self.backend: str | None = None
        self.message = ""


S = State()


def _set_msg(m: str):
    S.message = m


def _load_worker(repo: str, backend_name: str | None):
    try:
        S.message = "Inspecting repo..."
        if backend_name:
            cls = BACKENDS.get(backend_name)
            if cls is None:
                raise ValueError(f"Unknown backend {backend_name!r}")
        else:
            info = B.inspect_repo(repo)
            cls = B.pick_backend(info, BACKENDS)
            if cls is None:
                raise ValueError(
                    f"No backend knows how to run {repo} "
                    f"(model_type={info.model_type}, library={info.library_name}, "
                    f"pipeline={info.pipeline_tag}). A new backend needs to be added."
                )
        S.backend = cls.name
        envs.ensure_env(cls, status=_set_msg)  # first use: builds .tools/envs/<env>
        with S.lock:
            if S.worker is not None:
                S.message = "Unloading previous model..."
                S.worker.close()
                S.worker = None
            t = time.time()
            S.worker = Worker(cls, repo, _set_msg)
        save_model_entry(repo, cls.name)
        S.status, S.message = "ready", f"Loaded in {time.time() - t:.1f}s"
        log.info("loaded %s with %s", repo, cls.name)
    except Exception as e:
        log.error("load failed: %s", traceback.format_exc())
        err = f"{type(e).__name__}: {e}"
        if S.worker is not None:  # failed before swapping: previous model is still usable
            S.repo, S.backend, S.status = S.worker.repo, S.worker.cls.name, "ready"
            S.message = f"Could not load {repo} ({err}) - kept previous model"
        else:
            S.status, S.message = "error", err


# --- API -------------------------------------------------------------------

app = FastAPI(title="TTSS")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


_gpu_cache = {"t": 0.0, "v": None}


def gpu_info():
    if time.time() - _gpu_cache["t"] > 2:
        r = _nvidia_smi("name,memory.used,memory.total")
        _gpu_cache.update(t=time.time(), v=({"name": r[0], "used_mb": int(r[1]), "total_mb": int(r[2])}
                                            if r else {"name": "CPU"}))
    return _gpu_cache["v"]


@app.get("/api/state")
def state():
    return {
        "status": S.status, "repo": S.repo, "backend": S.backend, "message": S.message,
        "params": S.worker.params if S.status == "ready" and S.worker else [],
        "models": read_models(),
        "backends": [{"name": c.name, "description": c.description} for c in BACKENDS.values()],
        "gpu": gpu_info(),
    }


class LoadReq(BaseModel):
    model: str
    backend: str | None = None


@app.post("/api/load")
def load(req: LoadReq):
    if S.status == "loading":
        raise HTTPException(409, "A model is already loading")
    try:
        repo = B.parse_repo_id(req.model)
    except ValueError as e:
        raise HTTPException(400, str(e))
    S.status, S.repo, S.backend, S.message = "loading", repo, req.backend, "Starting..."
    threading.Thread(target=_load_worker, args=(repo, req.backend or None), daemon=True).start()
    return {"ok": True, "repo": repo}


@app.post("/api/unload")
def unload():
    with S.lock:
        if S.worker is not None:
            S.worker.close()
            S.worker = None
    S.status, S.repo, S.backend, S.message = "idle", None, None, ""
    return {"ok": True}


@app.delete("/api/models")
def forget(repo: str):
    models = [m for m in read_models() if m["repo"] != repo]
    MODELS_FILE.write_text(json.dumps(models, indent=2), encoding="utf-8")
    return {"ok": True}


@app.post("/api/tts")
async def tts(request: Request):
    form = await request.form()
    text = str(form.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Text is empty")
    if S.status != "ready" or S.worker is None:
        raise HTTPException(409, "No model loaded")
    params = json.loads(str(form.get("params") or "{}"))

    # uploaded files (e.g. reference voice) -> temp paths passed to the backend
    tmp_files = []
    for key, val in form.multi_items():
        if hasattr(val, "read") and key != "text":
            data = await val.read()
            if data:
                suffix = Path(val.filename or "x.wav").suffix or ".wav"
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tf.write(data)
                tf.close()
                tmp_files.append(tf.name)
                params[key] = tf.name

    import anyio
    try:
        audio, sr, elapsed, info = await anyio.to_thread.run_sync(_synth, text, params)
    except Exception as e:
        log.error("tts failed: %s", traceback.format_exc())
        raise HTTPException(500, str(e))
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    wav = buf.getvalue()
    name = f"{datetime.now():%Y%m%d-%H%M%S}_{S.repo.split('/')[-1]}.wav"
    (OUTPUTS / name).write_bytes(wav)
    return Response(wav, media_type="audio/wav", headers={
        "X-Filename": name, "X-Duration": f"{len(audio) / sr:.2f}", "X-Elapsed": f"{elapsed:.2f}",
        "X-Info": quote(str(info or "")),
        "Access-Control-Expose-Headers": "X-Filename, X-Duration, X-Elapsed, X-Info",
    })


def _synth(text: str, params: dict):
    with S.lock:
        if S.worker is None:
            raise RuntimeError("Model was unloaded")
        return S.worker.synth(text, params)
