"""Prepare models ahead of time: build each backend's Python env and download its weights.

Run from the repo root after setup.ps1 (which builds the main env):

    .tools\\venv\\Scripts\\python.exe scripts\\prepare.py --list
    .tools\\venv\\Scripts\\python.exe scripts\\prepare.py                      # all models in models.default.json
    .tools\\venv\\Scripts\\python.exe scripts\\prepare.py k2-fsa/OmniVoice     # one or more repos (or backend names)
    .tools\\venv\\Scripts\\python.exe scripts\\prepare.py --smoke              # + generate one short clip per model on the GPU

Options: --envs-only (skip downloads), --download-only (skip env builds), --smoke (GPU test per model).
Everything lands in .tools/ (envs in .tools/envs, weights in .tools/hf-home). Safe to re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".tools" / "hf-home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from app import backends as B  # noqa: E402
from app import envs  # noqa: E402

SMOKE_TEXT = "Hello! This is a short test of local speech synthesis."


def catalog() -> list[dict]:
    return json.loads((ROOT / "models.default.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="repo ids or backend names (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--envs-only", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    backends = B.discover()
    items = catalog()
    if a.list:
        for m in items:
            cls = backends.get(m["backend"])
            print(f"{m['repo']:42} backend={m['backend']:12} env={(cls.env if cls else '?') or 'main'}")
        return
    if a.models:
        items = [m for m in items if m["repo"] in a.models or m["backend"] in a.models] + \
                [{"repo": r, "backend": None} for r in a.models
                 if "/" in r and not any(m["repo"] == r for m in items)]

    failures = []
    for m in items:
        repo = m["repo"]
        try:
            cls = backends.get(m["backend"]) if m.get("backend") else B.pick_backend(B.inspect_repo(repo), backends)
            if cls is None:
                raise RuntimeError("no backend matches this repo")
            print(f"\n== {repo}  (backend {cls.name}, env {cls.env or 'main'})", flush=True)
            if not a.download_only:
                envs.ensure_env(cls, status=lambda s: print("  ", s, flush=True))
            if not a.envs_only:
                from huggingface_hub import snapshot_download
                t = time.time()
                path = snapshot_download(repo, allow_patterns=m.get("download"))
                print(f"   weights ready ({time.time() - t:.0f}s): {path}", flush=True)
            if a.smoke:
                out = ROOT / ".tools" / "test" / f"prepare_{cls.name}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(ROOT / "scripts" / "gpulock.py"), "--",
                       str(envs.python_for(cls.env)), "-m", "app.worker", "--backend", cls.name,
                       "--repo", repo, "--test", SMOKE_TEXT, "--out", str(out)]
                env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
                env.pop("VIRTUAL_ENV", None)
                r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace")
                lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[test]") and "params:" not in ln]
                print("\n".join("   " + ln for ln in lines) or r.stdout[-1500:] + r.stderr[-1500:], flush=True)
                if r.returncode:
                    raise RuntimeError(f"smoke test failed (exit {r.returncode})")
        except Exception as e:
            print(f"   FAILED: {e}", flush=True)
            failures.append(repo)

    print("\nDone." if not failures else f"\nFailed: {', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
