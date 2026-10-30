"""Per-backend Python environments under .tools/envs/<env>, built with the portable uv.

    .tools\\venv\\Scripts\\python.exe -m app.envs <backend_name> [--force]

The server also calls ensure_env() automatically the first time a backend is loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / ".tools"
UV = TOOLS / "uv" / "uv.exe"
ENVS = TOOLS / "envs"
LOGS = TOOLS / "logs"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
# default first install step: CUDA torch matching the main venv
TORCH_CUDA = ["torch", "torchaudio", "--index-url", CUDA_INDEX]


def python_for(env: str | None) -> Path:
    return (TOOLS / "venv" if not env else ENVS / env) / "Scripts" / "python.exe"


def uv_environ() -> dict:
    e = dict(os.environ)
    e.update(UV_PYTHON_INSTALL_DIR=str(TOOLS / "python"), UV_CACHE_DIR=str(TOOLS / "uv-cache"),
             UV_PYTHON_PREFERENCE="only-managed", UV_LINK_MODE="hardlink")
    e.pop("VIRTUAL_ENV", None)
    return e


def _spec_hash(cls) -> str:
    return hashlib.sha1(json.dumps([cls.python, cls.install]).encode()).hexdigest()[:12]


def is_ready(cls) -> bool:
    if not cls.env:
        return True
    marker = ENVS / cls.env / ".ttss-installed"
    return python_for(cls.env).exists() and marker.exists() and marker.read_text() == _spec_hash(cls)


def ensure_env(cls, status=print, force: bool = False) -> None:
    """Create/update the env for backend class `cls` if its install spec changed."""
    if not cls.env or (is_ready(cls) and not force):
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"env-{cls.env}.log"
    py = python_for(cls.env)
    with open(log_path, "a", encoding="utf-8") as log:
        def run(args, what):
            status(what)
            log.write(f"\n$ {' '.join(map(str, args))}\n")
            log.flush()
            r = subprocess.run([str(a) for a in args], stdout=log, stderr=subprocess.STDOUT, env=uv_environ())
            if r.returncode:
                raise RuntimeError(f"{what} failed (see {log_path})")

        if not py.exists() or force:
            run([UV, "venv", "--clear", ENVS / cls.env, "--python", cls.python], f"Creating env '{cls.env}'...")
        for i, step in enumerate(cls.install, 1):
            run([UV, "pip", "install", "--python", py, *step],
                f"Installing env '{cls.env}' ({i}/{len(cls.install)}): {' '.join(step)[:80]}")
    (ENVS / cls.env / ".ttss-installed").write_text(_spec_hash(cls))
    status(f"Env '{cls.env}' ready")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from app import backends as B

    name = sys.argv[1]
    cls = B.discover().get(name)
    if cls is None:
        sys.exit(f"unknown backend {name!r}")
    ensure_env(cls, force="--force" in sys.argv)
    print(python_for(cls.env))
