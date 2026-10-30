"""Run a command while holding an exclusive GPU lock, so parallel jobs don't run out of VRAM.

    .tools\\venv\\Scripts\\python.exe scripts\\gpulock.py -- <command ...>
"""
import msvcrt
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path(__file__).resolve().parent.parent / ".tools" / "gpu.lock"

if __name__ == "__main__":
    cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if cmd and Path(cmd[0]).exists():  # accept forward-slash relative paths from bash
        cmd[0] = str(Path(cmd[0]).resolve())
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a+b") as f:
        waited = 0.0
        while True:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if waited == 0:
                    print("[gpulock] waiting for GPU...", flush=True)
                time.sleep(2)
                waited += 2
        try:
            sys.exit(subprocess.call(cmd))
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
