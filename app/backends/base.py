"""Base class every model backend implements.

To add support for a new architecture, drop a module into app/backends/ that
defines a subclass of TTSBackend and exposes it as `BACKEND = MyBackend`.
The registry discovers it automatically.

Rules for backend modules:
  * the file name must equal the backend's `name` (the worker imports it by name)
  * top-level imports: stdlib, numpy and .base only. Import torch & model packages
    inside load()/synthesize(). The web server imports every backend module to read
    its metadata, and it does not have every backend's packages installed.
  * each backend runs in a worker process using the Python env named by `env`
    (.tools/envs/<env>), created from `install` (see app/envs.py). env=None = main venv.
"""
from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RepoInfo:
    """What we know about a Hugging Face repo, used to pick a backend."""
    repo_id: str
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def model_type(self) -> str | None:
        return self.config.get("model_type")


@dataclass
class Param:
    """A user-facing generation option the UI renders as a control.

    kind: "select" | "number" | "text" | "textarea" | "audio" (file upload) | "checkbox"
    """
    name: str
    label: str
    kind: str = "text"
    default: Any = None
    choices: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    help: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class TTSBackend:
    #: unique id, also the module's name by convention
    name: str = "base"
    #: short human description shown in the UI
    description: str = ""
    #: name of the isolated Python env under .tools/envs/ (None = main .tools/venv)
    env: str | None = None
    #: Python version for that env
    python: str = "3.11"
    #: install steps for the env: each entry is the argument list of one `uv pip install`
    #: call, run in order (e.g. torch from the CUDA index first, then the model package)
    install: list[list[str]] = []
    #: free-form text shown next to each generated clip (seed used, etc.); set in synthesize()
    last_info: str | None = None

    def __init__(self, repo_id: str, device: str = "cuda"):
        self.repo_id = repo_id
        self.device = device

    # --- detection -------------------------------------------------------
    @classmethod
    def match(cls, info: RepoInfo) -> int:
        """Return a score > 0 if this backend can run the repo (higher wins)."""
        return 0

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        for attr in list(vars(self)):
            if attr not in ("repo_id", "device"):
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- inference -------------------------------------------------------
    def params(self) -> list[Param]:
        """Options shown in the UI for this model (voice, speed, ...)."""
        return []

    def synthesize(self, text: str, **params: Any) -> tuple[np.ndarray, int]:
        """Return (mono float32 waveform in [-1, 1], sample_rate)."""
        raise NotImplementedError


# --- helpers shared by backends -------------------------------------------

def to_mono_float32(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)
    if audio.ndim > 1:
        # (channels, samples) or (samples, channels) -> mono
        audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[-1] else -1)
    return audio


def split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Split long text into chunks near sentence boundaries."""
    parts = re.split(r"(?<=[.!?;:。！？])\s+|\n+", text.strip())
    chunks, cur = [], ""
    for p in filter(None, (s.strip() for s in parts)):
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [text]


def concat(chunks: list[np.ndarray], sr: int, gap_s: float = 0.15) -> np.ndarray:
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    out = []
    for i, c in enumerate(chunks):
        if i:
            out.append(gap)
        out.append(c)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
