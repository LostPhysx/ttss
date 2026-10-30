"""Backend discovery and repo inspection."""
from __future__ import annotations

import importlib
import json
import logging
import pkgutil
import re

from .base import RepoInfo, TTSBackend

log = logging.getLogger("ttss")


def discover() -> dict[str, type[TTSBackend]]:
    found: dict[str, type[TTSBackend]] = {}
    for mod in pkgutil.iter_modules(__path__):
        if mod.name == "base" or mod.name.startswith("_"):  # _helpers are not backends
            continue
        try:
            m = importlib.import_module(f"{__name__}.{mod.name}")
        except Exception as e:  # a backend with missing deps shouldn't kill the app
            log.warning("backend %s failed to import: %s", mod.name, e)
            continue
        cls = getattr(m, "BACKEND", None)
        if cls is not None:
            found[cls.name] = cls
    return found


def parse_repo_id(url_or_id: str) -> str:
    s = url_or_id.strip().rstrip("/")
    m = re.match(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(.+)$", s)
    if m:
        s = m.group(1)
    parts = [p for p in s.split("/") if p]
    if parts and parts[0] in ("models", "spaces", "datasets"):
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"Not a Hugging Face model URL or id: {url_or_id!r}")
    return f"{parts[0]}/{parts[1]}"  # drops /tree/main, /blob/..., etc.


def inspect_repo(repo_id: str) -> RepoInfo:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    mi = api.model_info(repo_id, files_metadata=False)
    info = RepoInfo(
        repo_id=repo_id,
        pipeline_tag=mi.pipeline_tag,
        library_name=mi.library_name,
        tags=list(mi.tags or []),
        files=[s.rfilename for s in (mi.siblings or [])],
    )
    # HF-format config.json, or Mistral-format params.json (consolidated checkpoints)
    for name in ("config.json", "params.json"):
        if name in info.files:
            try:
                with open(hf_hub_download(repo_id, name), encoding="utf-8") as f:
                    info.config = json.load(f)
                break
            except Exception as e:
                log.warning("could not read %s for %s: %s", name, repo_id, e)
    return info


def pick_backend(info: RepoInfo, backends: dict[str, type[TTSBackend]]) -> type[TTSBackend] | None:
    scored = sorted(((cls.match(info), cls) for cls in backends.values()), key=lambda t: -t[0])
    return scored[0][1] if scored and scored[0][0] > 0 else None
