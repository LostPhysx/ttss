"""Generic fallback: any model that works with transformers' text-to-speech pipeline
(VITS / MMS-TTS, Bark, SeamlessM4T-v2, ...)."""
from __future__ import annotations

from .base import RepoInfo, TTSBackend, concat, split_sentences, to_mono_float32

KNOWN_TYPES = {"vits", "bark", "seamless_m4t", "seamless_m4t_v2", "fastspeech2_conformer"}


class HFPipelineBackend(TTSBackend):
    name = "hf_pipeline"
    description = "transformers text-to-speech pipeline (generic)"

    @classmethod
    def match(cls, info: RepoInfo) -> int:
        if info.model_type in KNOWN_TYPES:
            return 20
        if info.library_name == "transformers" and info.pipeline_tag in ("text-to-speech", "text-to-audio"):
            return 10
        return 0

    def load(self) -> None:
        import torch
        from transformers import pipeline

        self.pipe = pipeline(
            "text-to-speech",
            model=self.repo_id,
            device=self.device,
            torch_dtype=torch.float32,
        )

    def synthesize(self, text: str, **params):
        import torch
        chunks, sr = [], None
        for piece in split_sentences(text):
            with torch.inference_mode():
                out = self.pipe(piece)
            sr = out["sampling_rate"]
            chunks.append(to_mono_float32(out["audio"]))
        return concat(chunks, sr), sr


BACKEND = HFPipelineBackend
