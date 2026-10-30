"""Mistral Voxtral-4B-TTS (mistralai/Voxtral-4B-TTS-2603) running in plain PyTorch.

The official runtime is vLLM-Omni (Linux, 16 GB GPU). This backend uses our own port in
_voxtral_model.py, with the 3B LM backbone quantized so it fits an 8 GB card.
The open checkpoint ships without the speech encoder, so only the preset voices work.
"""
from __future__ import annotations

import json
import random
import re

import numpy as np

from .base import Param, RepoInfo, TTSBackend, concat, split_sentences

# Only English and German are exposed. Add others from params.json to VOICES to enable them.
VOICES = {
    "English": ["neutral_female", "neutral_male", "casual_female", "casual_male", "cheerful_female"],
    "German": ["de_female", "de_male"],
}
PRECISIONS = {
    "nf4": "4-bit (fast, ~2.8 GB VRAM)",
    "int8": "8-bit (closest to original, slower, ~4.3 GB)",
    "bf16": "bf16 original (needs ~7.5 GB free VRAM)",
}
FRAME_RATE = 12.5


class VoxtralTTSBackend(TTSBackend):
    name = "voxtral_tts"
    description = "Mistral Voxtral TTS (own PyTorch port, quantized LM)"
    requirements = ["mistral_common>=1.10", "bitsandbytes", "safetensors"]

    @classmethod
    def match(cls, info: RepoInfo) -> int:
        return 100 if info.model_type == "voxtral_tts" else 0

    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        from ._voxtral_model import VoxtralTTS

        path = snapshot_download(self.repo_id, allow_patterns=[
            "consolidated.safetensors", "params.json", "tekken.json", "voice_embedding/*"])
        with open(f"{path}/params.json", encoding="utf-8") as f:
            params = json.load(f)
        self.tokenizer = MistralTokenizer.from_file(f"{path}/tekken.json")
        available = params["multimodal"]["audio_tokenizer_args"].get("voice", {})
        self.voices = {
            v: torch.load(f"{path}/voice_embedding/{v}.pt", map_location="cpu", weights_only=True)
            for group in VOICES.values() for v in group if v in available
        }
        self.model = VoxtralTTS(f"{path}/consolidated.safetensors", params, self.device,
                                precision=self._default_precision())

    def _default_precision(self) -> str:
        return "nf4"

    def params(self) -> list[Param]:
        voice_choices = [{"value": v, "label": v.replace("_", " "), "group": g}
                         for g, vs in VOICES.items() for v in vs if v in self.voices]
        return [
            Param("voice", "Voice", "select", "neutral_female", choices=voice_choices),
            Param("precision", "LM precision", "select", "nf4",
                  choices=[{"value": k, "label": v} for k, v in PRECISIONS.items()],
                  help="Precision of the 3B language model. Changing it reloads the LM (~10-20 s)."),
            Param("cfg_alpha", "CFG strength", "number", 1.2, min=1.0, max=3.0, step=0.05,
                  help="Classifier-free guidance on the acoustic flow. Official default 1.2. "
                       "Higher = follows the voice/text more strongly, can sound harsher."),
            Param("flow_steps", "Flow steps", "number", 7, min=1, max=32, step=1,
                  help="Euler steps of the flow-matching acoustic decoder per frame. Official default 7."),
            Param("noise_scale", "Acoustic noise", "number", 1.0, min=0.0, max=1.5, step=0.05,
                  help="Scale of the starting noise for acoustic codes (1.0 = official). "
                       "Lower = flatter / more stable, higher = more varied."),
            Param("temperature", "Semantic temperature", "number", 0.0, min=0.0, max=1.5, step=0.05,
                  help="0 = greedy argmax, as in the official runtime. >0 samples semantic tokens "
                       "for more prosody variation (experimental)."),
            Param("top_k", "Semantic top-k", "number", 0, min=0, max=8192, step=1,
                  help="Only with temperature > 0. 0 = no limit."),
            Param("seed", "Seed", "number", None, min=0, step=1,
                  help="Empty = random. The seed used is shown with each clip so you can reproduce it."),
            Param("split", "Split text", "select", "paragraphs", choices=[
                {"value": "none", "label": "No splitting"},
                {"value": "paragraphs", "label": "By paragraph"},
                {"value": "sentences", "label": "By sentence (~300 chars)"}],
                help="Long text is generated piece by piece and joined. Each piece starts fresh "
                     "from the voice prompt, which keeps long outputs stable."),
            Param("gap", "Pause between pieces (s)", "number", 0.25, min=0.0, max=3.0, step=0.05),
            Param("max_seconds", "Max seconds per piece", "number", 60, min=5, max=300, step=5,
                  help="Safety cap if the model doesn't stop on its own."),
            Param("normalize", "Normalize loudness", "checkbox", True,
                  help="The English preset voices are recorded quietly (~-40 dBFS). "
                       "This scales the output to a peak of -1 dBFS."),
        ]

    def _pieces(self, text: str, mode: str) -> list[str]:
        if mode == "sentences":
            return split_sentences(text, max_chars=300)
        if mode == "paragraphs":
            return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [" ".join(text.split())]

    def synthesize(self, text: str, voice: str = "neutral_female", precision: str = "nf4",
                   cfg_alpha: float = 1.2, flow_steps: int = 7, noise_scale: float = 1.0,
                   temperature: float = 0.0, top_k: int = 0, seed=None, split: str = "paragraphs",
                   gap: float = 0.25, max_seconds: float = 60, normalize: bool = True, **_):
        import torch
        from mistral_common.protocol.speech.request import SpeechRequest

        if voice not in self.voices:
            raise ValueError(f"Unknown voice {voice!r}")
        if precision not in PRECISIONS:
            raise ValueError(f"Unknown precision {precision!r}")
        try:
            self.model.set_precision(precision)
        except torch.OutOfMemoryError:
            self.model.lm = None
            torch.cuda.empty_cache()
            self.model.set_precision("nf4")
            raise RuntimeError(f"Not enough free VRAM for {PRECISIONS[precision]}. Switched back to 4-bit.")

        seed = int(seed) if seed not in (None, "") else random.randrange(2**31)
        sr = self.model.sampling_rate
        chunks = []
        for i, piece in enumerate(self._pieces(text, split)):
            tokens = self.tokenizer.encode_speech_request(SpeechRequest(input=piece, voice=voice)).tokens
            wav = self.model.generate(
                tokens, self.voices[voice], cfg_alpha=float(cfg_alpha), flow_steps=int(flow_steps),
                noise_scale=float(noise_scale), temperature=float(temperature), top_k=int(top_k),
                max_frames=int(float(max_seconds) * FRAME_RATE), seed=seed + i,
            )
            chunks.append(wav.numpy().astype(np.float32))
        self.last_info = f"seed {seed} · {voice} · {precision}"
        audio = concat(chunks, sr, gap_s=float(gap))
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if normalize and peak > 1e-4:
            audio = audio * (0.891 / peak)  # -1 dBFS
        return audio, sr


BACKEND = VoxtralTTSBackend
