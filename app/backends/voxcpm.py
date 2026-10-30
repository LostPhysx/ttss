"""OpenBMB VoxCPM (VoxCPM2, VoxCPM1.5, VoxCPM-0.5B) via the official `voxcpm` package.

VoxCPM is a tokenizer-free diffusion-autoregressive TTS. VoxCPM2 (2B, 48 kHz, 30 languages)
adds voice design from a text description and timbre-only cloning from a reference clip;
the 1.x models (zh/en, 44.1 / 16 kHz) only clone by continuing a prompt clip + transcript.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np

from .base import Param, RepoInfo, TTSBackend, concat, split_sentences

CU128 = "https://download.pytorch.org/whl/cu128"
ROOT = Path(__file__).resolve().parents[2]


class VoxCPMBackend(TTSBackend):
    name = "voxcpm"
    description = "OpenBMB VoxCPM (voice design, voice cloning; VoxCPM2 = 30 languages, 48 kHz)"
    env = "voxcpm"
    python = "3.11"
    # torch 2.8 keeps torchaudio's soundfile backend (2.9+ needs torchcodec + FFmpeg DLLs),
    # which the optional ZipEnhancer denoiser uses for loading/saving audio.
    install = [
        ["torch==2.8.0", "torchaudio==2.8.0", "--index-url", CU128],
        ["voxcpm==2.0.3"],
    ]

    @classmethod
    def match(cls, info: RepoInfo) -> int:
        # needs the PyTorch checkpoint layout (config.json "architecture" + audiovae.pth);
        # GGUF / ONNX / nano-vllm repos that merely carry the tag can't run with this package
        arch = str(info.config.get("architecture", "")).lower()
        if arch not in ("voxcpm", "voxcpm2") or "audiovae.pth" not in info.files:
            return 0
        return 90 if (info.library_name == "voxcpm" or "voxcpm" in info.tags) else 70

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / ".tools" / "modelscope"))
        from voxcpm import VoxCPM
        from voxcpm.model.voxcpm2 import VoxCPM2Model

        # optimize=False: torch.compile/triton is not available on Windows.
        # The denoiser (ModelScope ZipEnhancer) is loaded lazily when first requested.
        self.model = VoxCPM.from_pretrained(self.repo_id, load_denoiser=False, optimize=False,
                                            device=self.device)
        self.tts = self.model.tts_model
        self.is_v2 = isinstance(self.tts, VoxCPM2Model)
        self.sample_rate = int(self.tts.sample_rate)
        self._denoiser = None

    def _get_denoiser(self):
        if self._denoiser is None:
            from voxcpm.zipenhancer import ZipEnhancer
            self._denoiser = ZipEnhancer("iic/speech_zipenhancer_ans_multiloss_16k_base")
        return self._denoiser

    # --- UI --------------------------------------------------------------
    def params(self) -> list[Param]:
        v2 = getattr(self, "is_v2", True)
        ps: list[Param] = []
        if v2:
            ps.append(Param(
                "voice_description", "Voice description / style", "textarea", "",
                help="Optional. Without reference audio this designs a new voice from the description "
                     "(e.g. 'A middle-aged man, deep calm voice, slow pace'). With reference audio it "
                     "steers style only (e.g. 'slightly faster, cheerful tone'). Write it in English. "
                     "Ignored in 'continuation' cloning mode."))
        ps += [
            Param("ref_audio", "Reference audio (voice cloning)", "audio", None,
                  help="Optional clip of the voice to clone (a few to ~15 s, clean speech). "
                       + ("Without a transcript only the timbre is cloned." if v2 else
                          "VoxCPM 1.x needs the exact transcript too.")),
            Param("ref_text", "Reference transcript", "textarea", "",
                  help="Exact words spoken in the reference audio. Enables 'continuation' cloning, which "
                       "also copies rhythm, emotion and recording style (most faithful)."),
        ]
        if v2:
            ps.append(Param("clone_mode", "Cloning mode", "select", "auto", choices=[
                {"value": "auto", "label": "Auto (best for what you provide)"},
                {"value": "reference", "label": "Timbre only (reference, allows style control)"},
                {"value": "continuation", "label": "Continuation only (audio + transcript)"},
                {"value": "ultimate", "label": "Ultimate (continuation + reference)"}],
                help="Auto = 'ultimate' if a transcript is given, otherwise 'timbre only'. "
                     "Ultimate gives the highest similarity; timbre only is more flexible and "
                     "follows the voice description."))
        ps += [
            Param("cfg_value", "CFG value", "number", 2.0, min=1.0, max=5.0, step=0.1,
                  help="Classifier-free guidance of the diffusion decoder. Official default 2.0. "
                       "Higher = follows text/prompt more closely but can sound strained; lower = more relaxed."),
            Param("inference_timesteps", "Inference timesteps", "number", 10, min=1, max=50, step=1,
                  help="Diffusion steps per audio patch. Official default 10. More = slightly better "
                       "quality, slower; 4-6 is faster."),
            Param("normalize", "Normalize text", "checkbox", False,
                  help="Run the built-in text normalizer (numbers, dates, symbols -> words) before "
                       "synthesis. Designed for Chinese/English; leave off for German."),
            Param("denoise", "Denoise reference audio", "checkbox", False,
                  help="Clean the reference clip with the ZipEnhancer denoiser (16 kHz) before cloning. "
                       "First use downloads a small model. Only helps with noisy references."),
            Param("trim_silence", "Trim reference silence", "checkbox", False,
                  help="Trim long leading/trailing silence from the reference clip (VAD) before cloning."),
            Param("retry_badcase", "Retry bad cases", "checkbox", True,
                  help="If the generated audio is unreasonably long for the text (model babbling or "
                       "stuck), retry with the next seed. Official default on."),
            Param("retry_badcase_max_times", "Max retries", "number", 3, min=1, max=10, step=1,
                  help="Maximum attempts when 'Retry bad cases' is on."),
            Param("retry_badcase_ratio_threshold", "Bad-case length ratio", "number", 6.0,
                  min=2.0, max=20.0, step=0.5,
                  help="Audio-patches-per-text-token ratio above which an output counts as a bad case. "
                       "Also caps the output length. Official default 6; raise it for very slow speech."),
            Param("max_len", "Max length (patches)", "number", 4096, min=64, max=8192, step=64,
                  help="Hard cap on generated audio patches per piece (VoxCPM2/1.5: 6.25 patches per second)."),
            Param("seed", "Seed", "number", None, min=0, step=1,
                  help="Empty = random. The seed used is shown with each clip so you can reproduce it."),
            Param("split", "Split text", "select", "paragraphs", choices=[
                {"value": "none", "label": "No splitting"},
                {"value": "paragraphs", "label": "By paragraph"},
                {"value": "sentences", "label": "By sentence (~300 chars)"}],
                help="Long text is generated piece by piece and joined; shorter pieces are more "
                     "stable. Pieces share the reference voice (or see 'Keep voice across pieces')."),
            Param("keep_voice", "Keep voice across pieces", "checkbox", True,
                  help="Without reference audio, use the first generated piece as the voice prompt "
                       "for the following pieces, so a split text keeps one voice. Off = every piece "
                       "is generated independently (new random voice unless described)."),
            Param("gap", "Pause between pieces (s)", "number", 0.2, min=0.0, max=3.0, step=0.05),
            Param("loudness", "Normalize loudness", "checkbox", True,
                  help="Scale the output to a peak of -1 dBFS."),
        ]
        return ps

    # --- inference -------------------------------------------------------
    @staticmethod
    def _sig(fn) -> set:
        import inspect
        return set(inspect.signature(fn).parameters)

    def _pieces(self, text: str, mode: str) -> list[str]:
        if mode == "sentences":
            return split_sentences(text, max_chars=300)
        if mode == "paragraphs":
            return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [" ".join(text.split())]

    def synthesize(self, text: str, voice_description: str = "", ref_audio: str | None = None,
                   ref_text: str = "", clone_mode: str = "auto", cfg_value: float = 2.0,
                   inference_timesteps: int = 10, normalize: bool = False, denoise: bool = False,
                   trim_silence: bool = False, retry_badcase: bool = True,
                   retry_badcase_max_times: int = 3, retry_badcase_ratio_threshold: float = 6.0,
                   max_len: int = 4096, seed=None, split: str = "paragraphs", gap: float = 0.2,
                   keep_voice: bool = True, loudness: bool = True, **_):
        import tempfile

        import soundfile as sf
        import torch

        desc = (voice_description or "").strip().strip("()").strip() if self.is_v2 else ""
        ref_text = " ".join((ref_text or "").split())
        ref_audio = ref_audio or None
        if ref_audio and not os.path.exists(ref_audio):
            raise FileNotFoundError(f"reference audio not found: {ref_audio}")

        # which prompt inputs to use
        prompt_wav = reference_wav = None
        mode = "design" if desc else "zero-shot"
        if ref_audio:
            if not self.is_v2:
                if not ref_text:
                    raise ValueError("VoxCPM 1.x needs the reference transcript for cloning.")
                prompt_wav, mode = ref_audio, "continuation"
            else:
                if clone_mode == "auto":
                    clone_mode = "ultimate" if ref_text else "reference"
                if clone_mode in ("continuation", "ultimate") and not ref_text:
                    raise ValueError(f"Cloning mode '{clone_mode}' needs the reference transcript.")
                if clone_mode in ("continuation", "ultimate"):
                    prompt_wav = ref_audio
                if clone_mode in ("reference", "ultimate"):
                    reference_wav = ref_audio
                mode = clone_mode
        if mode == "continuation":
            desc = ""  # style instructions would be read as part of the continued speech

        temps = []
        try:
            if denoise and ref_audio:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    temps.append(tmp.name)
                self._get_denoiser().enhance(ref_audio, output_path=tmp.name)
                prompt_wav = tmp.name if prompt_wav else None
                reference_wav = tmp.name if reference_wav else None

            cache = None
            if prompt_wav or reference_wav:
                kw = dict(prompt_text=ref_text if prompt_wav else None, prompt_wav_path=prompt_wav)
                if self.is_v2:
                    kw["reference_wav_path"] = reference_wav
                if trim_silence and "trim_silence_vad" in self._sig(self.tts.build_prompt_cache):
                    kw["trim_silence_vad"] = True
                cache = self.tts.build_prompt_cache(**kw)

            seed = int(seed) if seed not in (None, "") else random.randrange(2**31)
            normalizer = None
            if normalize:
                normalizer = getattr(self.model, "text_normalizer", None)
                if normalizer is None:
                    from voxcpm.utils.text_normalize import TextNormalizer
                    normalizer = self.model.text_normalizer = TextNormalizer()

            pieces = self._pieces(text, split)
            chunks, used = [], []
            for i, piece in enumerate(pieces):
                if normalizer is not None:
                    piece = normalizer.normalize(piece)
                spoken = piece
                if desc:
                    piece = f"({desc}){piece}"
                kw = {"seed": seed + i} if "seed" in self._sig(self.tts._generate_with_prompt_cache) else {}
                torch.manual_seed(seed + i)  # voxcpm 2.0.3 has no seed argument yet
                gen = self.tts._generate_with_prompt_cache(
                    target_text=piece, prompt_cache=cache, min_len=2, max_len=int(max_len),
                    inference_timesteps=int(inference_timesteps), cfg_value=float(cfg_value),
                    retry_badcase=bool(retry_badcase),
                    retry_badcase_max_times=max(1, int(retry_badcase_max_times)),
                    retry_badcase_ratio_threshold=float(retry_badcase_ratio_threshold),
                    streaming=False, **kw,
                )
                try:
                    wav, _, _ = next(gen)
                finally:
                    gen.close()
                used.append(getattr(self.tts, "last_successful_seed", seed + i))
                chunks.append(wav.squeeze(0).float().cpu().numpy())
                if i == 0 and cache is None and keep_voice and len(pieces) > 1:
                    # continue from the first piece so the rest keeps its voice
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temps.append(tmp.name)
                    sf.write(tmp.name, chunks[0], self.sample_rate)
                    kw = dict(prompt_text=spoken, prompt_wav_path=tmp.name)
                    if self.is_v2:
                        kw["reference_wav_path"] = tmp.name
                    cache = self.tts.build_prompt_cache(**kw)
                    desc = ""  # the description would be read inside the continued text
        finally:
            for t in temps:
                try:
                    os.unlink(t)
                except OSError:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        retried = any(u != seed + i for i, u in enumerate(used))
        info = [f"seed {seed}" + (f" (retried: {', '.join(map(str, used))})" if retried else ""),
                mode, f"cfg {float(cfg_value):g}", f"{int(inference_timesteps)} steps"]
        self.last_info = " · ".join(info)
        audio = concat(chunks, self.sample_rate, gap_s=float(gap))
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if loudness and peak > 1e-4:
            audio = audio * (0.891 / peak)  # -1 dBFS
        return audio, self.sample_rate


BACKEND = VoxCPMBackend
