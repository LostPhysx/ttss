"""k2-fsa OmniVoice (k2-fsa/OmniVoice): 0.6B diffusion-LM zero-shot TTS, 600+ languages.

Three modes, all through the official `omnivoice` package (model.generate):
  * voice cloning   - a reference clip (+ optional transcript) is given
  * voice design    - speaker attributes (gender, age, pitch, accent, whisper) as `instruct`
  * auto voice      - neither: the model picks a random voice (the seed decides which)
Attributes can be combined with a reference clip (they then stabilise the clone).
Long text is split by the package itself (audio_chunk_*); without a reference the first
chunk becomes the voice reference for the rest, so the voice stays the same.
"""
from __future__ import annotations

import random

import numpy as np

from .base import Param, RepoInfo, TTSBackend

LANGUAGES = [
    {"value": "auto", "label": "Auto-detect"},
    {"value": "en", "label": "English"},
    {"value": "de", "label": "German"},
]
AUTO = {"value": "", "label": "Auto (model decides)"}
GENDERS = ["male", "female"]
AGES = ["child", "teenager", "young adult", "middle-aged", "elderly"]
PITCHES = ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"]
ACCENTS = ["american accent", "british accent", "australian accent", "canadian accent",
           "indian accent", "chinese accent", "korean accent", "japanese accent",
           "portuguese accent", "russian accent"]
NONVERBAL = ("[laughter] [sigh] [confirmation-en] [question-en] [question-ah] [question-oh] "
             "[surprise-ah] [surprise-oh] [dissatisfaction-hnn]")


def _choices(values: list[str]) -> list:
    return [AUTO] + [{"value": v, "label": v} for v in values]


class OmniVoiceBackend(TTSBackend):
    name = "omnivoice"
    description = "k2-fsa OmniVoice (voice cloning, voice design, 600+ languages)"
    env = "omnivoice"
    python = "3.11"
    install = [
        # omnivoice is tested with torch 2.8 (its uv constraint); pin the CUDA build first
        ["torch==2.8.0", "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"],
        ["omnivoice==0.2.1", "num2words"],
    ]

    @classmethod
    def match(cls, info: RepoInfo) -> int:
        if info.model_type == "omnivoice" or (info.config.get("architectures") or [None])[0] == "OmniVoice":
            return 90
        if info.library_name == "omnivoice":
            return 80
        return 0

    def load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(self.repo_id, device_map=self.device, dtype=torch.float16)
        self.sr = int(self.model.sampling_rate)
        self._prompt_key = None
        self._prompt = None

    def params(self) -> list[Param]:
        return [
            Param("language", "Language", "select", "auto", choices=LANGUAGES,
                  help="Language of the text. Auto works, setting it explicitly is slightly better."),
            Param("ref_audio", "Reference voice (clone)", "audio",
                  help="Optional 3-10 s clip of the voice to clone. Without it the voice comes from "
                       "the attributes below, or is random (pick a seed you like to keep it). "
                       "Use a clip in the same language as the text, otherwise it keeps the accent."),
            Param("ref_text", "Reference transcript", "textarea", "",
                  help="What is said in the reference clip. Empty = transcribed automatically "
                       "with Whisper large-v3-turbo (first use downloads ~1.6 GB and needs ~1.6 GB VRAM)."),
            Param("gender", "Gender", "select", "", choices=_choices(GENDERS),
                  help="Voice design attribute. All attributes are optional and can be combined."),
            Param("age", "Age", "select", "", choices=_choices(AGES)),
            Param("pitch", "Pitch", "select", "", choices=_choices(PITCHES)),
            Param("accent", "English accent", "select", "", choices=_choices(ACCENTS),
                  help="Only has an effect on English text."),
            Param("whisper", "Whisper", "checkbox", False, help="Whispering voice (design attribute)."),
            Param("instruct", "Extra attributes", "text", "",
                  help="Optional, comma separated, e.g. 'female, elderly, low pitch'. Added to the "
                       "attributes above. Voice design was trained on English/Chinese only, so it "
                       "can be less stable for German. Non-verbal tags in the text also work: " + NONVERBAL),
            Param("speed", "Speed", "number", 1.0, min=0.5, max=1.5, step=0.05,
                  help=">1 faster, <1 slower. Ignored when a duration is set."),
            Param("duration", "Fixed duration (s)", "number", None, min=0.5, max=600, step=0.5,
                  help="Empty = the model estimates the length from the text. Set to force a length."),
            Param("num_step", "Steps", "number", 32, min=4, max=64, step=1,
                  help="Iterative unmasking steps. Official default 32; 16 is about twice as fast."),
            Param("guidance_scale", "Guidance (CFG)", "number", 2.0, min=0.0, max=4.0, step=0.1,
                  help="Classifier-free guidance. Official default 2.0."),
            Param("t_shift", "Time-step shift", "number", 0.1, min=0.01, max=1.0, step=0.01,
                  help="Noise-schedule shift. Smaller values emphasise the early steps. Default 0.1."),
            Param("position_temperature", "Position temperature", "number", 5.0, min=0.0, max=20.0, step=0.5,
                  help="Randomness of which positions get unmasked first. 0 = deterministic. Default 5."),
            Param("class_temperature", "Token temperature", "number", 0.0, min=0.0, max=2.0, step=0.05,
                  help="Randomness when sampling tokens. 0 = greedy (default)."),
            Param("layer_penalty_factor", "Layer penalty", "number", 5.0, min=0.0, max=20.0, step=0.5,
                  help="Makes the lower codebook layers unmask first. Default 5."),
            Param("denoise", "Denoise", "checkbox", True,
                  help="Ask the model for clean speech (no background noise from the reference)."),
            Param("seed", "Seed", "number", None, min=0, step=1,
                  help="Empty = random. The seed used is shown with each clip. With auto voice it "
                       "also picks the voice."),
            Param("normalize_text", "Numbers to words", "checkbox", False,
                  help="Spell out numbers (e.g. 2345) before synthesis, using num2words in the "
                       "selected language (English if Auto)."),
            Param("audio_chunk_duration", "Chunk length (s)", "number", 15.0, min=5, max=60, step=1,
                  help="Long text is split into chunks of about this many seconds of audio."),
            Param("audio_chunk_threshold", "Chunk above (s)", "number", 30.0, min=5, max=120, step=1,
                  help="Splitting only starts when the estimated audio is longer than this."),
            Param("preprocess_prompt", "Clean reference", "checkbox", True,
                  help="Remove long silences from the reference clip and add end punctuation to "
                       "its transcript."),
            Param("postprocess_output", "Trim silences", "checkbox", True,
                  help="Remove long silences from the generated audio."),
            Param("normalize", "Normalize loudness", "checkbox", False,
                  help="Scale the output to a peak of -1 dBFS. Off = loudness follows the reference."),
        ]

    def _normalize_text(self, text: str, lang: str | None) -> str:
        from omnivoice.utils import text as T
        code = lang or "en"
        try:
            return T.normalize_text(text, code)
        except ImportError:  # English normaliser (WeTextProcessing/pynini) has no Windows wheel
            return T._apply_with_protection(text, lambda s: T._num2words_segment(s, code),
                                            protect_pinyin=False)

    def _clone_prompt(self, ref_audio: str, ref_text: str | None, preprocess: bool):
        import hashlib
        with open(ref_audio, "rb") as f:
            key = (hashlib.sha1(f.read()).hexdigest(), ref_text, preprocess)
        if key != self._prompt_key:
            self._prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, preprocess_prompt=preprocess)
            self._prompt_key = key
            if self.model._asr_pipe is not None:  # free Whisper (~1.6 GB VRAM) after auto-transcribing
                import torch
                self.model._asr_pipe = None
                torch.cuda.empty_cache()
        return self._prompt

    def synthesize(self, text: str, language: str = "auto", ref_audio: str | None = None,
                   ref_text: str = "", gender: str = "", age: str = "", pitch: str = "",
                   accent: str = "", whisper: bool = False, instruct: str = "", speed=1.0,
                   duration=None, num_step=32, guidance_scale=2.0, t_shift=0.1,
                   position_temperature=5.0, class_temperature=0.0, layer_penalty_factor=5.0,
                   denoise: bool = True, seed=None, normalize_text: bool = False,
                   audio_chunk_duration=15.0, audio_chunk_threshold=30.0,
                   preprocess_prompt: bool = True, postprocess_output: bool = True,
                   normalize: bool = False, **_):
        import torch
        from omnivoice import OmniVoiceGenerationConfig

        lang = None if language in (None, "", "auto") else language
        text = text.strip()
        if normalize_text:
            text = self._normalize_text(text, lang)

        attrs = [a for a in (gender, age, pitch, accent) if a]
        if whisper:
            attrs.append("whisper")
        attrs += [a.strip() for a in (instruct or "").replace("，", ",").split(",") if a.strip()]
        instruct_str = ", ".join(dict.fromkeys(attrs)) or None

        cfg = OmniVoiceGenerationConfig(
            num_step=int(num_step), guidance_scale=float(guidance_scale), t_shift=float(t_shift),
            layer_penalty_factor=float(layer_penalty_factor),
            position_temperature=float(position_temperature),
            class_temperature=float(class_temperature), denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt), postprocess_output=bool(postprocess_output),
            audio_chunk_duration=float(audio_chunk_duration),
            audio_chunk_threshold=float(audio_chunk_threshold),
        )
        kw = dict(text=text, language=lang, instruct=instruct_str, generation_config=cfg)
        if duration not in (None, "") and float(duration) > 0:
            kw["duration"] = float(duration)
        elif speed not in (None, "") and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if ref_audio:
            kw["voice_clone_prompt"] = self._clone_prompt(
                ref_audio, (ref_text or "").strip() or None, bool(preprocess_prompt))

        seed = int(seed) if seed not in (None, "") else random.randrange(2**31)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        with torch.inference_mode():
            audio = self.model.generate(**kw)[0]
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        mode = "clone" if ref_audio else ("design" if instruct_str else "auto voice")
        info = [f"seed {seed}", mode]
        if ref_audio and not (ref_text or "").strip():
            t = kw["voice_clone_prompt"].ref_text
            info.append(f'heard "{t[:60]}{"..." if len(t) > 60 else ""}"')
        if instruct_str:
            info.append(instruct_str)
        if lang:
            info.append(lang)
        self.last_info = " · ".join(info)

        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if normalize and peak > 1e-4:
            audio = audio * (0.891 / peak)  # -1 dBFS
        return audio, self.sr


BACKEND = OmniVoiceBackend
