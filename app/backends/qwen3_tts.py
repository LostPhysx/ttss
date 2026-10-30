"""Qwen3-TTS 12Hz (Qwen/Qwen3-TTS-12Hz-1.7B-* and 0.6B-*).

One backend for all three checkpoint kinds; config.json's `tts_model_type` says which:
  * custom_voice  -> preset speakers + optional style instruction (instruction: 1.7B only)
  * base          -> voice cloning from a reference clip (+ its transcript)
  * voice_design  -> voice created from a natural-language description

Inference uses `faster-qwen3-tts` (CUDA graphs around the official model, ~8-10x faster than
plain HF generate, which is launch-overhead bound: 16 tiny forward passes per 80 ms frame).
It wraps the official `qwen_tts` package (as `qwen-tts-hf`, the Transformers-5 build), so the
original generate path stays available as the "reference" engine. bf16 + SDPA (no flash-attn
wheels on Windows), ~4.3 GB VRAM for the 1.7B models.
"""
from __future__ import annotations

import hashlib
import random
import re

import numpy as np

from .base import Param, RepoInfo, TTSBackend, concat, split_sentences

# Only English and German are exposed (the model also does zh/ja/ko/fr/ru/pt/es/it).
LANGUAGES = [
    {"value": "Auto", "label": "Auto-detect"},
    {"value": "English", "label": "English"},
    {"value": "German", "label": "German"},
]
# CustomVoice preset speakers (from the model card). None of them is a native German speaker,
# but every speaker can speak every supported language. The Beijing/Sichuan dialect speakers
# (Dylan, Eric) are left out.
SPEAKERS = [
    ("ryan", "Ryan (male, dynamic, strong rhythm)", "English native"),
    ("aiden", "Aiden (male, sunny American, clear)", "English native"),
    ("serena", "Serena (female, warm, gentle)", "Other native language (accent possible)"),
    ("vivian", "Vivian (female, bright, slightly edgy)", "Other native language (accent possible)"),
    ("ono_anna", "Ono Anna (female, playful, light)", "Other native language (accent possible)"),
    ("sohee", "Sohee (female, warm, emotional)", "Other native language (accent possible)"),
    ("uncle_fu", "Uncle Fu (male, low, mellow, older)", "Other native language (accent possible)"),
]
DEFAULT_DESIGN = ("A warm, clear female voice in her thirties, speaking at a calm, moderate pace "
                  "with a friendly tone.")
TOKENS_PER_SECOND = 12.5  # codec frame rate
MAX_SEQ_LEN = 2048        # static KV cache of the CUDA-graph talker (prompt + generated frames)


class Qwen3TTSBackend(TTSBackend):
    name = "qwen3_tts"
    description = "Qwen3-TTS (preset speakers + style, voice cloning, voice design)"
    env = "qwen3tts"
    python = "3.12"
    install = [
        ["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"],
        ["faster-qwen3-tts==0.5.4", "transformers==5.15.1"],
    ]

    @classmethod
    def match(cls, info: RepoInfo) -> int:
        if info.model_type == "qwen3_tts":
            return 100
        if "qwen3_tts" in info.tags or info.library_name == "qwen-tts":
            return 80
        return 0

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.fq = FasterQwen3TTS.from_pretrained(
            self.repo_id, device="cuda" if self.device == "cuda" else self.device,
            dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=MAX_SEQ_LEN)
        self.tts = self.fq.model                            # official qwen_tts.Qwen3TTSModel
        self.kind = self.tts.model.tts_model_type          # custom_voice | base | voice_design
        self.size = self.tts.model.tts_model_size          # 1b7 | 0b6
        supported = set(self.tts.get_supported_speakers() or [])
        self.speakers = [s for s in SPEAKERS if not supported or s[0] in supported]
        self._ref_hash = None
        self._clone_cache = None  # (key, prompt items) for the reference engine
        self._warmup()

    def _warmup(self) -> None:
        """Capture the CUDA graphs now, then run one short throwaway generation.
        Without this, the first request triggers graph capture mid-generation and comes out
        as garbled speech (seen with faster-qwen3-tts 0.5.4 / torch 2.11); later ones are fine."""
        self.fq.warmup(prefill_len=100)
        kw = dict(text="Hello there.", language="English", max_new_tokens=25)
        if self.kind == "custom_voice":
            self.fq.generate_custom_voice(speaker=self.speakers[0][0], **kw)
        elif self.kind == "voice_design":
            self.fq.generate_voice_design(instruct=DEFAULT_DESIGN, **kw)
        elif self.kind == "base":
            sr = 24000
            t = np.arange(sr * 2) / sr  # 2 s synthetic tone; only the speaker embedding is used
            items = self.tts.create_voice_clone_prompt(
                ref_audio=((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr),
                x_vector_only_mode=True)
            vcp = self.tts._prompt_items_to_voice_clone_prompt(items)
            self.fq.generate_voice_clone(voice_clone_prompt=vcp, **kw)

    # --- UI ----------------------------------------------------------------
    def params(self) -> list[Param]:
        kind = getattr(self, "kind", "custom_voice")
        ps: list[Param] = []
        if kind == "custom_voice":
            spk = getattr(self, "speakers", SPEAKERS)
            ps.append(Param("speaker", "Speaker", "select", "ryan",
                            choices=[{"value": v, "label": l, "group": g} for v, l, g in spk],
                            help="Preset voices. Ryan and Aiden are native English speakers; there is no "
                                 "native German preset, but every speaker can speak German."))
            if getattr(self, "size", "1b7") != "0b6":
                ps.append(Param("instruct", "Style instruction", "textarea", "",
                                help="Optional natural-language direction for tone, emotion and pace, e.g. "
                                     "'Speak slowly and calmly.' or 'Very excited, almost shouting.' "
                                     "Empty = neutral. (This is also how to change speed.)"))
        elif kind == "voice_design":
            ps.append(Param("instruct", "Voice description", "textarea", DEFAULT_DESIGN,
                            help="Describe the voice: gender, age, pitch, timbre, accent, emotion, pace. "
                                 "The voice is re-invented on every generation; fix the seed to keep it."))
        elif kind == "base":
            ps += [
                Param("ref_audio", "Reference audio", "audio",
                      help="Clip of the voice to clone (about 3-20 s of clean speech works best)."),
                Param("ref_text", "Reference transcript", "textarea", "",
                      help="Exact words spoken in the reference clip. Strongly improves similarity. "
                           "If empty, speaker-embedding-only mode is used."),
                Param("x_vector_only", "Speaker embedding only", "checkbox", False,
                      help="Ignore the transcript and clone only the timbre from a speaker embedding. "
                           "Less similar, but no transcript needed and cleaner when the reference is in "
                           "another language than the text."),
            ]
        ps += [
            Param("language", "Language", "select", "Auto", choices=LANGUAGES,
                  help="Language of the text. Setting it explicitly is recommended by Qwen."),
            Param("temperature", "Temperature", "number", 0.9, min=0.05, max=2.0, step=0.05,
                  help="Sampling temperature (official default 0.9). Lower = steadier/flatter, "
                       "higher = more expressive/varied."),
            Param("top_k", "Top-k", "number", 50, min=0, max=3072, step=1,
                  help="Sample only from the k most likely codec tokens (official 50; 0 = off)."),
            Param("top_p", "Top-p", "number", 1.0, min=0.05, max=1.0, step=0.01,
                  help="Nucleus sampling cutoff (official 1.0 = off)."),
            Param("repetition_penalty", "Repetition penalty", "number", 1.05, min=1.0, max=2.0, step=0.01,
                  help="Discourages repeated codec tokens (official 1.05). Raise slightly if audio loops."),
            Param("do_sample", "Sampling", "checkbox", True,
                  help="Off = greedy decoding (deterministic, but tends to sound monotone or loop)."),
            Param("seed", "Seed", "number", None, min=0, step=1,
                  help="Empty = random. The seed used is shown with each clip so you can reproduce it."),
            Param("split", "Split text", "select", "sentences", choices=[
                {"value": "none", "label": "No splitting"},
                {"value": "paragraphs", "label": "By paragraph"},
                {"value": "sentences", "label": "By sentence (~300 chars)"}],
                help="Long text is generated piece by piece and joined (same voice and seed for every "
                     "piece). Pieces are limited to ~150 s of audio by the fast engine's cache."),
            Param("gap", "Pause between pieces (s)", "number", 0.2, min=0.0, max=3.0, step=0.05),
            Param("max_seconds", "Max seconds per piece", "number", 120, min=5, max=150, step=5,
                  help="Safety cap if the model doesn't stop on its own."),
            Param("normalize", "Normalize loudness", "checkbox", True,
                  help="Scale the output to a peak of -1 dBFS."),
            Param("engine", "Engine", "select", "fast", choices=[
                {"value": "fast", "label": "Fast (CUDA graphs)"},
                {"value": "reference", "label": "Reference (official HF generate, ~8x slower)"}],
                help="Both run the same model. 'Reference' is the unmodified official code path and is "
                     "the only one that uses the sub-talker settings below."),
            Param("subtalker_temperature", "Sub-talker temperature", "number", 0.9, min=0.05, max=2.0,
                  step=0.05, help="Reference engine only: temperature of the code predictor that fills in "
                                  "the 15 fine acoustic codebooks of each frame (official 0.9; the fast "
                                  "engine always uses 0.9 / top-k 50 / top-p 1.0)."),
            Param("subtalker_top_k", "Sub-talker top-k", "number", 50, min=0, max=2048, step=1,
                  help="Reference engine only (official 50)."),
            Param("subtalker_top_p", "Sub-talker top-p", "number", 1.0, min=0.05, max=1.0, step=0.01,
                  help="Reference engine only (official 1.0)."),
        ]
        return ps

    # --- inference -----------------------------------------------------------
    @staticmethod
    def _pieces(text: str, mode: str) -> list[str]:
        if mode == "sentences":
            return split_sentences(text, max_chars=300)
        if mode == "paragraphs":
            return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]
        return [" ".join(text.split())]

    def _check_ref(self, ref_audio) -> str:
        """Hash the reference clip; drop cached clone prompts when the clip changes
        (faster-qwen3-tts caches by file path, and upload temp paths may be reused)."""
        if not ref_audio:
            raise ValueError("Upload a reference audio clip to clone a voice.")
        with open(ref_audio, "rb") as f:
            h = hashlib.sha1(f.read()).hexdigest()
        if h != self._ref_hash:
            getattr(self.fq, "_voice_prompt_cache", {}).clear()
            self._ref_hash = h
        return h

    def synthesize(self, text: str, speaker: str = "ryan", instruct: str = "", ref_audio=None,
                   ref_text: str = "", x_vector_only: bool = False, language: str = "Auto",
                   temperature: float = 0.9, top_k: int = 50, top_p: float = 1.0,
                   repetition_penalty: float = 1.05, do_sample: bool = True, seed=None,
                   split: str = "sentences", gap: float = 0.2, max_seconds: float = 120,
                   normalize: bool = True, engine: str = "fast", subtalker_temperature: float = 0.9,
                   subtalker_top_k: int = 50, subtalker_top_p: float = 1.0, **_):
        import torch

        if language not in {c["value"] for c in LANGUAGES}:
            raise ValueError(f"Unsupported language {language!r}")
        fast = engine != "reference"
        seed = int(seed) if seed not in (None, "") else random.randrange(2**31)
        max_new = int(min(float(max_seconds), 150) * TOKENS_PER_SECOND)
        gen = dict(do_sample=bool(do_sample), temperature=float(temperature), top_k=int(top_k),
                   top_p=float(top_p), repetition_penalty=float(repetition_penalty),
                   max_new_tokens=max(25, max_new))
        if not fast:
            gen.update(subtalker_dosample=bool(do_sample), subtalker_temperature=float(subtalker_temperature),
                       subtalker_top_k=int(subtalker_top_k), subtalker_top_p=float(subtalker_top_p))
        instruct = (instruct or "").strip()
        ref_text = (ref_text or "").strip()
        api = self.fq if fast else self.tts

        if self.kind == "custom_voice":
            if speaker not in {s[0] for s in self.speakers}:
                raise ValueError(f"Unknown speaker {speaker!r}")
            label = next(l for v, l, _g in self.speakers if v == speaker).split(" (")[0]
            run = lambda t: api.generate_custom_voice(  # noqa: E731
                text=t, speaker=speaker, language=language, instruct=instruct or None, **gen)
        elif self.kind == "voice_design":
            if not instruct:
                raise ValueError("Enter a voice description.")
            label = "designed voice"
            run = lambda t: api.generate_voice_design(  # noqa: E731
                text=t, instruct=instruct, language=language, **gen)
        elif self.kind == "base":
            xvec = bool(x_vector_only) or not ref_text
            h = self._check_ref(ref_audio)
            label = "clone (embedding only)" if xvec else "clone"
            if fast:
                run = lambda t: self.fq.generate_voice_clone(  # noqa: E731
                    text=t, language=language, ref_audio=ref_audio, ref_text=ref_text,
                    xvec_only=xvec, **gen)
            else:
                key = (h, ref_text, xvec)
                if self._clone_cache is None or self._clone_cache[0] != key:
                    self._clone_cache = (key, self.tts.create_voice_clone_prompt(
                        ref_audio=ref_audio, ref_text=ref_text or None, x_vector_only_mode=xvec))
                prompt = self._clone_cache[1]
                run = lambda t: self.tts.generate_voice_clone(  # noqa: E731
                    text=t, language=language, voice_clone_prompt=prompt, **gen)
        else:
            raise RuntimeError(f"Unsupported Qwen3-TTS model type {self.kind!r}")

        chunks, sr = [], 24000
        for piece in self._pieces(text, split):
            torch.manual_seed(seed)  # same seed per piece keeps a designed voice consistent
            wavs, sr = run(piece)
            w = wavs[0]
            if hasattr(w, "detach"):
                w = w.detach().float().cpu().numpy()
            chunks.append(np.asarray(w, dtype=np.float32).reshape(-1))
        audio = concat(chunks, int(sr), gap_s=float(gap))
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if normalize and peak > 1e-4:
            audio = audio * (0.891 / peak)  # -1 dBFS
        self.last_info = f"seed {seed} · {label} · {language}" + ("" if fast else " · reference engine")
        return audio, int(sr)


BACKEND = Qwen3TTSBackend
