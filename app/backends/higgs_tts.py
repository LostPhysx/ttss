"""Boson AI Higgs TTS 3 (bosonai/higgs-tts-3-4b, a.k.a. higgs-audio-v3-tts-4b).

The official runtimes (SGLang-Omni, vLLM-Omni) are Linux server stacks for >= 24 GB GPUs.
This backend runs our own PyTorch port (_higgs_tts_model.py): the Qwen3-4B decoder is
quantized (nf4 or int8) so the whole model fits an 8 GB card, the bundled audio codec runs fp32.

Features: zero-shot voices, voice cloning from a reference clip (+ transcript), inline control
tags (<|emotion:..|>, <|style:..|>, <|prosody:..|>, <|sfx:..|>), multi-speaker scripts.
The open checkpoint has no named preset voices; the "preset" voices below are the official
example reference clips from Boson's repos (downloaded on first use, then cloned).
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np

from .base import Param, RepoInfo, TTSBackend, concat, split_sentences

ROOT = Path(__file__).resolve().parents[2]
VOICE_DIR = ROOT / ".tools" / "voices" / "higgs"

_HIGGS = "https://raw.githubusercontent.com/boson-ai/higgs-audio/05a145bb490501b534563bf51bf2f7aa2326b271/examples/voice_prompts/"
_SGL = "https://raw.githubusercontent.com/sgl-project/sglang-omni/466da750daa587331f19a44aab7c6b6ee3acd1cc/docs/_static/audio/"
# id: (label, url, transcript). All English reference clips; they also speak German (with the
# reference speaker's accent). Movie/TV-character and non-EN/DE clips from the repos are left out.
VOICES = {
    "en_woman": ("Woman, neutral (en_woman)", _HIGGS + "en_woman.wav",
                 "The device would work during the day as well, if you took steps to either block direct sunlight or point it away from the sun."),
    "en_man": ("Man, neutral (en_man)", _HIGGS + "en_man.wav",
               "Maintaining your ability to learn translates into increased marketability, improved career options and higher salaries."),
    "belinda": ("Belinda, warm storyteller", _HIGGS + "belinda.wav",
                "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."),
    "adam": ("Adam, friendly male", _SGL + "male-voice.wav",
             "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time."),
    "teacher": ("Female, clear lecturer", _SGL + "female-voice.wav",
                "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say."),
    "mabel": ("Mabel, British woman", _HIGGS + "mabel.wav",
              "You do talk an awful lot about weather, did you know that? Sometimes I wonder if you're actually content to be a wizard or if you're secretly harbouring a desire to become a seer of the clouds."),
    "chadwick": ("Chadwick, British man", _HIGGS + "chadwick.wav",
                 "Oh dear, who left all this junk lying around? Whoops, there it goes! Mind your pointed little pink head, starfish man."),
    "broom_salesman": ("Old wandmaker, elderly British man", _HIGGS + "broom_salesman.wav",
                       "I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic. And the bond between you and your wand should only grow stronger. Do not be surprised at your new wand's ability to perceive your intentions - particularly in a moment of need."),
    "vex": ("Vex, bored young voice", _HIGGS + "vex.wav",
            "Uhh, this is going to take forever. Why is everything so far?"),
}
RANDOM = "random"

PRECISIONS = {
    "int8": "8-bit (near-lossless, ~4.8 GB VRAM, RTF ~1.6 on an RTX 3070)",
    "nf4": "4-bit nf4 (~4.5x faster, ~3.1 GB VRAM, slightly less faithful)",
}

EMOTIONS = ["elation", "amusement", "enthusiasm", "determination", "pride", "contentment", "affection",
            "relief", "contemplation", "confusion", "surprise", "awe", "longing", "arousal", "anger",
            "fear", "disgust", "bitterness", "sadness", "shame", "helplessness"]
SPEAKER_RE = re.compile(r"^\s*\[\s*(?:speaker|s)\s*[_ ]?(\d+)\s*\]\s*:?\s*", re.I)
TAG_RE = re.compile(r"<\|[^|>]*\|>")
LEAD_RE = re.compile(r"^(?:\s*<\|[^|>]*\|>)*")
POSITIONAL_RE = re.compile(r"<\|(?:sfx:|prosody:pause|prosody:long_pause)")


def _voice_choices():
    out = [{"value": RANDOM, "label": "Random voice (varies with seed)", "group": "Zero-shot"}]
    out += [{"value": k, "label": v[0], "group": "Reference voices (English, also speak German)"}
            for k, v in VOICES.items()]
    return out


class HiggsTTSBackend(TTSBackend):
    name = "higgs_tts"
    description = "Boson AI Higgs TTS 3 (own PyTorch port, nf4/int8 Qwen3 backbone)"
    env = "higgs"
    python = "3.11"
    install = [
        ["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"],
        ["transformers==5.5.4", "bitsandbytes==0.50.2", "safetensors", "tokenizers",
         "soundfile", "huggingface_hub", "numpy"],
    ]

    @classmethod
    def match(cls, info: RepoInfo) -> int:
        if info.model_type == "higgs_multimodal_qwen3":
            return 100
        arch = (info.config.get("architectures") or [""])[0]
        return 90 if arch == "HiggsMultimodalQwen3ForConditionalGeneration" else 0

    # --- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        from huggingface_hub import snapshot_download

        from ._higgs_tts_model import HiggsTTS

        path = snapshot_download(self.repo_id, allow_patterns=[
            "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"])
        self.model = HiggsTTS(path, self.device, precision="int8")
        self.voice_cache: dict[str, tuple] = {}   # voice id / ref file -> (codes, transcript)

    def params(self) -> list[Param]:
        return [
            Param("voice", "Voice (speaker 1)", "select", "en_woman", choices=_voice_choices(),
                  help="Reference voices are short official example clips that get cloned. "
                       "'Random' lets the model invent a voice (changes with the seed; stays the "
                       "same across the pieces of one long text). An uploaded reference clip "
                       "below overrides this."),
            Param("ref_audio", "Reference audio (voice cloning)", "audio",
                  help="5-20 s of clean speech of the voice to clone, any language. Only clone "
                       "voices you have permission to use."),
            Param("ref_text", "Reference transcript", "textarea", "",
                  help="Exactly what is said in the reference clip. Strongly recommended: it "
                       "noticeably improves cloning. Leave empty if unknown."),
            Param("emotion", "Emotion", "select", "",
                  choices=[{"value": "", "label": "(none)"}] + [{"value": e, "label": e} for e in EMOTIONS],
                  help="Prepends <|emotion:...|> to every piece. You can also type control tags "
                       "inline, e.g. '<|emotion:surprise|>Oh! <|sfx:laughter|>Haha, really?'. Tags: "
                       "emotion:*, style:singing|shouting|whispering, prosody:speed_slow|speed_fast|"
                       "pitch_low|pitch_high|pause|long_pause|expressive_high|expressive_low, "
                       "sfx:cough|laughter|sigh|sniff|sneeze|humming|crying|screaming|burping "
                       "(follow each sfx tag with its sound, e.g. '<|sfx:sigh|>Uh,')."),
            Param("style", "Style", "select", "", choices=[
                {"value": "", "label": "(normal)"}, {"value": "whispering", "label": "whispering"},
                {"value": "shouting", "label": "shouting"}, {"value": "singing", "label": "singing"}]),
            Param("speed", "Speed", "select", "", choices=[
                {"value": "speed_very_slow", "label": "very slow (~0.65x)"},
                {"value": "speed_slow", "label": "slow (~0.85x)"},
                {"value": "", "label": "normal"},
                {"value": "speed_fast", "label": "fast (~1.2x)"},
                {"value": "speed_very_fast", "label": "very fast (~1.4x)"}]),
            Param("pitch", "Pitch", "select", "", choices=[
                {"value": "pitch_low", "label": "low (~-3 semitones)"}, {"value": "", "label": "normal"},
                {"value": "pitch_high", "label": "high (~+2.5 semitones)"}]),
            Param("expressive", "Expressiveness", "select", "", choices=[
                {"value": "expressive_low", "label": "flatter"}, {"value": "", "label": "normal"},
                {"value": "expressive_high", "label": "more expressive"}]),
            Param("voice_2", "Voice (speaker 2)", "select", "en_man", choices=_voice_choices(),
                  help="Multi-speaker: start lines with [Speaker_1]: / [Speaker_2]: (or [S1] / [S2]). "
                       "Each turn is generated in that speaker's voice. Speakers 3+ get random "
                       "voices that stay consistent through the script."),
            Param("ref_audio_2", "Reference audio (speaker 2)", "audio",
                  help="Optional clip to clone for speaker 2 (overrides 'Voice (speaker 2)')."),
            Param("ref_text_2", "Reference transcript (speaker 2)", "textarea", ""),
            Param("precision", "Backbone precision", "select", "int8",
                  choices=[{"value": k, "label": v} for k, v in PRECISIONS.items()],
                  help="Quantization of the 4B Qwen3 backbone (bf16 would need ~9.5 GB). "
                       "Changing it reloads the backbone (~10-30 s)."),
            Param("temperature", "Temperature", "number", 0.8, min=0.0, max=2.0, step=0.05,
                  help="Official default 0.8. Lower = steadier, higher = more varied. 0 = greedy."),
            Param("top_k", "Top-k", "number", 50, min=0, max=1026, step=1,
                  help="Sample only from the k most likely codes per codebook (official 50, 0 = off)."),
            Param("top_p", "Top-p", "number", 1.0, min=0.05, max=1.0, step=0.01,
                  help="Nucleus sampling (1.0 = off, the official default; 0.95 is also common)."),
            Param("max_new_tokens", "Max audio tokens per piece", "number", 1024, min=64, max=4096, step=64,
                  help="Safety cap per generated piece; 25 tokens = 1 s of audio (1024 = ~40 s)."),
            Param("seed", "Seed", "number", None, min=0, step=1,
                  help="Empty = random. The seed used is shown with each clip so you can reproduce it."),
            Param("split", "Split text", "select", "sentences", choices=[
                {"value": "none", "label": "No splitting"},
                {"value": "paragraphs", "label": "By paragraph"},
                {"value": "sentences", "label": "By sentence (~250 chars)"}],
                help="Long text is generated piece by piece and joined; each piece re-uses the "
                     "same voice reference, which keeps long outputs stable."),
            Param("gap", "Pause between pieces (s)", "number", 0.2, min=0.0, max=3.0, step=0.05),
            Param("normalize", "Normalize loudness", "checkbox", True,
                  help="Scale the output to a peak of -1 dBFS."),
        ]

    # --- helpers -----------------------------------------------------------
    def _load_wav(self, path: str):
        import soundfile as sf
        import torch

        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            raise ValueError(f"Could not read reference audio ({e}). Use WAV, FLAC, OGG or MP3.")
        wav = torch.from_numpy(data.mean(axis=1))
        # trim leading/trailing silence (< -42 dB) and cap at 30 s (each second costs 25 tokens)
        frame = max(1, sr // 100)
        n = wav.numel() // frame * frame
        if n:
            rms = wav[:n].view(-1, frame).pow(2).mean(1).sqrt()
            active = (rms > 10 ** (-42 / 20)).nonzero().flatten()
            if active.numel():
                pad = sr // 20
                wav = wav[max(0, int(active[0]) * frame - pad):(int(active[-1]) + 1) * frame + pad]
        return wav[: 30 * sr].contiguous(), sr

    def _reference(self, voice: str, ref_audio, ref_text: str):
        """-> (codes or None, transcript or None) for one speaker."""
        if ref_audio:
            import hashlib
            key = "file:" + hashlib.sha1(Path(ref_audio).read_bytes()).hexdigest()
            if key not in self.voice_cache:
                wav, sr = self._load_wav(ref_audio)
                self.voice_cache[key] = (self.model.encode_audio(wav, sr), None)
            return self.voice_cache[key][0], (ref_text or "").strip() or None
        if voice in (None, "", RANDOM):
            return None, None
        if voice not in VOICES:
            raise ValueError(f"Unknown voice {voice!r}")
        if voice not in self.voice_cache:
            import urllib.request
            label, url, transcript = VOICES[voice]
            f = VOICE_DIR / f"{voice}.wav"
            if not f.exists():
                VOICE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(f)
            wav, sr = self._load_wav(str(f))
            self.voice_cache[voice] = (self.model.encode_audio(wav, sr), transcript)
        return self.voice_cache[voice]

    @staticmethod
    def _pieces(text: str, mode: str) -> list[str]:
        if mode == "sentences":
            out = []
            for para in re.split(r"\n\s*\n", text):
                if para.strip():
                    out += split_sentences(" ".join(para.split()), max_chars=250)
            return out
        if mode == "paragraphs":
            return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [" ".join(text.split())]

    @staticmethod
    def _turns(text: str) -> list[tuple[int, str]]:
        """Split a [Speaker_N]: script into (speaker, text) turns; plain text = speaker 1."""
        turns: list[tuple[int, str]] = []
        for line in text.splitlines():
            m = SPEAKER_RE.match(line)
            if m:
                turns.append((int(m.group(1)), line[m.end():]))
            elif turns:
                turns[-1] = (turns[-1][0], turns[-1][1] + "\n" + line)
            elif line.strip():
                turns.append((1, line))
        return [(s, t.strip()) for s, t in turns if t.strip()] or [(1, text.strip())]

    # --- inference ---------------------------------------------------------
    def synthesize(self, text: str, voice: str = "en_woman", ref_audio=None, ref_text: str = "",
                   emotion: str = "", style: str = "", speed: str = "", pitch: str = "",
                   expressive: str = "", voice_2: str = "en_man", ref_audio_2=None, ref_text_2: str = "",
                   precision: str = "int8", temperature: float = 0.8, top_k: int = 50, top_p: float = 1.0,
                   max_new_tokens: int = 1024, seed=None, split: str = "sentences", gap: float = 0.2,
                   normalize: bool = True, **_):
        import torch

        if not text or not text.strip():
            raise ValueError("Text is empty.")
        if precision not in PRECISIONS:
            raise ValueError(f"Unknown precision {precision!r}")
        try:
            self.model.set_precision(precision)
        except torch.OutOfMemoryError:
            self.model.lm = None
            torch.cuda.empty_cache()
            self.model.set_precision("nf4")
            raise RuntimeError(f"Not enough free VRAM for {PRECISIONS[precision]}. Switched back to nf4.")

        seed = int(seed) if seed not in (None, "") else random.randrange(2**31)
        prefix = "".join(f"<|{k}|>" for k in (
            f"emotion:{emotion}" if emotion else "", f"style:{style}" if style else "",
            f"prosody:{speed}" if speed else "", f"prosody:{pitch}" if pitch else "",
            f"prosody:{expressive}" if expressive else "") if k)

        turns = self._turns(text)
        multi = len({s for s, _ in turns}) > 1
        refs: dict[int, tuple] = {1: self._reference(voice, ref_audio, ref_text)}
        if multi:
            refs[2] = self._reference(voice_2, ref_audio_2, ref_text_2)

        chunks, sr, n_piece, truncated = [], 24000, 0, 0
        gen_kw = dict(temperature=float(temperature), top_p=float(top_p) if top_p not in (None, "") else 1.0,
                      top_k=int(top_k or 0), max_new_tokens=int(max_new_tokens))
        for t_i, (spk, turn) in enumerate(turns):
            turn_audio = []
            # delivery tags at the start of a turn (emotion/style/speed...) apply to all its pieces
            lead = "".join(t for t in TAG_RE.findall(LEAD_RE.match(turn).group(0))
                           if not POSITIONAL_RE.match(t))
            for i, piece in enumerate(self._pieces(turn, split)):
                if not TAG_RE.sub("", piece).strip():
                    continue
                codes_ref, txt_ref = refs.get(spk, (None, None))
                if i and not piece.startswith("<|"):
                    piece = lead + piece
                codes, stopped = self.model.generate(prefix + piece, codes_ref, txt_ref,
                                                     seed=seed + n_piece, **gen_kw)
                n_piece += 1
                truncated += not stopped
                if len(codes) == 0:
                    continue
                if codes_ref is None and stopped:
                    # random voice: re-use the first generated piece as this speaker's reference
                    # so the voice stays the same for the rest of the text
                    refs[spk] = (codes[: 30 * 25], TAG_RE.sub("", piece).strip())
                turn_audio.append(self.model.decode_audio(codes).numpy())
            if turn_audio:
                chunks.append(concat(turn_audio, sr, gap_s=float(gap)))
        if not chunks:
            raise RuntimeError("The model produced no audio.")
        audio = concat(chunks, sr, gap_s=max(float(gap), 0.3) if multi else float(gap))
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if normalize and peak > 1e-4:
            audio = audio * (0.891 / peak)  # -1 dBFS

        who = "clone" if ref_audio else voice
        info = [f"seed {seed}", who if not multi else f"{len(turns)} turns", precision]
        if truncated:
            info.append(f"{truncated} piece(s) hit the token cap")
        self.last_info = " · ".join(info)
        return audio.astype(np.float32), sr


BACKEND = HiggsTTSBackend
