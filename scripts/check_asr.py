"""Transcribe WAV files with Whisper to check that generated speech says the right thing.

    .tools\\venv\\Scripts\\python.exe scripts\\gpulock.py -- .tools\\venv\\Scripts\\python.exe scripts\\check_asr.py <lang> file.wav [...]
lang: en | de
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".tools" / "hf-home"))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from transformers import pipeline  # noqa: E402

lang = {"en": "english", "de": "german"}[sys.argv[1]]
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cuda", dtype=torch.float16)
for f in sys.argv[2:]:
    wav, sr = sf.read(f, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    g = np.gcd(sr, 16000)
    w16 = resample_poly(wav, 16000 // g, sr // g)
    out = asr({"raw": w16, "sampling_rate": 16000}, generate_kwargs={"language": lang, "task": "transcribe"})
    print(f"{Path(f).name} ({len(wav) / sr:.1f}s): {out['text'].strip()}")
