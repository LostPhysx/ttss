# TTSS: local text-to-speech on your NVIDIA GPU

A small web UI for running current open TTS models locally on a Windows PC with an 8 GB NVIDIA GPU.
Type text, press **Generate**, and listen right away. Switch models from a dropdown, or paste any
Hugging Face model URL. The UI works on your phone too.

- **Model-agnostic.** Each model family is a small *backend* plugin. The right one is picked
  automatically from the Hugging Face repo.
- **Isolated dependencies.** Each backend gets its own Python environment, so models that need
  conflicting library versions can live side by side.
- **Fits 8 GB cards.** Models too big for bf16 run with quantized (8-bit / 4-bit) language-model
  backbones. Two models that only ship Linux/vLLM runtimes upstream have their own plain-PyTorch
  ports here.
- **Self-contained.** uv, Python, every environment and every model download live in `.tools/`
  inside the repo. Nothing is installed globally. Delete the folder and it's gone.

## Models included

All were tested in English and German on an RTX 3070 (8 GB), with output checked by Whisper.

| Model | Backend | VRAM | Speed (RTF*) | Voices / control | Weights license |
|---|---|---|---|---|---|
| [Qwen3-TTS 1.7B CustomVoice / Base / VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | `qwen3_tts` | ~4.5 GB | ~0.45 | 7 preset speakers + style instructions; cloning (Base); voice from description (VoiceDesign) | Apache-2.0 |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | `omnivoice` | ~2 GB | ~0.45 | cloning, voice design (gender, age, pitch, accent), non-verbal tags | CC BY-NC |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `voxcpm` | ~5.5 GB | ~2 | cloning (4 modes), voice description | Apache-2.0 |
| [Higgs TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | `higgs_tts` | ~5 GB (int8) | ~1.8 int8 / ~0.35 nf4 | 9 example voices, cloning, emotion/style tags, 2 speakers | Boson non-commercial |
| [Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | `voxtral_tts` | ~2.7 GB (nf4) | ~0.4 nf4 / ~0.9 int8 | 20 preset voices (English + German exposed) | CC BY-NC 4.0 |
| Anything that runs in the `transformers` TTS pipeline, e.g. [MMS-TTS](https://huggingface.co/facebook/mms-tts-eng) | `hf_pipeline` | small | fast | – | per model |

\*RTF = seconds of compute per second of audio (below 1 = faster than real time). Measured on an
RTX 3070. Every model's settings (sampling, guidance, seed, text splitting, …) are shown in the UI.

## Requirements

- Windows 10/11, x64
- NVIDIA GPU with 8 GB+ VRAM and a recent driver (CUDA 12.8 support, driver 570 or newer)
- ~60 GB of free disk space if you want every model (weights + environments); ~10 GB for one model
- Internet access for the first setup and model downloads

## Quick start

```powershell
git clone https://github.com/LostPhysx/ttss.git
cd ttss
powershell -ExecutionPolicy Bypass -File setup.ps1     # portable uv + Python + main env (~3 GB)
.tools\venv\Scripts\python.exe scripts\prepare.py      # optional: build all model envs + download all weights
run.cmd                                                # starts the server and opens http://127.0.0.1:7860
```

If you skip `prepare.py`, a model's environment and weights are fetched the first time you load it
in the UI. The status line shows progress, and the first load can take many minutes.

Prepare only some models:

```powershell
.tools\venv\Scripts\python.exe scripts\prepare.py --list
.tools\venv\Scripts\python.exe scripts\prepare.py k2-fsa/OmniVoice Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
.tools\venv\Scripts\python.exe scripts\prepare.py --smoke     # also generate one test clip per model
```

For gated Hugging Face models, set `HF_TOKEN` before running `prepare.py` or `run.cmd`.

## Using it

- **Model**: pick a model from the dropdown and press **Load**. Only one model is in VRAM at a time.
  Loading another one replaces it, and **Unload** frees the GPU completely.
- **Add a model from Hugging Face**: paste a URL or `org/name`. The matching backend is detected
  automatically, or you can choose one. Models are remembered in `models.json`.
- **Settings** appear per model (voice, reference audio for cloning, sampling, seed, …). Hover a
  setting for an explanation. Each clip shows the seed it used, so you can reproduce a take.
- **Samples**: short English text, plus a ~30 s English and German adventure passage.
- Every clip is also saved to `outputs/`.

### From your phone

`run.cmd` listens on your local network and prints the address (e.g. `http://192.168.1.20:7860`).
Allow the port once in an **admin** PowerShell (private networks and your own subnet only):

```powershell
New-NetFirewallRule -DisplayName 'TTSS web UI (TCP 7860, LAN only)' -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow -Profile Private -RemoteAddress LocalSubnet
```

There is no login: anyone on your network can use the UI. For this PC only, run
`set HOST=127.0.0.1` before `run.cmd`. Use `PORT` to change the port.

## How it works

```
browser ──HTTP──▶ app/server.py         (main env, no GPU)
                       │ local authenticated socket
                       ▼
                  app/worker.py         (backend's own env in .tools/envs/<env>, owns the GPU)
                       │
                  app/backends/<name>.py  load() / params() / synthesize()
```

| Path | What |
|---|---|
| `app/server.py` | FastAPI server, model state, starts and stops workers |
| `app/worker.py` | runs one backend in a subprocess; `--test` mode for development |
| `app/envs.py` | builds per-backend envs with the portable uv |
| `app/backends/` | one module per model family (`base.py` = the contract) |
| `app/static/index.html` | the whole UI (plain HTML/CSS/JS, no build step) |
| `models.default.json` | models shown in the dropdown on a fresh install, used by `prepare.py` |
| `scripts/prepare.py` | build envs + download weights (+ optional smoke test) |
| `scripts/gpulock.py`, `scripts/check_asr.py` | dev helpers: exclusive GPU lock, Whisper transcript check |

## Adding a model

See [CLAUDE.md](CLAUDE.md) for the backend contract and the development/test loop. It's written
for AI coding agents, but it works just as well for people. In short: add
`app/backends/<name>.py` with a `match()` rule, an `env` + `install` spec, and
`load()` / `params()` / `synthesize()`. The UI picks it up automatically.

## Troubleshooting

- **A load fails**: read `.tools/logs/worker-<backend>.log` (runtime) and `.tools/logs/env-<env>.log`
  (installation).
- **Out of VRAM**: close other GPU apps, or pick a smaller precision in the model's settings
  (Voxtral, Higgs).
- **The first load of a model is very slow**: the environment is being built and the weights
  downloaded. Run `prepare.py` beforehand to get that out of the way.
- **Rebuild a model's env**: `.tools\venv\Scripts\python.exe -m app.envs <backend> --force`.

## License

The code is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). It includes ports of
Apache-2.0 code from vLLM-Omni and SGLang-Omni. Model weights are **not** included. They are
downloaded from Hugging Face and keep their own licenses, several of which are non-commercial.
