# CLAUDE.md: guide for AI agents working on TTSS

TTSS is a local TTS web UI for Windows + an NVIDIA GPU (developed on an RTX 3070, 8 GB). The read
order for orientation is README.md, then this file, then `app/backends/base.py`.

## Hard rules

1. **No global installs.** Never `pip install` into a system Python, and never use winget/choco/npm -g,
   etc. Everything lives under `.tools/` in the repo:
   `.tools/uv/uv.exe` (portable uv), `.tools/python/` (managed CPython), `.tools/venv/` (main env),
   `.tools/envs/<env>/` (per-backend envs), `.tools/hf-home/` (HF cache), `.tools/logs/`.
   When calling uv yourself, set `UV_PYTHON_INSTALL_DIR=.tools/python`, `UV_CACHE_DIR=.tools/uv-cache`
   and `UV_PYTHON_PREFERENCE=only-managed` (`app/envs.py:uv_environ()` does this).
2. **Don't change system settings** (firewall, PATH, registry, drivers). If one is needed, give the
   user the command to run.
3. **Don't install into the main venv just for one model.** Model-specific packages belong in that
   backend's `install` spec, which builds an isolated env.
4. **GPU jobs go through the lock** whenever more than one agent or process could use the GPU:
   `.tools\venv\Scripts\python.exe scripts\gpulock.py -- <command ...>`.
5. Keep testing proportionate. Verify the core path once (one English and one German clip, checked
   with Whisper), then hand it over. The repo owner dislikes long verification loops.

## Setting up a fresh checkout (download + prepare all models)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1          # uv + Python 3.11 + main env (torch 2.11 cu128)
.tools\venv\Scripts\python.exe scripts\prepare.py --list    # what's in models.default.json
.tools\venv\Scripts\python.exe scripts\prepare.py           # build every backend env + download all weights
.tools\venv\Scripts\python.exe scripts\prepare.py --smoke   # optional: one GPU test clip per model (.tools/test/)
```

- `prepare.py [repo|backend ...]` limits the run to some models. `--envs-only` and `--download-only`
  split the two phases. It's idempotent, so just re-run it after a failure.
- Disk: weights total ~45 GB (Voxtral 8 GB, Higgs 8.5 GB, VoxCPM2 ~5 GB, Qwen3 2 × ~4.5 GB,
  OmniVoice ~2.5 GB), plus ~3–5 GB per env. uv hardlinks from its cache, so duplicate wheels are cheap.
- Some models download extra files lazily, on first use of a feature: OmniVoice fetches Whisper
  large-v3-turbo (~1.6 GB) for auto-transcribing reference clips; VoxCPM fetches the ZipEnhancer
  denoiser from ModelScope into `.tools/modelscope`; Higgs fetches its example voice clips into
  `.tools/voices/higgs`.
- Start the app with `run.cmd` (listens on 0.0.0.0:7860; `HOST`/`PORT` env vars override it). If you
  start it from an agent shell, run it in the background:
  `.tools\venv\Scripts\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 7860`.

## Architecture

- `app/server.py` (main env, never imports torch): HTTP API + UI. On load it calls
  `envs.ensure_env(cls)`, then starts `app/worker.py` in the backend's env and talks to it over
  `multiprocessing.connection` (JSON messages + raw float32 audio bytes). Unload = kill the worker,
  which frees all VRAM. Only one model is loaded at a time.
- API: `GET /api/state`, `POST /api/load {model, backend?}`, `POST /api/unload`,
  `POST /api/tts` (multipart: `text`, `params` JSON, file fields for `audio` params) → WAV with
  `X-Duration`, `X-Elapsed`, `X-Info` headers. `DELETE /api/models?repo=`.
- `app/backends/__init__.py`: `discover()` imports every non-underscore module in `app/backends/`
  and reads `BACKEND`. `inspect_repo()` collects HF metadata + `config.json`/`params.json`.
  `pick_backend()` picks the highest `match()` score.
- `models.default.json` (shipped) seeds the model dropdown and `prepare.py`. `models.json` is the
  user's own list (git-ignored).

## Backend contract (`app/backends/base.py`)

- File name == `name`. Module exposes `BACKEND = <class>`.
- **Top-level imports: stdlib, numpy and `.base` only.** Import torch and model packages inside
  `load()`/`synthesize()`, because the server imports every backend module from the main env.
  Underscore helper modules (e.g. `_voxtral_model.py`) are skipped by discovery and may import
  anything; import them lazily.
- `env` (None = main venv), `python` (version), `install`: a list of `uv pip install` argument lists,
  run in order. Put CUDA torch first (`["torch", "torchaudio", "--index-url",
  "https://download.pytorch.org/whl/cu128"]`, pinned if the package pins torch), then the model
  package. Changing `install` makes the env rebuild on next use (hash marker in the env).
- `match(info: RepoInfo) -> int`: be specific (model_type / architectures / library tag). The
  generic `hf_pipeline` scores 10–20, so dedicated backends should use 50–100.
- `params() -> list[Param]`: every meaningful setting with **static** official defaults and a `help`
  text. Kinds: `select` (choices = str or `{value,label,group}`), `number` (default None = empty /
  omitted), `text`, `textarea`, `checkbox`, `audio` (the backend receives a temp file path).
- `synthesize(text, **params) -> (float32 mono np.ndarray, sample_rate)`. Accept `**_`. Set
  `self.last_info` (e.g. `"seed 123 · voice x"`). Report random seeds so takes can be reproduced.
- Conventions used by the existing backends: a `seed` param (empty = random), `split` (none /
  paragraphs / sentences) + `gap`, `normalize` (peak −1 dBFS), and only **English + German** voices
  where voices are language-specific (the owner's languages; other languages are easy to add back).
- Peak VRAM must stay under ~6.5 GB (8 GB card minus the desktop). Prefer bf16, and quantize the
  LM backbone (bitsandbytes nf4, or the int8 weight-only `Linear` in `_voxtral_model.py`) if needed.

## Development / test loop for a backend

```powershell
.tools\venv\Scripts\python.exe -m app.envs <backend>                      # build/refresh its env
.tools\venv\Scripts\python.exe scripts\gpulock.py -- .tools\envs\<env>\Scripts\python.exe -m app.worker `
    --backend <backend> --repo <org/model> --test "Hello there." --params "{}" --out .tools\test\x_en.wav
.tools\venv\Scripts\python.exe scripts\gpulock.py -- .tools\venv\Scripts\python.exe scripts\check_asr.py en .tools\test\x_en.wav
```

`--test` prints load time, peak VRAM, RTF and the params JSON. `check_asr.py <en|de> files...`
prints Whisper transcripts. For main-env backends use `.tools\venv\Scripts\python.exe` as the worker
interpreter. Check matching with:
`python -c "import sys; sys.path.insert(0,'.'); from app import backends as B; print(B.pick_backend(B.inspect_repo('<repo>'), B.discover()))"`.

## Windows gotchas learned so far

- There's no compiler or CUDA toolkit, so use wheels only. flash-attn is unavailable (use SDPA).
  Triton / `torch.compile` don't work out of the box (VoxCPM runs with `optimize=False` and is
  therefore slower than real time).
- The main per-frame bottleneck of autoregressive TTS here is Python kernel-launch overhead. CUDA
  graphs for the decode step gave 2–5× speedups (Voxtral, Higgs, faster-qwen3-tts).
- bitsandbytes 0.50 has CUDA wheels for Windows. nf4 batch-1 matmuls are about as fast as bf16;
  bnb int8 (`Linear8bitLt`) is ~3× slower and less accurate than a simple per-row int8 weight-only scheme.
- torchaudio ≥ 2.9 needs torchcodec + FFmpeg DLLs for loading audio. OmniVoice and VoxCPM pin
  torch 2.8 for that reason.
- `faster-qwen3-tts` 0.5.4 garbles the first request if the CUDA graphs get captured mid-request.
  The backend captures them and runs a warm-up during `load()`.
- The first import in a fresh env can take minutes (likely antivirus scanning new `.pyd` files).
- Passing paths to `gpulock.py` from bash works with forward slashes (it resolves them), but inside
  `--params` JSON use forward slashes, because backslashes break the JSON escaping.
- The console code page mangles `·` and umlauts in worker test output. That's cosmetic only.

## Adding a new model (checklist)

1. Read the model card + official inference code. Note the supported languages, voices, settings
   and VRAM needs.
2. Find a Windows-compatible runtime (pip wheel). If it only ships Linux (vLLM/SGLang), a port to plain
   PyTorch like `_voxtral_model.py` / `_higgs_tts_model.py` is an option. Credit the source in NOTICE.
3. Write `app/backends/<name>.py`, build the env, test En + De with Whisper, then add the repo to
   `models.default.json` and the README model table.
