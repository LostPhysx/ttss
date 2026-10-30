"""Standalone PyTorch runtime for Boson AI Higgs TTS 3 (bosonai/higgs-tts-3-4b).

The official runtimes are SGLang-Omni / vLLM-Omni (Linux, >= 24 GB). This is a small port of
their reference path (sglang_omni/models/higgs_tts) for one request at a time on an 8 GB card:

  prompt  <|tts|> [<|ref_text|> ref transcript] [<|ref_audio|> ref codes] <|text|> text <|audio|>
  Qwen3-4B decoder (36 layers) --> fused 8-codebook head (tied with the fused audio embedding)
  8 codes per 40 ms frame, staggered by a delay pattern, fed back as the next input embedding
  codes --> Higgs Audio v2 tokenizer (bundled in the checkpoint) --> 24 kHz waveform

The decoder's linear layers are quantized (nf4 via bitsandbytes, or int8 weight-only) and the
decode step runs as a CUDA graph. The text-embedding table stays on the CPU (prompt only).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

log = logging.getLogger("ttss")

BOC, EOC = 1024, 1025          # per-codebook begin / end-of-audio codes (delay-pattern padding)
PLACEHOLDER = -100             # prompt slot that receives a reference-audio embedding
SAMPLE_RATE = 24_000
FRAME_RATE = 25
CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."

# bosonai/higgs-audio-v2-tokenizer config.json (as bundled by sglang-omni)
CODEC_CONFIG = {
    "acoustic_model_config": {
        "codebook_dim": 8, "codebook_loss_weight": 1.0, "codebook_size": 1024,
        "commitment_loss_weight": 0.25, "decoder_hidden_size": 1024,
        "downsampling_ratios": [8, 5, 4, 2, 3], "encoder_hidden_size": 64, "hidden_size": 256,
        "hop_length": 960, "model_type": "dac", "n_codebooks": 9, "quantizer_dropout": 0,
        "sampling_rate": 16000, "upsampling_ratios": [8, 5, 4, 2, 3],
    },
    "block_dilations": [1, 1], "channel_ratios": [1, 1], "codebook_dim": 64, "codebook_size": 1024,
    "downsample_factor": 320, "initializer_range": 0.02, "kernel_size": 3,
    "sample_rate": 24000,
    "semantic_model_config": {
        "activation_dropout": 0.1, "apply_spec_augment": True, "attention_dropout": 0.1,
        "bos_token_id": 1, "classifier_proj_size": 256, "conv_bias": False,
        "conv_dim": [512] * 7, "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
        "conv_pos_batch_norm": False, "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "ctc_loss_reduction": "sum", "ctc_zero_infinity": False, "do_stable_layer_norm": False,
        "eos_token_id": 2, "feat_extract_activation": "gelu", "feat_extract_norm": "group",
        "feat_proj_dropout": 0.0, "feat_proj_layer_norm": True, "final_dropout": 0.1,
        "hidden_act": "gelu", "hidden_dropout": 0.1, "hidden_size": 768,
        "initializer_range": 0.02, "intermediate_size": 3072, "layer_norm_eps": 1e-05,
        "layerdrop": 0.1, "mask_feature_length": 10, "mask_feature_min_masks": 0,
        "mask_feature_prob": 0.0, "mask_time_length": 10, "mask_time_min_masks": 2,
        "mask_time_prob": 0.0, "model_type": "hubert", "num_attention_heads": 12,
        "num_conv_pos_embedding_groups": 16, "num_conv_pos_embeddings": 128,
        "num_feat_extract_layers": 7, "num_hidden_layers": 12, "pad_token_id": 0,
        "use_weighted_layer_sum": False, "vocab_size": 32,
    },
    "semantic_sample_rate": 16000, "strides": [1, 1], "target_bandwidths": [0.5, 1, 1.5, 2],
    "unit_kernel_size": 3,
}


# --- building blocks ---------------------------------------------------------

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Linear:
    """y = x @ W^T with W stored as bf16, int8 (weight-only, per-row scale) or nf4 (bitsandbytes)."""

    def __init__(self, w: torch.Tensor, mode: str, device: str):
        self.mode = mode
        if mode == "bf16":
            self.w = w.to(device, torch.bfloat16)
        elif mode == "int8":
            wf = w.to(device, torch.float32)
            s = wf.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            self.w = (wf / s[:, None]).round_().clamp_(-127, 127).to(torch.int8)
            self.s = s.to(torch.bfloat16)
            del wf
        elif mode == "nf4":
            import bitsandbytes as bnb
            lin = bnb.nn.Linear4bit(w.shape[1], w.shape[0], bias=False,
                                    compute_dtype=torch.bfloat16, quant_type="nf4")
            lin.weight = bnb.nn.Params4bit(w.to(torch.bfloat16), requires_grad=False, quant_type="nf4")
            self.lin = lin.to(device)
        else:
            raise ValueError(f"unknown precision {mode!r}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "bf16":
            return F.linear(x, self.w)
        if self.mode == "int8":
            # (torch._weight_int8pack_mm and W8A8 int8 GEMMs were not faster / less accurate here)
            return F.linear(x, self.w.to(x.dtype)) * self.s
        return self.lin(x)


# --- Qwen3 decoder -------------------------------------------------------------

class Qwen3:
    """Qwen3 decoder stack (no embeddings / LM head) with a static KV cache."""

    def __init__(self, cfg: dict, get, device: str, precision: str):
        self.device, self.precision = device, precision
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads, self.n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or cfg["hidden_size"] // self.n_heads
        self.eps = cfg["rms_norm_eps"]
        theta = (cfg.get("rope_parameters") or {}).get("rope_theta") or cfg.get("rope_theta", 1e6)
        self.q_dim, self.kv_dim = self.n_heads * self.hd, self.n_kv * self.hd
        self.inter = cfg["intermediate_size"]
        bf = lambda k: get(k).to(device, torch.bfloat16)
        self.layers = []
        for i in range(self.n_layers):
            p = f"body.layers.{i}."
            qkv = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
            gate_up = torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")])
            self.layers.append({
                "in_norm": bf(p + "input_layernorm.weight"),
                "post_norm": bf(p + "post_attention_layernorm.weight"),
                "q_norm": bf(p + "self_attn.q_norm.weight"),
                "k_norm": bf(p + "self_attn.k_norm.weight"),
                "qkv": Linear(qkv, precision, device),
                "o": Linear(get(p + "self_attn.o_proj.weight"), precision, device),
                "gate_up": Linear(gate_up, precision, device),
                "down": Linear(get(p + "mlp.down_proj.weight"), precision, device),
            })
            del qkv, gate_up
        self.norm = bf("body.norm.weight")
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32, device=device) / self.hd))
        self.max_seq = 0

    def ensure_cache(self, n: int) -> bool:
        """Grow the KV cache to >= n positions. Returns True if it was reallocated."""
        if n <= self.max_seq:
            return False
        n = int(2 ** math.ceil(math.log2(max(n, 1024))))
        self.k_cache = self.v_cache = None
        torch.cuda.empty_cache()
        shape = (1, self.n_kv, n, self.hd)
        self.k_cache = [torch.zeros(shape, device=self.device, dtype=torch.bfloat16) for _ in range(self.n_layers)]
        self.v_cache = [torch.zeros(shape, device=self.device, dtype=torch.bfloat16) for _ in range(self.n_layers)]
        self.max_seq = n
        self.arange = torch.arange(n, device=self.device)
        return True

    def _cos_sin(self, pos: torch.Tensor):
        # like transformers: angles in fp32, cos/sin rounded to bf16, rotation applied in bf16
        ang = pos[:, None].float() * self.inv_freq[None]
        emb = torch.cat((ang, ang), dim=-1)
        return emb.cos().to(torch.bfloat16)[None, None], emb.sin().to(torch.bfloat16)[None, None]

    def _qkv(self, L, h, T, cos, sin):
        qkv = L["qkv"](rms_norm(h, L["in_norm"], self.eps))
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = rms_norm(q.view(1, T, self.n_heads, self.hd), L["q_norm"], self.eps).transpose(1, 2)
        k = rms_norm(k.view(1, T, self.n_kv, self.hd), L["k_norm"], self.eps).transpose(1, 2)
        v = v.view(1, T, self.n_kv, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k, v

    def _mlp(self, L, h):
        g, u = L["gate_up"](rms_norm(h, L["post_norm"], self.eps)).split([self.inter, self.inter], dim=-1)
        return L["down"](F.silu(g) * u)

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start: int) -> torch.Tensor:
        """Prefill. h: (1, T, D) embeddings at positions [start, start+T). Returns normed hidden."""
        T = h.shape[1]
        end = start + T
        if end > self.max_seq:
            raise RuntimeError(f"sequence too long ({end} > {self.max_seq})")
        cos, sin = self._cos_sin(torch.arange(start, end, device=h.device))
        for i, L in enumerate(self.layers):
            q, k, v = self._qkv(L, h, T, cos, sin)
            self.k_cache[i][:, :, start:end] = k
            self.v_cache[i][:, :, start:end] = v
            o = F.scaled_dot_product_attention(
                q, self.k_cache[i][:, :, :end], self.v_cache[i][:, :, :end],
                is_causal=(T > 1 and start == 0), enable_gqa=True)
            h = h + L["o"](o.transpose(1, 2).reshape(1, T, -1))
            h = h + self._mlp(L, h)
        return rms_norm(h, self.norm, self.eps)

    @torch.inference_mode()
    def step(self, h: torch.Tensor, pos: torch.Tensor, n_ctx: int) -> torch.Tensor:
        """One decode step with fixed shapes (CUDA-graph friendly). h: (1, 1, D); pos: (1,) long."""
        mask = torch.zeros(1, 1, 1, n_ctx, device=h.device, dtype=h.dtype).masked_fill(
            (self.arange[:n_ctx] > pos).view(1, 1, 1, n_ctx), float("-inf"))
        cos, sin = self._cos_sin(pos)
        for i, L in enumerate(self.layers):
            q, k, v = self._qkv(L, h, 1, cos, sin)
            self.k_cache[i].index_copy_(2, pos, k)
            self.v_cache[i].index_copy_(2, pos, v)
            o = F.scaled_dot_product_attention(
                q, self.k_cache[i][:, :, :n_ctx], self.v_cache[i][:, :, :n_ctx], attn_mask=mask, enable_gqa=True)
            h = h + L["o"](o.transpose(1, 2).reshape(1, 1, -1))
            h = h + self._mlp(L, h)
        return rms_norm(h, self.norm, self.eps)


# --- delay pattern -----------------------------------------------------------

def apply_delay(codes: torch.Tensor) -> torch.Tensor:
    """(T, N) raw codes -> (T + N - 1, N): codebook c shifted down by c rows, BOC/EOC padded."""
    t, n = codes.shape
    out = torch.full((t + n - 1, n), EOC, dtype=codes.dtype)
    for c in range(n):
        out[:c, c] = BOC
        out[c:c + t, c] = codes[:, c]
    return out


def reverse_delay(delayed: torch.Tensor) -> torch.Tensor:
    l, n = delayed.shape
    t = l - (n - 1)
    if t <= 0:
        return delayed[:0]
    return torch.stack([delayed[c:c + t, c] for c in range(n)], dim=1)


def sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, gen) -> torch.Tensor:
    """logits (N, V) float -> (N,) codes; each codebook sampled independently."""
    if temperature <= 1e-5:
        return logits.argmax(-1)
    logits = logits / temperature
    if 0 < top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0 < top_p < 1.0:
        srt, idx = logits.sort(dim=-1, descending=True)
        remove = srt.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.masked_fill(torch.zeros_like(remove).scatter_(-1, idx, remove), float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1, generator=gen)[:, 0]


# --- full model --------------------------------------------------------------

class HiggsTTS:
    def __init__(self, path: str, device: str = "cuda", precision: str = "nf4"):
        from tokenizers import Tokenizer

        self.path, self.device = Path(path), device
        self.cfg = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        self.text_cfg = self.cfg["text_config"]
        enc = self.cfg.get("audio_encoder_config") or {}
        self.n_cb = int(enc.get("num_codebooks", 8))
        self.cb_vocab = int(enc.get("vocab_size", 1026))
        self.weights = str(self.path / "model.safetensors")

        self.tok = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        ids = {t: self.tok.token_to_id(t) for t in ("<|tts|>", "<|ref_text|>", "<|ref_audio|>", "<|text|>", "<|audio|>")}
        if None in ids.values():
            raise RuntimeError(f"tokenizer is missing Higgs special tokens: {ids}")
        self.special = ids

        with safe_open(self.weights, framework="pt", device="cpu") as f:
            self.text_emb = f.get_tensor("tied.embedding.text_embedding.weight").to(torch.bfloat16)  # CPU
            self.audio_emb = f.get_tensor("tied.embedding.modality_embeddings.0.embedding.weight").to(
                device, torch.bfloat16)                                                   # (N*V, D), tied head
        self.offsets = torch.arange(self.n_cb, device=device) * self.cb_vocab
        self.codec = self._load_codec()
        self.lm: Qwen3 | None = None
        self.use_graphs = device.startswith("cuda")
        self._reset_graphs()
        self.set_precision(precision)

    # -- codec --
    def _load_codec(self):
        from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

        state = {}
        with safe_open(self.weights, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(CODEC_PREFIX):
                    state[k[len(CODEC_PREFIX):]] = f.get_tensor(k)
        model = HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig(**CODEC_CONFIG))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > len(state) // 2:
            raise RuntimeError(f"codec weights don't match ({len(missing)} missing)")
        if missing or unexpected:
            log.info("codec: %d missing, %d unexpected keys", len(missing), len(unexpected))
        model = model.float().eval().to(self.device)   # fp32: decoder ConvTranspose is unstable in bf16
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @torch.inference_mode()
    def encode_audio(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """mono float waveform -> (T, N) raw codes (CPU)."""
        import torchaudio.functional as AF
        wav = wav.float().view(1, 1, -1)
        if sr != SAMPLE_RATE:
            wav = AF.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[-1] < SAMPLE_RATE:
            wav = F.pad(wav, (0, SAMPLE_RATE - wav.shape[-1]))
        with torch.backends.cudnn.flags(enabled=False):
            codes = self.codec.encode(wav.to(self.device)).audio_codes      # (1, N, T)
        return codes[0, :self.n_cb].transpose(0, 1).long().cpu()

    @torch.inference_mode()
    def decode_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """(T, N) raw codes -> waveform (CPU float32)."""
        codes = codes.clamp(0, BOC - 1).transpose(0, 1)[None].to(self.device)
        return self.codec.decode(codes).audio_values.reshape(-1).float().cpu()

    # -- LM / graphs --
    def set_precision(self, precision: str):
        if self.lm is not None and self.lm.precision == precision:
            return
        self.lm = None
        self._reset_graphs()
        torch.cuda.empty_cache()
        with safe_open(self.weights, framework="pt", device="cpu") as f:
            self.lm = Qwen3(self.text_cfg, f.get_tensor, self.device, precision)
        self.warmup()

    @torch.inference_mode()
    def warmup(self, n: int = 2048):
        """Allocate the KV cache and capture the decode-step graphs up front (first clip is faster)."""
        if not self.use_graphs:
            return
        self.lm.ensure_cache(n)
        self._reset_graphs()
        b = 256
        while b <= self.lm.max_seq:
            self._run_step(b)
            b *= 2

    def _reset_graphs(self):
        dev, D = self.device, self.text_cfg["hidden_size"]
        self._graphs = {}
        self._pool = torch.cuda.graph_pool_handle() if self.use_graphs else None
        self.b_codes = torch.zeros(self.n_cb, dtype=torch.long, device=dev)
        self.b_pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.b_logits = torch.zeros(self.n_cb, self.cb_vocab, dtype=torch.float32, device=dev)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (..., N) -> summed embedding (..., D)."""
        return F.embedding(codes + self.offsets, self.audio_emb).sum(dim=-2)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """h (D,) -> logits (N, V) fp32."""
        return F.linear(h, self.audio_emb).view(self.n_cb, self.cb_vocab).float()

    def _step_fn(self, n_ctx):
        def fn():
            h = self.lm.step(self.embed_codes(self.b_codes).view(1, 1, -1), self.b_pos, n_ctx)
            self.b_logits.copy_(self.head(h[0, -1]))
        return fn

    def _capture(self, fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            fn()
        return g

    def _run_step(self, n_ctx):
        if not self.use_graphs:
            return self._step_fn(n_ctx)()
        g = self._graphs.get(n_ctx)
        if g is None:
            try:
                g = self._graphs[n_ctx] = self._capture(self._step_fn(n_ctx))
            except Exception as e:
                log.warning("CUDA graph capture failed, running eagerly: %s", e)
                self.use_graphs = False
                return self._step_fn(n_ctx)()
        g.replay()

    # -- generation --
    def build_prompt(self, text: str, ref_text: str | None, n_ref: int) -> list[int]:
        s = self.special
        ids = [s["<|tts|>"]]
        if ref_text and n_ref:
            ids += [s["<|ref_text|>"]] + self.tok.encode(ref_text, add_special_tokens=False).ids
        if n_ref:
            ids += [s["<|ref_audio|>"]] + [PLACEHOLDER] * n_ref
        ids += [s["<|text|>"]] + self.tok.encode(text, add_special_tokens=False).ids + [s["<|audio|>"]]
        return ids

    @torch.inference_mode()
    def generate(self, text: str, ref_codes: torch.Tensor | None = None, ref_text: str | None = None, *,
                 temperature: float = 0.8, top_p: float = 1.0, top_k: int = 50,
                 max_new_tokens: int = 1024, seed: int = 0) -> tuple[torch.Tensor, bool]:
        """Returns (raw codes (T, N) on CPU, stopped_by_model)."""
        dev, n = self.device, self.n_cb
        delayed_ref = apply_delay(ref_codes) if ref_codes is not None and len(ref_codes) else None
        ids = torch.tensor(self.build_prompt(text, ref_text, 0 if delayed_ref is None else len(delayed_ref)))
        emb = self.text_emb[ids.clamp(min=0)].to(dev)                     # (L, D)
        if delayed_ref is not None:
            slots = (ids == PLACEHOLDER).nonzero().squeeze(1).to(dev)
            emb[slots] = self.embed_codes(delayed_ref.to(dev)).to(emb.dtype)

        if self.lm.ensure_cache(len(ids) + max_new_tokens + 1):
            self._reset_graphs()  # cache reallocated -> captured graphs point to old memory
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        h = self.lm.forward(emb[None], 0)[0, -1]
        logits = self.head(h)
        pos = len(ids)
        rows, delay, countdown, done = [], 0, None, False
        for _ in range(max_new_tokens):
            codes = sample(logits, temperature, top_p, top_k, gen)
            if delay < n:              # delay window: codebooks > delay are still BOC
                codes[delay + 1:] = BOC
                delay += 1
            elif countdown is not None:  # cb0 ended; let the other codebooks finish
                countdown -= 1
                done = countdown <= 0
            elif int(codes[0]) == EOC:
                countdown = n - 2
            rows.append(codes)
            if done:
                break
            self.b_codes.copy_(codes)
            self.b_pos.fill_(pos)
            n_ctx = min(self.lm.max_seq, max(256, 1 << pos.bit_length()))  # attention length bucket
            self._run_step(n_ctx)
            logits = self.b_logits
            pos += 1
        if not rows:
            return torch.zeros(0, n, dtype=torch.long), done
        raw = reverse_delay(torch.stack(rows).cpu())
        return raw, done
