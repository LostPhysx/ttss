"""Standalone PyTorch implementation of Mistral's Voxtral-4B-TTS-2603.

Ported from vLLM-Omni (vllm_omni/model_executor/models/voxtral_tts/*), without vLLM:

  text + voice prompt --> Mistral LM backbone (26 layers) --> hidden state per audio frame
  hidden --> semantic head (argmax / sampling) + flow-matching acoustic transformer (36 FSQ codes)
  codes  --> fed back into the LM as the next input embedding
  all codes --> Voxtral codec decoder --> 24 kHz waveform (1920 samples per 12.5 Hz frame)

The LM's linear layers can be bf16, int8 (weight-only, per-channel) or nf4 (bitsandbytes)
so the model fits an 8 GB GPU. Everything else stays bf16. The text-embedding table lives
on the CPU since it is only used once per prompt.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from safetensors import safe_open

N_SPECIAL = 2        # audio special tokens: 0 = [EMPTY_AUDIO], 1 = [END_AUDIO]
END_AUDIO = 1
EMPTY_AUDIO = 0


# --- building blocks ---------------------------------------------------------

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(x.dtype) * w


class Linear:
    """y = x @ W^T with W stored as bf16, int8 (+ per-row scale) or nf4."""

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
            return F.linear(x, self.w.to(x.dtype)) * self.s
        return self.lin(x)


def swiglu(x, w1, w2, w3):
    return w2(F.silu(w1(x)) * w3(x))


# --- Mistral LM backbone ------------------------------------------------------

class LanguageModel:
    def __init__(self, cfg: dict, sd, device: str, precision: str, max_seq: int = 1024):
        self.cfg, self.device, self.precision = cfg, device, precision
        self.n_layers, self.n_heads, self.n_kv = cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"]
        self.hd, self.eps = cfg["head_dim"], cfg["norm_eps"]
        bf = lambda k: sd(k).to(device, torch.bfloat16)
        self.layers = []
        for i in range(self.n_layers):
            p = f"layers.{i}."
            self.layers.append({
                "attention_norm": bf(p + "attention_norm.weight"),
                "ffn_norm": bf(p + "ffn_norm.weight"),
                **{n: Linear(sd(p + f"attention.{n}.weight"), precision, device) for n in ("wq", "wk", "wv", "wo")},
                **{n: Linear(sd(p + f"feed_forward.{n}.weight"), precision, device) for n in ("w1", "w2", "w3")},
            })
        self.norm = bf("norm.weight")
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        self.inv_freq = inv.to(device)
        self.max_seq = 0
        self._ensure_cache(max_seq)

    def _ensure_cache(self, n: int):
        if n <= self.max_seq:
            return
        n = int(2 ** math.ceil(math.log2(max(n, 1024))))
        shape = (1, self.n_kv, n, self.hd)
        self.k_cache = [torch.zeros(shape, device=self.device, dtype=torch.bfloat16) for _ in range(self.n_layers)]
        self.v_cache = [torch.zeros(shape, device=self.device, dtype=torch.bfloat16) for _ in range(self.n_layers)]
        self.max_seq = n
        self.arange = torch.arange(n, device=self.device)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Mistral "consolidated" layout: rotate adjacent (even, odd) pairs.
        ang = pos[:, None].float() * self.inv_freq[None]          # (T, hd/2)
        cos, sin = ang.cos()[None, :, None], ang.sin()[None, :, None]
        xf = x.float().unflatten(-1, (-1, 2))
        x0, x1 = xf[..., 0], xf[..., 1]
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2).to(x.dtype)

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start: int) -> torch.Tensor:
        """h: (1, T, D) input embeddings at positions [start, start+T). Returns normed hidden (1, T, D)."""
        T = h.shape[1]
        end = start + T
        if end > self.max_seq:
            raise RuntimeError(f"sequence too long ({end} > {self.max_seq})")
        pos = torch.arange(start, end, device=h.device)
        for i, L in enumerate(self.layers):
            a = rms_norm(h, L["attention_norm"], self.eps)
            q = self._rope(L["wq"](a).view(1, T, self.n_heads, self.hd), pos).transpose(1, 2)
            k = self._rope(L["wk"](a).view(1, T, self.n_kv, self.hd), pos).transpose(1, 2)
            v = L["wv"](a).view(1, T, self.n_kv, self.hd).transpose(1, 2)
            self.k_cache[i][:, :, start:end] = k
            self.v_cache[i][:, :, start:end] = v
            o = F.scaled_dot_product_attention(
                q, self.k_cache[i][:, :, :end], self.v_cache[i][:, :, :end],
                is_causal=(T > 1 and start == 0), enable_gqa=True,
            )
            h = h + L["wo"](o.transpose(1, 2).reshape(1, T, -1))
            h = h + swiglu(rms_norm(h, L["ffn_norm"], self.eps), L["w1"], L["w2"], L["w3"])
        return rms_norm(h, self.norm, self.eps)

    @torch.inference_mode()
    def step(self, h: torch.Tensor, pos: torch.Tensor, n_ctx: int) -> torch.Tensor:
        """One decode step with fixed shapes (CUDA-graph friendly).
        h: (1, 1, D); pos: (1,) long tensor; attends to cache[:n_ctx] masked to <= pos."""
        mask = torch.zeros(1, 1, 1, n_ctx, device=h.device, dtype=h.dtype).masked_fill(
            (self.arange[:n_ctx] > pos).view(1, 1, 1, n_ctx), float("-inf"))
        for i, L in enumerate(self.layers):
            a = rms_norm(h, L["attention_norm"], self.eps)
            q = self._rope(L["wq"](a).view(1, 1, self.n_heads, self.hd), pos).transpose(1, 2)
            k = self._rope(L["wk"](a).view(1, 1, self.n_kv, self.hd), pos).transpose(1, 2)
            v = L["wv"](a).view(1, 1, self.n_kv, self.hd).transpose(1, 2)
            self.k_cache[i].index_copy_(2, pos, k)
            self.v_cache[i].index_copy_(2, pos, v)
            o = F.scaled_dot_product_attention(
                q, self.k_cache[i][:, :, :n_ctx], self.v_cache[i][:, :, :n_ctx], attn_mask=mask, enable_gqa=True)
            h = h + L["wo"](o.transpose(1, 2).reshape(1, 1, -1))
            h = h + swiglu(rms_norm(h, L["ffn_norm"], self.eps), L["w1"], L["w2"], L["w3"])
        return rms_norm(h, self.norm, self.eps)


# --- flow-matching acoustic transformer ----------------------------------------

class AcousticTransformer:
    def __init__(self, args: dict, audio_args: dict, sd, device: str):
        bf = lambda k: sd("acoustic_transformer." + k).to(device, torch.bfloat16)
        self.n_heads, self.n_kv, self.hd = args["n_heads"], args["n_kv_heads"], args["head_dim"]
        self.eps = args.get("norm_eps", 1e-5)
        self.dim = args["dim"]
        self.n_acoustic = audio_args["n_acoustic_codebook"]
        self.levels = audio_args["acoustic_codebook_size"]
        self.semantic_size = audio_args["semantic_codebook_size"]
        lin = lambda k: Linear(sd("acoustic_transformer." + k), "bf16", device)
        self.input_projection = lin("input_projection.weight")
        self.time_projection = lin("time_projection.weight")
        self.llm_projection = lin("llm_projection.weight")
        self.semantic_out = lin("semantic_codebook_output.weight")
        self.acoustic_out = lin("acoustic_codebook_output.weight")
        self.norm = bf("norm.weight")
        self.layers = []
        for i in range(args["n_layers"]):
            p = f"layers.{i}."
            self.layers.append({
                "attention_norm": bf(p + "attention_norm.weight"),
                "ffn_norm": bf(p + "ffn_norm.weight"),
                **{n: lin(p + f"attention.{n}.weight") for n in ("wq", "wk", "wv", "wo")},
                **{n: lin(p + f"feed_forward.{n}.weight") for n in ("w1", "w2", "w3")},
            })
        half = self.dim // 2
        self.inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half).to(device)
        self._sched: dict[int, tuple] = {}

    def _schedule(self, steps: int):
        if steps not in self._sched:
            ts = torch.linspace(0, 1, steps + 1, device=self.inv_freq.device)
            emb = ts[:, None] * self.inv_freq[None]
            t_emb = torch.cat((emb.cos(), emb.sin()), dim=-1).to(torch.bfloat16)
            self._sched[steps] = (self.time_projection(t_emb), (ts[1:] - ts[:-1]).tolist())
        return self._sched[steps]

    def _layers(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, S, D), bidirectional, no RoPE
        B, S, _ = x.shape
        for L in self.layers:
            a = rms_norm(x, L["attention_norm"], self.eps)
            q = L["wq"](a).view(B, S, self.n_heads, self.hd).transpose(1, 2)
            k = L["wk"](a).view(B, S, self.n_kv, self.hd).transpose(1, 2)
            v = L["wv"](a).view(B, S, self.n_kv, self.hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            x = x + L["wo"](o.transpose(1, 2).reshape(B, S, -1))
            x = x + swiglu(rms_norm(x, L["ffn_norm"], self.eps), L["w1"], L["w2"], L["w3"])
        return rms_norm(x, self.norm, self.eps)

    @torch.inference_mode()
    def semantic_logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.semantic_out(h).float()
        logits[:, EMPTY_AUDIO] = -float("inf")
        logits[:, N_SPECIAL + self.semantic_size:] = -float("inf")
        return logits

    @torch.inference_mode()
    def acoustic_codes(self, h: torch.Tensor, x0: torch.Tensor, cfg_alpha: torch.Tensor, steps: int) -> torch.Tensor:
        """Euler-integrate the flow from noise x0 (1, 36) to the 36 FSQ acoustic values, with CFG."""
        t_proj, dts = self._schedule(steps)
        x = x0
        llm_proj = self.llm_projection(torch.cat([h, torch.zeros_like(h)], dim=0))  # cond, uncond
        for i, dt in enumerate(dts):
            xb = x.to(torch.bfloat16).expand(2, -1)
            inp = torch.stack([self.input_projection(xb), t_proj[i].expand(2, -1), llm_proj], dim=1)
            v = self.acoustic_out(self._layers(inp)[:, 0]).float()
            v = cfg_alpha * v[0:1] + (1 - cfg_alpha) * v[1:2]
            x = x + v * dt
        x = x.clamp(-1, 1)
        return ((x + 1) / 2 * (self.levels - 1)).round().long() + N_SPECIAL  # (1, 36)


# --- Voxtral codec decoder -----------------------------------------------------

def _weight_norm(sd, prefix: str) -> torch.Tensor:
    g = sd(prefix + "conv.parametrizations.weight.original0").float()
    v = sd(prefix + "conv.parametrizations.weight.original1").float()
    return g * v / v.flatten(1).norm(dim=1).clamp(min=1e-12).view(-1, *([1] * (v.dim() - 1)))


def _pad1d(x: torch.Tensor, left: int, mode: str) -> torch.Tensor:
    if mode == "reflect" and x.shape[-1] <= left:
        extra = left - x.shape[-1] + 1
        return F.pad(F.pad(x, (0, extra)), (left, 0), mode="reflect")[..., : -extra]
    return F.pad(x, (left, 0), mode=mode)


class CodecDecoder:
    def __init__(self, args: dict, sd, device: str):
        self.args, self.device = args, device
        p = "audio_tokenizer."
        bf = lambda k: sd(p + k).to(device, torch.bfloat16)
        self.patch = args["pretransform_patch_size"]
        self.levels = args["acoustic_codebook_size"]
        self.n_heads, self.hd = args["n_heads"], args["head_dim"]
        self.norm_eps, self.qk_eps = args["norm_eps"], args["qk_norm_eps"]
        usage = sd(p + "quantizer.semantic_codebook.cluster_usage").float()
        emb_sum = sd(p + "quantizer.semantic_codebook.embedding_sum").float()
        self.semantic_emb = (emb_sum / usage.clamp(min=1e-5)[:, None]).to(device)

        kernels = [int(k) for k in args["decoder_convs_kernels_str"].split(",")]
        strides = [int(s) for s in args["decoder_convs_strides_str"].split(",")]
        lengths = [int(n) for n in args["decoder_transformer_lengths_str"].split(",")]
        enc_strides = [int(s) for s in args.get("encoder_convs_strides_str", "2,2,2,1").split(",")]
        # Attention window: halved per encoder downsampling, doubled per decoder upsampling.
        window = args["attn_sliding_window_size"]
        if args.get("half_attn_window_upon_downsampling", True):
            for s in enc_strides:
                if s > 1:
                    window //= 2

        self.blocks = []
        idx = 0
        self.blocks.append(("conv", _weight_norm(sd, f"{p}decoder_blocks.{idx}.").to(device, torch.bfloat16),
                            strides[0], "replicate"))
        if strides[0] > 1:
            window *= 2
        idx += 1
        for j, n in enumerate(lengths):
            layers = []
            for li in range(n):
                q = f"decoder_blocks.{idx}.layers.{li}."
                layers.append({
                    "attention_norm": bf(q + "attention_norm.weight"), "ffn_norm": bf(q + "ffn_norm.weight"),
                    "attention_scale": bf(q + "attention_scale"), "ffn_scale": bf(q + "ffn_scale"),
                    "q_norm": bf(q + "attention.q_norm.weight"), "k_norm": bf(q + "attention.k_norm.weight"),
                    **{w: bf(q + f"attention.{w}.weight") for w in ("wq", "wk", "wv", "wo")},
                    **{w: bf(q + f"feed_forward.{w}.weight") for w in ("w1", "w2", "w3")},
                })
            self.blocks.append(("transformer", layers, window))
            idx += 1
            if j + 1 < len(lengths) and (kernels[j + 1] != 1 or strides[j + 1] != 1):
                self.blocks.append(("convT", _weight_norm(sd, f"{p}decoder_blocks.{idx}.").to(device, torch.bfloat16),
                                    strides[j + 1]))
                if strides[j + 1] > 1:
                    window *= 2
                idx += 1
        self.out_w = _weight_norm(sd, p + "output_proj.").to(device, torch.bfloat16)
        r = 2.0 ** (-8.0 / self.n_heads)  # ALiBi slopes (n_heads is a power of 2)
        self.slopes = torch.tensor([r ** i for i in range(self.n_heads)], device=device)

    def _attn(self, x: torch.Tensor, L: dict, window: int) -> torch.Tensor:
        """Causal sliding-window attention with ALiBi, computed as a band (exact, low memory)."""
        B, S, _ = x.shape
        H, D = self.n_heads, self.hd
        q = rms_norm(F.linear(x, L["wq"]), L["q_norm"], self.qk_eps).view(B, S, H, D).transpose(1, 2)
        k = rms_norm(F.linear(x, L["wk"]), L["k_norm"], self.qk_eps).view(B, S, H, D).transpose(1, 2)
        v = F.linear(x, L["wv"]).view(B, S, H, D).transpose(1, 2)
        W = window + 1
        kw = F.pad(k, (0, 0, window, 0)).unfold(2, W, 1)   # (B, H, S, D, W): keys s-window .. s
        vw = F.pad(v, (0, 0, window, 0)).unfold(2, W, 1)
        scores = torch.einsum("bhsd,bhsdw->bhsw", q, kw).float() * D ** -0.5
        rel = torch.arange(W, device=x.device) - window          # j - i in [-window, 0]
        scores = scores + self.slopes.view(1, H, 1, 1) * rel
        valid = (torch.arange(S, device=x.device)[:, None] + rel[None]) >= 0
        scores = scores.masked_fill(~valid, float("-inf"))
        o = torch.einsum("bhsw,bhsdw->bhsd", scores.softmax(-1).to(x.dtype), vw)
        return F.linear(o.transpose(1, 2).reshape(B, S, H * D), L["wo"])

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (T, 37) without the special-token offset. Returns waveform (T * 1920,) float32."""
        codes = codes.to(self.device)
        sem = F.embedding(codes[:, 0], self.semantic_emb)                    # (T, 256)
        aco = codes[:, 1:].float() * 2 / (self.levels - 1) - 1               # (T, 36)
        x = torch.cat([sem, aco], dim=-1).to(torch.bfloat16).T[None]       # (1, 292, T)
        for blk in self.blocks:
            if blk[0] == "conv":
                _, w, stride, mode = blk
                x = F.conv1d(_pad1d(x, w.shape[-1] - stride, mode), w, stride=stride)
            elif blk[0] == "convT":
                _, w, stride = blk
                x = F.conv_transpose1d(x, w, stride=stride)
                trim = w.shape[-1] - stride
                x = x[..., : x.shape[-1] - trim]
            else:
                _, layers, window = blk
                h = x.transpose(1, 2)
                for L in layers:
                    h = h + L["attention_scale"] * self._attn(rms_norm(h, L["attention_norm"], self.norm_eps), L, window)
                    f = rms_norm(h, L["ffn_norm"], self.norm_eps)
                    h = h + L["ffn_scale"] * F.linear(F.silu(F.linear(f, L["w1"])) * F.linear(f, L["w3"]), L["w2"])
                x = h.transpose(1, 2)
        x = F.conv1d(_pad1d(x, self.out_w.shape[-1] - 1, "reflect"), self.out_w)  # (1, 240, T*8)
        return x[0].T.reshape(-1).float().cpu()                                  # frames of 240 samples


# --- full model ----------------------------------------------------------------

class VoxtralTTS:
    def __init__(self, weights_path: str, params: dict, device: str = "cuda", precision: str = "nf4"):
        self.params, self.device, self.weights_path = params, device, weights_path
        mm = params["multimodal"]
        self.audio_args = mm["audio_model_args"]
        self.codec_args = mm["audio_tokenizer_args"]
        self.audio_token_id = self.audio_args["audio_token_id"]
        self.sampling_rate = self.codec_args["sampling_rate"]
        sizes = [self.audio_args["semantic_codebook_size"] + N_SPECIAL] + \
                [self.audio_args["acoustic_codebook_size"] + N_SPECIAL] * self.audio_args["n_acoustic_codebook"]
        self.code_offsets = torch.tensor([0] + sizes[:-1], device=device).cumsum(0)

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            sd = f.get_tensor
            self.tok_embeddings = sd("mm_audio_embeddings.tok_embeddings.weight")  # stays on CPU
            self.audio_embeddings = sd("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight").to(device)
            at_args = dict(self.audio_args["acoustic_transformer_args"])
            self.acoustic = AcousticTransformer(at_args, self.audio_args, sd, device)
            self.codec = CodecDecoder(self.codec_args, sd, device)
        self.lm: LanguageModel | None = None
        self.use_graphs = device.startswith("cuda")
        self._reset_graphs()
        self.set_precision(precision)

    # CUDA graphs: each frame = 1 replay for the flow decoder + 1 for the LM step, instead of
    # ~1000 small kernel launches from Python (which made generation CPU-bound).
    def _reset_graphs(self):
        dev, D = self.device, self.params["dim"]
        self._g_lm, self._g_ac = {}, {}
        self._pool = torch.cuda.graph_pool_handle() if self.use_graphs else None
        self.b_codes = torch.zeros(37, dtype=torch.long, device=dev)
        self.b_pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.b_h = torch.zeros(1, D, dtype=torch.bfloat16, device=dev)
        self.b_x0 = torch.zeros(1, self.audio_args["n_acoustic_codebook"], device=dev)
        self.b_alpha = torch.ones(1, 1, device=dev)
        self.b_aco = torch.zeros(1, self.audio_args["n_acoustic_codebook"], dtype=torch.long, device=dev)

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

    def _lm_step_fn(self, n_ctx):
        return lambda: self.b_h.copy_(self.lm.step(self.embed_codes(self.b_codes), self.b_pos, n_ctx)[:, -1])

    def _ac_fn(self, steps):
        return lambda: self.b_aco.copy_(self.acoustic.acoustic_codes(self.b_h, self.b_x0, self.b_alpha, steps))

    def _run(self, table, key, make_fn):
        if not self.use_graphs:
            return make_fn(key)()
        g = table.get(key)
        if g is None:
            try:
                g = table[key] = self._capture(make_fn(key))
            except Exception as e:  # e.g. a kernel that can't be captured: fall back to eager
                import logging
                logging.getLogger("ttss").warning("CUDA graph capture failed, running eagerly: %s", e)
                self.use_graphs = False
                return make_fn(key)()
        g.replay()

    def set_precision(self, precision: str):
        if self.lm is not None and self.lm.precision == precision:
            return
        self.lm = None
        self._reset_graphs()
        torch.cuda.empty_cache()
        with safe_open(self.weights_path, framework="pt", device="cpu") as f:
            self.lm = LanguageModel(self.params, f.get_tensor, self.device, precision)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (37,) with special offset -> summed embedding (1, 1, D)."""
        return self.audio_embeddings[codes + self.code_offsets].sum(0).view(1, 1, -1)

    @torch.inference_mode()
    def generate(self, tokens: list[int], voice_emb: torch.Tensor, *, cfg_alpha: float = 1.2,
                 flow_steps: int = 7, noise_scale: float = 1.0, temperature: float = 0.0,
                 top_k: int = 0, max_frames: int = 1500, seed: int = 0) -> torch.Tensor:
        dev = self.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

        ids = torch.tensor(tokens)
        emb = self.tok_embeddings[ids].to(dev)                        # (L, D) bf16
        slots = (ids == self.audio_token_id).nonzero().squeeze(1).to(dev)
        if len(slots) != voice_emb.shape[0]:
            raise ValueError(f"voice has {voice_emb.shape[0]} frames but prompt has {len(slots)} slots")
        emb[slots] = voice_emb.to(dev, torch.bfloat16)

        old_size = self.lm.max_seq
        self.lm._ensure_cache(len(tokens) + max_frames + 1)
        if self.lm.max_seq != old_size:  # cache reallocated -> captured graphs point to old memory
            self._reset_graphs()
        self.b_h.copy_(self.lm.forward(emb[None], 0)[:, -1])           # prefill, (1, D)
        self.b_alpha.fill_(cfg_alpha)
        pos = len(tokens)
        frames = []
        for step in range(max_frames):
            logits = self.acoustic.semantic_logits(self.b_h)
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    kth = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, -float("inf"))
                sem = torch.multinomial(logits.softmax(-1), 1, generator=gen)[:, 0]
            else:
                sem = logits.argmax(-1)
            if int(sem) == END_AUDIO:
                break
            self.b_x0.copy_(torch.randn(self.b_x0.shape, device=dev, generator=gen) * noise_scale)
            self._run(self._g_ac, flow_steps, self._ac_fn)
            codes = torch.cat([sem.view(1), self.b_aco[0]])            # (37,)
            frames.append(codes)
            self.b_codes.copy_(codes)
            self.b_pos.fill_(pos)
            n_ctx = min(self.lm.max_seq, max(256, 1 << pos.bit_length()))  # attention length bucket
            self._run(self._g_lm, n_ctx, self._lm_step_fn)
            pos += 1
        if not frames:
            return torch.zeros(0)
        return self.codec.decode(torch.stack(frames) - N_SPECIAL)
