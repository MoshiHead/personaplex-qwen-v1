"""Streaming Qwen3 backbone for PPQ.

A from-scratch Qwen3 decoder that is
  * numerically identical to HF ``Qwen3Model`` (verified by unit tests),
  * streaming-capable through moshi's ``StreamingModule`` machinery with a
    ring-buffer KV cache, so single-step inference can be CUDA-graph captured
    exactly like PersonaPlex's Helium transformer.

Weight names mirror HF (``layers.N.self_attn.q_proj.weight`` ...) so loading
from an official Qwen3 checkpoint is a trivial key remap.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from moshi.modules.streaming import StreamingModule, StreamingContainer, _NullState
from moshi.modules.transformer import RingKVCache, KVCacheResult

from .config import QwenBackboneConfig


class Qwen3RMSNorm(nn.Module):
    """Exact copy of HF Qwen3RMSNorm (compute in fp32, scale after cast)."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    """Precomputed rotate-half RoPE tables, matching HF Qwen3RotaryEmbedding.

    Buffers are float32 [max_position, head_dim] (cos/sin), non-persistent.
    """

    def __init__(self, cfg: QwenBackboneConfig, device=None):
        super().__init__()
        self.head_dim = cfg.head_dim
        self._cfg = cfg  # kept so the tables can be rebuilt (e.g. after meta init)
        self.build_tables(device)

    def build_tables(self, device=None) -> None:
        cfg = self._cfg
        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32, device=device) / cfg.head_dim)
        )
        t = torch.arange(cfg.max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [T] long on device. Returns cos/sin [1, 1, T, D].
        cos = self.cos_cached.index_select(0, positions).to(dtype)
        sin = self.sin_cached.index_select(0, positions).to(dtype)
        return cos[None, None], sin[None, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)



@dataclass
class _QwenAttnState:
    kv_cache: RingKVCache
    offset: torch.Tensor  # device long [1], absolute position of the next token
    offset_cpu: int

    def reset(self):
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0


class Qwen3Attention(StreamingModule[_QwenAttnState]):
    """Qwen3 attention: GQA + per-head Q/K RMSNorm + rotate-half RoPE.

    Non-streaming: full causal attention over the whole sequence (training).
    Streaming: one token per call, KV in a ring cache (CUDA-graph safe).
    """

    def __init__(self, cfg: QwenBackboneConfig, rotary: Qwen3RotaryEmbedding, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.context = cfg.context
        self.rotary = rotary
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False, **factory)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False, **factory)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False, **factory)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False, **factory)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, **factory)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, **factory)

    def _init_streaming_state(self, batch_size: int) -> _QwenAttnState:
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        kv_cache = RingKVCache(batch_size, self.num_kv_heads, self.head_dim,
                               capacity=self.context, device=device, dtype=dtype)
        return _QwenAttnState(kv_cache, torch.zeros(1, device=device, dtype=torch.long), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        state = self._streaming_state
        if state is None:
            offset = torch.zeros(1, device=x.device, dtype=torch.long)
        else:
            offset = state.offset

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, H, T, D]
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        positions = offset + torch.arange(T, device=x.device, dtype=torch.long)
        cos, sin = self.rotary(positions, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if state is None:
            k, v, pos_k = KVCacheResult.from_kv(k, v)
        else:
            k, v, pos_k = state.kv_cache.complete(k, v)

        # Causal mask built from absolute positions (identical logic to moshi's
        # StreamingMultiheadAttention so it works with the ring cache).
        pos_k = pos_k.view(1, -1)  # [1, S]
        pos_q = offset + torch.arange(T, device=x.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)  # [T, S]

        # GQA: expand KV heads to query heads.
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return out


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: QwenBackboneConfig, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, **factory)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, **factory)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(StreamingModule[_NullState]):
    """Pre-norm decoder layer. Weight names mirror HF Qwen3DecoderLayer."""

    def __init__(self, cfg: QwenBackboneConfig, rotary: Qwen3RotaryEmbedding, device=None, dtype=None):
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, rotary, device=device, dtype=dtype)
        self.mlp = Qwen3MLP(cfg, device=device, dtype=dtype)
        self.input_layernorm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, device=device, dtype=dtype)

    def _init_streaming_state(self, batch_size: int) -> _NullState:
        return _NullState()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Backbone(StreamingContainer):
    """Qwen3 decoder stack + final norm. Input is already-embedded tokens.

    Mirrors HF ``Qwen3Model`` minus the embedding layer (PPQ feeds the sum of
    17 stream embeddings, like Helium in PersonaPlex).
    """

    def __init__(self, cfg: QwenBackboneConfig, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.rotary = Qwen3RotaryEmbedding(cfg, device=device)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg, self.rotary, device=device, dtype=dtype)
             for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, device=device, dtype=dtype)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        return self.norm(x)

    # -- Weight loading from HF Qwen3 checkpoints ---------------------------
    def load_hf_state_dict(self, hf_state: dict, strict: bool = True) -> list[str]:
        """Load weights from a HF Qwen3 state dict (keys like
        ``model.layers.0.self_attn.q_proj.weight``). Returns missing keys."""
        mapped = {}
        prefix = "model."
        for key, tensor in hf_state.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name == "embed_tokens.weight":
                continue  # handled by PPQLMModel
            if name.startswith("layers.") or name == "norm.weight":
                mapped[name] = tensor
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("rotary.")]
        if strict and unexpected:
            raise RuntimeError(f"Unexpected keys loading Qwen3 weights: {unexpected}")
        return list(missing)

