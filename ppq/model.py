"""PPQ language model: Qwen3 backbone + 17 Mimi/text streams + depformer.

Implements the exact same duck-typed interface as PersonaPlex's ``LMModel``
(moshi/models/lm.py) so that the stock ``LMGen`` streaming engine works
unmodified: ``forward_codes``, ``forward_embeddings``, ``embed_codes``,
``forward_depformer``, ``_get_initial_token``, the id properties, and a
moshi ``StreamingTransformer`` depformer (so PersonaPlex depformer weights
can be transplanted 1:1 where shapes allow).

Small helpers (ScaledEmbedding, delay/undelay, LMOutput) are copied from
moshi/models/lm.py to avoid pulling in its audio-IO dependencies here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moshi.modules.streaming import StreamingContainer
from moshi.modules.transformer import StreamingTransformer, create_norm_fn

from .config import PPQModelConfig
from .qwen_backbone import Qwen3Backbone


# --- Helpers copied from moshi/models/lm.py (Apache/MIT licensed, Kyutai/NVIDIA) ---

@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes, so no extra
    # shift is required when computing cross-entropy.
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]


def _delay_sequence(delays: list[int], tensor: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    B, K, T = tensor.shape
    assert len(delays) == K, (len(delays), K)
    outs = []
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            line[:, :delay] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(delays: list[int], tensor: torch.Tensor,
                      fill_value: float = float("NaN")) -> tuple[torch.Tensor, torch.Tensor]:
    B, K, T, *_ = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []
    if all(delay == 0 for delay in delays):
        return tensor, mask
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


class ScaledEmbedding(nn.Embedding):
    """Embedding that outputs exactly 0 for `zero_idx` inputs (moshi convention)."""

    def __init__(self, *args, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        return torch.where(is_zero[..., None], zero, y)


class PPQTextEmbedding(nn.Module):
    """Text embedding over the Qwen vocabulary.

    Rows 0..text_card-1 are Qwen's own embedding table (PAD/EPAD live in
    unused reserved rows and are trained by us). The extra "initial" token
    (id == text_card) is a learned vector, replacing Helium's row 32000.
    """

    def __init__(self, cfg: PPQModelConfig, device=None, dtype=None):
        super().__init__()
        self.text_card = cfg.text_card
        self.init_id = cfg.text_card
        self.zero_idx = -1
        self.emb = nn.Embedding(cfg.text_card, cfg.dim, device=device, dtype=dtype)
        self.init_vec = nn.Parameter(
            torch.zeros(cfg.dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_zero = x == self.zero_idx
        is_init = x == self.init_id
        y = self.emb(x.clamp(min=0, max=self.text_card - 1))
        zero = torch.zeros(1, dtype=y.dtype, device=y.device)
        y = torch.where(is_zero[..., None], zero, y)
        y = torch.where(is_init[..., None], self.init_vec.to(y.dtype), y)
        return y


class PPQLMModel(StreamingContainer):
    """The PPQ language model. Drop-in replacement for PersonaPlex's LMModel."""

    def __init__(self, cfg: PPQModelConfig, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.n_q = cfg.n_q
        self.dep_q = cfg.dep_q
        self.card = cfg.card
        self.text_card = cfg.text_card
        self.delays = list(cfg.delays)
        self.dim = cfg.dim
        dep_dim = cfg.depformer_dim

        # 16 audio embedding tables (8 agent + 8 user), freshly trained.
        self.emb = nn.ModuleList(
            [ScaledEmbedding(self.card + 1, self.dim, zero_idx=-1, device=device, dtype=dtype)
             for _ in range(self.n_q)]
        )
        self.text_emb = PPQTextEmbedding(cfg, device=device, dtype=dtype)
        self.text_linear = nn.Linear(self.dim, self.text_card, bias=False, device=device, dtype=dtype)

        # Qwen3 temporal transformer (includes the final RMSNorm, like HF).
        self.backbone = Qwen3Backbone(cfg.qwen, device=device, dtype=dtype)
        self.out_norm = None  # final norm lives inside the backbone

        # Depformer: identical construction to PersonaPlex's LMModel so that
        # pretrained depformer weights transplant cleanly.
        self.depformer_in = nn.ModuleList(
            [nn.Linear(self.dim, dep_dim, bias=False, device=device, dtype=dtype)
             for _ in range(self.dep_q)]
        )
        # Only dep_q - 1 tables: the last codebook is never an input.
        self.depformer_emb = nn.ModuleList(
            [ScaledEmbedding(self.card + 1, dep_dim, zero_idx=-1, device=device, dtype=dtype)
             for _ in range(self.dep_q - 1)]
        )
        self.depformer_text_emb = ScaledEmbedding(
            self.text_card + 1, dep_dim, zero_idx=-1, device=device, dtype=dtype)
        self.depformer = StreamingTransformer(
            d_model=dep_dim,
            num_heads=cfg.depformer_num_heads,
            num_layers=cfg.depformer_num_layers,
            dim_feedforward=cfg.depformer_dim_feedforward,
            norm="rms_norm_f32",
            positional_embedding="none",
            context=None,
            causal=True,
            layer_scale=None,
            max_period=10_000,
            gating="silu",
            weights_per_step=self.dep_q,
            device=device,
            dtype=dtype,
        )
        self.depformer.set_streaming_propagate(False)
        self.linears = nn.ModuleList(
            [nn.Linear(dep_dim, self.card, bias=False, device=device, dtype=dtype)
             for _ in range(self.dep_q)]
        )

    # --- id conventions (mirroring LMModel) ---------------------------------
    @property
    def initial_token_id(self) -> int:
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        return self.cfg.text_pad_id

    @property
    def end_of_text_padding_id(self) -> int:
        return self.cfg.text_epad_id

    @property
    def zero_token_id(self) -> int:
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        return -2

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # [1, K=17, 1]: text initial token + audio initial tokens.
        device = next(iter(self.parameters())).device
        special = torch.full([1, 1, 1], self.initial_token_id, device=device, dtype=torch.long)
        text_special = torch.full_like(special, self.text_initial_token_id)
        audio_token = special.expand(-1, self.num_audio_codebooks, -1)
        return torch.cat([text_special, audio_token], dim=1)

    # --- forwards ------------------------------------------------------------
    def embed_codes(self, sequence: torch.Tensor) -> torch.Tensor:
        B, K, S = sequence.shape
        assert K == self.num_codebooks, (
            f"Sequence shape {sequence.shape} must match the number of codebooks.")
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](sequence[:, cb_index + self.audio_offset])
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb
        return input_

    def forward_codes(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embed_codes(sequence))

    def forward_embeddings(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(input)  # final RMSNorm applied inside the backbone
        text_logits = self.text_linear(h)
        text_logits = text_logits[:, None]
        return h, text_logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, S = sequence.shape
        assert K == 1, f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert S == 1, f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert transformer_out.shape[1] == 1, "Transformer out should be for a single step."
        depformer_input = self.depformer_in[depformer_cb_index](transformer_out)
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](sequence[:, 0])
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim]; the streaming state of the
        # depformer ensures the proper per-step weights are used.
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, 1, 1, card]
        return logits

    def forward_depformer_training(
        self,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, T = sequence.shape
        Ka = self.dep_q
        assert K == self.num_codebooks, (
            f"Codebooks for Depformer training should be passed all at once, got {K}.")
        depformer_inputs = []
        for cb_index in range(Ka):
            transformer_in = self.depformer_in[cb_index](transformer_out)
            if cb_index == 0:
                token_in = self.depformer_text_emb(sequence[:, 0])
            else:
                token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + self.audio_offset - 1])
            depformer_inputs.append(token_in + transformer_in)
        depformer_input = torch.stack(depformer_inputs, 2)
        # [B, T, K, depformer_dim] -> [B * T, K, D]
        depformer_input = depformer_input.view(B * T, Ka, -1)
        depformer_output = self.depformer(depformer_input)
        all_logits = []
        for cb_index in range(Ka):
            logits = self.linears[cb_index](depformer_output[:, cb_index])
            all_logits.append(logits.view(B, T, -1))
        logits = torch.stack(all_logits, 1)
        assert logits.dim() == 4, logits.shape  # [B, Ka, T, card]
        return logits

    def forward_train(self, codes: torch.Tensor) -> LMOutput:
        B, K, T = codes.shape
        initial = self._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial)
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)

        transformer_out, text_logits = self.forward_codes(delayed_codes[:, :, :-1])
        logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

        # Map logits back onto the original (undelayed) codes.
        logits, logits_mask = _undelay_sequence(
            self.delays[self.audio_offset:self.audio_offset + self.dep_q],
            logits, fill_value=float("NaN"))
        logits_mask &= (codes[:, self.audio_offset:self.audio_offset + self.dep_q] != self.zero_token_id)
        text_logits, text_logits_mask = _undelay_sequence(
            self.delays[:1], text_logits, fill_value=float("NaN"))
        text_logits_mask &= (codes[:, :1] != self.zero_token_id)
        return LMOutput(logits, logits_mask, text_logits, text_logits_mask)


