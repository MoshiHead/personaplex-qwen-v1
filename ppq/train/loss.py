"""Loss computation for PPQ training (mirrors moshi-finetune weighting)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import PPQModelConfig, TrainConfig
from ..model import LMOutput


def compute_losses(lm_out: LMOutput, codes: torch.Tensor, loss_mask: torch.Tensor,
                   tcfg: TrainConfig, mcfg: PPQModelConfig) -> dict[str, torch.Tensor]:
    """codes: [B, 17, T] int; loss_mask: [B, 17, T] float.

    Returns a dict with 'text', 'cb0'..'cb15', and 'total'.
    """
    device = codes.device
    zero = torch.zeros((), device=device)
    losses: dict[str, torch.Tensor] = {}

    # --- text stream ---
    text_logits = lm_out.text_logits[:, 0]           # [B, T, V]
    text_sel = lm_out.text_mask[:, 0] & (loss_mask[:, 0] > 0)  # [B, T]
    text_target = codes[:, 0]
    w = torch.where(text_target == mcfg.text_pad_id,
                    torch.tensor(tcfg.text_padding_weight, device=device),
                    torch.tensor(1.0, device=device))
    B, T, V = text_logits.shape
    ce = F.cross_entropy(
        text_logits.reshape(B * T, V).float(), text_target.reshape(B * T),
        reduction="none").reshape(B, T)
    denom = (w * text_sel).sum().clamp(min=1.0)
    losses["text"] = (ce * w * text_sel).sum() / denom

    # --- audio codebooks (dep_q = 16: 8 agent + 8 user) ---
    for cb in range(lm_out.logits.shape[1]):
        logits = lm_out.logits[:, cb]                # [B, T, card]
        sel = lm_out.mask[:, cb] & (loss_mask[:, cb + 1] > 0)
        is_agent = cb < 8
        if cb == 0:
            weight = tcfg.first_codebook_weight_multiplier
        elif is_agent:
            weight = 1.0
        else:
            weight = tcfg.user_codebook_weight
        if weight == 0.0 or not bool(sel.any()):
            losses[f"cb{cb}"] = zero
            continue
        cb_ce = F.cross_entropy(logits[sel].float(), codes[:, cb + 1][sel])
        losses[f"cb{cb}"] = cb_ce * weight

    # Normalize the audio weights so the total scale is LR-friendly:
    # cb0 gets mult/(mult+7) of the audio budget, like moshi-finetune.
    mult = tcfg.first_codebook_weight_multiplier
    norm = mult + 7.0 + 8.0 * tcfg.user_codebook_weight
    total = losses["text"]
    for cb in range(lm_out.logits.shape[1]):
        total = total + losses[f"cb{cb}"] * (8.0 / norm)
    losses["total"] = total
    return losses
