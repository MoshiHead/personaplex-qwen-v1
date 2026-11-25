"""Checkpoint save/load for PPQ models.

A PPQ checkpoint directory contains:
  - config.json         (PPQModelConfig)
  - model.safetensors   (full model weights)
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from .config import PPQModelConfig
from .model import PPQLMModel
from .transplant import load_qwen_weights, transplant_depformer

logger = logging.getLogger(__name__)

MODEL_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
COMPLETE_NAME = "COMPLETE"


def save_pretrained(model: PPQLMModel, out_dir: str | Path, complete: bool = True) -> Path:
    """Save atomically: config + weights are written to temp files and renamed,
    and a COMPLETE marker is written last. A checkpoint directory without the
    marker is treated as incomplete (e.g. after a crash mid-save)."""
    import os
    from .progress import atomic_write_text
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # remove a stale marker first so a crash mid-save can't look complete
    (out_dir / COMPLETE_NAME).unlink(missing_ok=True)
    model.cfg.to_json(out_dir / CONFIG_NAME)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    tmp = out_dir / (MODEL_NAME + ".tmp")
    save_file(state, str(tmp))
    os.replace(tmp, out_dir / MODEL_NAME)
    if complete:
        atomic_write_text(out_dir / COMPLETE_NAME, "ok\n")
    logger.info("Saved PPQ checkpoint to %s", out_dir)
    return out_dir


def is_complete(ckpt_dir: str | Path) -> bool:
    ckpt_dir = Path(ckpt_dir)
    return ((ckpt_dir / COMPLETE_NAME).exists()
            and (ckpt_dir / CONFIG_NAME).exists()
            and (ckpt_dir / MODEL_NAME).exists())


def load_pretrained(ckpt_dir: str | Path, device: str | torch.device = "cpu",
                    dtype: torch.dtype = torch.bfloat16) -> PPQLMModel:
    ckpt_dir = Path(ckpt_dir)
    cfg = PPQModelConfig.from_json(ckpt_dir / CONFIG_NAME)
    model = PPQLMModel(cfg, device="meta", dtype=dtype)
    state = load_file(str(ckpt_dir / MODEL_NAME), device="cpu")
    state = {k: v.to(dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    # non-persistent buffers (RoPE tables) are not in the state dict and are
    # still on meta after the assign-load; rebuild them on the target device.
    model.backbone.rotary.build_tables(device)
    model = model.to(device=device)
    model.eval()
    return model


def build_model_from_scratch(
    cfg: PPQModelConfig,
    qwen_dir: str | Path | None = None,
    depformer_source: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> PPQLMModel:
    """Build a PPQ model: fresh audio-side modules + Qwen backbone weights
    + transplanted depformer weights."""
    model = PPQLMModel(cfg, device="cpu", dtype=dtype)
    if qwen_dir is not None:
        load_qwen_weights(model, qwen_dir)
    if depformer_source is not None:
        transplant_depformer(model, depformer_source)
    model = model.to(device=device)
    model.eval()
    return model


def get_mimi(mimi_weight: str | Path, device: str | torch.device = "cpu"):
    """Load the frozen Mimi codec (reuses moshi's loader)."""
    from moshi.models import loaders
    return loaders.get_mimi(str(mimi_weight), device)
