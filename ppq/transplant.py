"""Weight loading and transplant utilities for PPQ.

Two sources of pretrained weights:
  1. A Qwen3-Base checkpoint (HF format) -> backbone layers, final norm,
     text embedding table, text output head.
  2. A PersonaPlex (or moshiko) checkpoint -> depformer transformer layers,
     depformer token embeddings and audio output heads. The depformer's
     inner dimension (1024) is backbone-independent, so most weights load
     1:1. Per-step weights are stored with 8 copies in the source and 16 in
     PPQ (dep_q=16), so the upper 8 copies are duplicated from the lower 8,
     exactly like moshi's own `get_moshi_lm` patch logic.

What is NOT transplanted (trained from scratch / from Qwen):
  - the 16 audio embedding tables (dimension mismatch: 4096 -> 2048),
  - depformer_in projections (dimension mismatch),
  - depformer_text_emb (vocabulary mismatch: SentencePiece 32k -> Qwen 151k),
  - text_emb / text_linear (come from Qwen instead).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .model import PPQLMModel

logger = logging.getLogger(__name__)


def _iter_safetensors(path: str | Path):
    """Yield (key, tensor) from a safetensors file or a directory of shards."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files in {path}")
        for f in files:
            yield from _iter_safetensors(f)
    else:
        if path.suffix in (".safetensors", ".sft", ".sfts"):
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
        else:  # torch checkpoint
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            yield from state.items()


def load_qwen_weights(model: PPQLMModel, qwen_path: str | Path) -> dict:
    """Load Qwen3-Base weights into the backbone + text embedding/head."""
    cfg = model.cfg
    qwen_path = Path(qwen_path)
    cfg_file = qwen_path / "config.json" if qwen_path.is_dir() else qwen_path.parent / "config.json"
    report = {"backbone_missing": [], "text_emb": False, "text_head": False}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            hf_cfg = json.load(f)
        for field in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                      "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size"):
            assert getattr(cfg.qwen, field) == hf_cfg[field], (
                f"Qwen config mismatch for {field}: model {getattr(cfg.qwen, field)} "
                f"vs checkpoint {hf_cfg[field]}")

    hf_state = {}
    embed_tokens = None
    lm_head = None
    for key, tensor in _iter_safetensors(qwen_path):
        if key == "model.embed_tokens.weight":
            embed_tokens = tensor
        elif key == "lm_head.weight":
            lm_head = tensor
        else:
            hf_state[key] = tensor
    report["backbone_missing"] = model.backbone.load_hf_state_dict(hf_state, strict=True)

    assert embed_tokens is not None, "model.embed_tokens.weight not found in Qwen checkpoint"
    ref = embed_tokens
    param = model.text_emb.emb.weight
    assert param.shape == ref.shape, (param.shape, ref.shape)
    with torch.no_grad():
        param.copy_(ref.to(param.dtype))
        # Initialize the "initial" text token vector to the mean embedding.
        model.text_emb.init_vec.copy_(ref.float().mean(dim=0).to(param.dtype))
    report["text_emb"] = True

    head = lm_head if lm_head is not None else embed_tokens  # tied weights
    with torch.no_grad():
        model.text_linear.weight.copy_(head.to(model.text_linear.weight.dtype))
    report["text_head"] = True
    if not report["backbone_missing"]:
        logger.info("Qwen backbone weights loaded cleanly.")
    else:
        logger.warning("Missing backbone keys: %s", report["backbone_missing"])
    return report


# Modules we transplant from a PersonaPlex/moshiko checkpoint, and how.
# Keys inside the depformer transformer whose first dimension is stacked
# per-step weights (8 copies in the source, 16 in PPQ -> duplicate).
_STACKED_HINTS = ("self_attn.in_proj_weight", "self_attn.out_proj.weight")
# Per-step ModuleLists where entries 8..15 should be copied from entries 0..7
# (gating.N inside depformer layers, depformer_emb.N, linears.N).
_STEP_LIST_PATTERN = re.compile(r"(gating|depformer_emb|linears)\.(\d+)\.")


def _low_step_name(name: str) -> str | None:
    """For a per-step key with index >= 8, return the name with index - 8."""
    m = _STEP_LIST_PATTERN.search(name)
    if m is None:
        return None
    idx = int(m.group(2))
    if idx < 8:
        return None
    return name[: m.start(2)] + str(idx - 8) + name[m.end(2):]


def transplant_depformer(model: PPQLMModel, source_path: str | Path) -> dict:
    """Transplant depformer weights from a PersonaPlex/moshiko checkpoint.

    Returns a report dict {copied, expanded, copied_from_low, missing}.
    """
    src = dict(_iter_safetensors(source_path))
    model_sd = model.state_dict()
    prefixes = ("depformer.", "depformer_emb.", "depformer_in.", "depformer_text_emb.", "linears.")
    report = {"copied": [], "expanded": [], "copied_from_low": [], "missing": []}

    with torch.no_grad():
        for name, param in model_sd.items():
            if not name.startswith(prefixes):
                continue
            if name.startswith("depformer_in.") or name.startswith("depformer_text_emb."):
                report["missing"].append((name, "dimension/vocab mismatch, trained fresh"))
                continue
            tensor = src.get(name)
            if tensor is not None and tensor.shape == param.shape:
                param.copy_(tensor.to(param.dtype))
                report["copied"].append(name)
                continue
            if (tensor is not None and tensor.dim() >= 1 and param.dim() >= 1
                    and param.shape[0] == 2 * tensor.shape[0]
                    and param.shape[1:] == tensor.shape[1:]
                    and any(h in name for h in _STACKED_HINTS)):
                both = torch.cat([tensor, tensor], dim=0)
                param.copy_(both.to(param.dtype))
                report["expanded"].append(name)
                continue
            # Per-step entries 8..15 (gating.N, depformer_emb.N, linears.N):
            # copy from entry N-8.
            low_name = _low_step_name(name)
            if low_name is not None:
                low = src.get(low_name)
                if low is not None and low.shape == param.shape:
                    param.copy_(low.to(param.dtype))
                    report["copied_from_low"].append((name, low_name))
                    continue
            report["missing"].append((name, "not found or shape mismatch"))

    n_params = sum(1 for k in model_sd if k.startswith(prefixes))
    logger.info(
        "Depformer transplant: %d copied, %d expanded 8->16 steps, %d copied to high steps, "
        "%d left fresh (of %d depformer-side tensors).",
        len(report["copied"]), len(report["expanded"]), len(report["copied_from_low"]),
        len(report["missing"]), n_params)
    for name, why in report["missing"]:
        logger.info("  fresh: %s (%s)", name, why)
    return report

