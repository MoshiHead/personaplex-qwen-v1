"""Validation helpers for PPQ token samples (used by the HF data notebook).

check_sample / check_manifest verify the on-disk training format;
decode_channels turns code rows back into audio for listening;
loader_smoke pulls real batches through a TINY PPQ model + the real loss to
prove the training path works end to end without loading the 1.7B backbone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import PPQModelConfig, MIMI_CARD


def check_sample(path: str | Path, cfg: PPQModelConfig,
                 segment_frames: int = 1250) -> list[str]:
    """Format checks for one .npz sample. Returns a list of problems ([] = ok)."""
    errors: list[str] = []
    try:
        data = np.load(path)
    except Exception as e:  # noqa: BLE001
        return [f"cannot load: {e}"]
    if "codes" not in data or "loss_mask" not in data:
        return ["missing codes/loss_mask arrays"]
    codes, mask = data["codes"], data["loss_mask"]
    if codes.ndim != 2 or codes.shape[0] != 17:
        errors.append(f"codes shape {codes.shape} != [17, T]")
        return errors
    if mask.shape != codes.shape:
        errors.append(f"loss_mask shape {mask.shape} != codes shape {codes.shape}")
    if codes.dtype != np.int32:
        errors.append(f"codes dtype {codes.dtype} != int32")
    if mask.dtype != np.float16:
        errors.append(f"loss_mask dtype {mask.dtype} != float16")
    T = codes.shape[1]
    if T > segment_frames:
        errors.append(f"T={T} > segment_frames={segment_frames}")
    if errors:
        return errors
    audio = codes[1:].astype(np.int64)
    if audio.size and (audio.min() < 0 or audio.max() >= MIMI_CARD):
        errors.append(f"audio codes out of range [0, {MIMI_CARD})")
    text = codes[0].astype(np.int64)
    if text.size and (text.min() < 0 or text.max() >= cfg.text_card):
        errors.append(f"text tokens out of range [0, {cfg.text_card})")
    m = mask.astype(np.float32)
    if not ((m == 0) | (m == 1)).all():
        errors.append("loss_mask has values other than 0/1")
    if m[1:9].sum() == 0:
        errors.append("no supervised agent audio frames")
    return errors


def check_manifest(manifest_path: str | Path, cfg: PPQModelConfig,
                   segment_frames: int = 1250, max_check: int = 25) -> dict:
    """Stats + sampled format checks for one manifest JSONL."""
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    names = [e.get("name", Path(e["path"]).stem) for e in entries]
    dupes = len(names) - len(set(names))
    hours = sum(e.get("duration_s", e["frames"] / 12.5) for e in entries) / 3600.0
    errors: list[str] = []
    step = max(1, len(entries) // max_check) if entries else 1
    checked = entries[::step]
    for e in checked:
        for err in check_sample(e["path"], cfg, segment_frames):
            errors.append(f"{e.get('name', e['path'])}: {err}")
    return {"manifest": str(manifest_path), "entries": len(entries),
            "duplicate_names": dupes, "hours": round(hours, 2),
            "checked": len(checked), "errors": errors}


def alignment_score(words, speech) -> float:
    """Fraction of word midpoints that fall inside a speech interval."""
    if not words:
        return 1.0
    inside = sum(1 for _, s, e in words
                 if any(a <= 0.5 * (s + e) <= b for a, b in speech))
    return inside / len(words)


def decode_channels(mimi, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """codes [17, T] -> (agent_wav, user_wav) float32 mono at 24 kHz."""
    import torch
    device = next(mimi.parameters()).device
    with torch.no_grad():
        c = torch.from_numpy(np.ascontiguousarray(codes)).long()[None].to(device)
        agent = mimi.decode(c[:, 1:9])[0, 0].float().cpu().numpy()
        user = mimi.decode(c[:, 9:17])[0, 0].float().cpu().numpy()
    return agent, user


def loader_smoke(manifest_paths, tcfg, n_batches: int = 2,
                 batch_size: int = 2) -> dict:
    """Pull real batches through a tiny PPQ model + the real training loss.

    Uses the full text vocabulary (so token ranges are exercised) but a tiny
    backbone, so it runs on CPU in seconds. Returns mean losses per stream.
    """
    import torch
    from ..config import QwenBackboneConfig
    from ..model import PPQLMModel
    from ..train.loss import compute_losses
    from .dataset import TokenShardDataset, collate

    tiny_qwen = QwenBackboneConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=tcfg.model.text_card)
    # The depformer runs sequentially per frame, so it must be tiny too -
    # the default 6x1024 depformer takes minutes per 1250-frame batch on CPU.
    mcfg = PPQModelConfig(qwen=tiny_qwen,
                          depformer_dim=64, depformer_dim_feedforward=128,
                          depformer_num_heads=4, depformer_num_layers=1,
                          depformer_context=4)
    ds = TokenShardDataset(manifest_paths, mcfg,
                           segment_frames=tcfg.segment_frames, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collate,
                                         num_workers=0, drop_last=True)
    model = PPQLMModel(mcfg, device="cpu", dtype=torch.float32)
    model.eval()
    sums: dict[str, list[float]] = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            codes = batch["codes"]
            mask = batch["loss_mask"]
            lm_out = model.forward_train(codes)
            losses = compute_losses(lm_out, codes, mask, tcfg, mcfg)
            for k, v in losses.items():
                val = float(v)
                if not np.isfinite(val):
                    raise RuntimeError(f"non-finite loss '{k}': {val}")
                sums.setdefault(k, []).append(val)
    if not sums:
        raise RuntimeError("loader produced no batches")
    return {k: sum(vs) / len(vs) for k, vs in sums.items()}
