"""Training dataset over pre-tokenized shards.

On-disk format (written by ppq.data.worker): one ``.npz`` per sample with
  - codes:     int32  [17, T]
  - loss_mask: float16 [17, T]
plus a manifest JSONL: {"path": ..., "frames": T, "source": ...}.

The dataset returns random crops of ``segment_frames`` (padded when the
sample is shorter).
"""
from __future__ import annotations

import bisect
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PPQModelConfig, SILENCE_TOKENS, SINE_TOKENS


class TokenShardDataset(Dataset):
    def __init__(self, manifest_paths: list[str | Path], cfg: PPQModelConfig,
                 segment_frames: int = 1250, seed: int = 0,
                 epoch_multiplier: int = 20,
                 source_weights: dict | None = None):
        self.cfg = cfg
        self.segment_frames = segment_frames
        self.entries: list[dict] = []
        for mp in manifest_paths:
            with open(mp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))
        assert self.entries, f"empty manifest(s): {manifest_paths}"
        self.rng = random.Random(seed)
        # Logical epoch = several passes with different random crops.
        self.epoch_len = len(self.entries) * epoch_multiplier
        # Optional per-source sampling weights (manifest "source" -> weight).
        self._cum_weights: list[float] | None = None
        if source_weights:
            acc = 0.0
            cum = []
            for e in self.entries:
                acc += float(source_weights.get(e.get("source", ""), 1.0))
                cum.append(acc)
            if acc <= 0:
                raise ValueError(f"source_weights {source_weights} give every "
                                 "entry zero weight")
            self._cum_weights = cum
            self._weight_total = acc

    def __len__(self) -> int:
        return self.epoch_len

    def _pick_entry(self, idx: int) -> dict:
        if self._cum_weights is None:
            return self.entries[idx % len(self.entries)]
        i = bisect.bisect_left(self._cum_weights,
                               self.rng.random() * self._weight_total)
        return self.entries[min(i, len(self.entries) - 1)]

    def __getitem__(self, idx: int) -> dict:
        entry = self._pick_entry(idx)
        data = np.load(entry["path"])
        codes = data["codes"].astype(np.int64)          # [17, T]
        mask = data["loss_mask"].astype(np.float32)     # [17, T]
        T = codes.shape[1]
        S = self.segment_frames
        if T >= S:
            start = self.rng.randint(0, T - S)
            codes = codes[:, start: start + S]
            mask = mask[:, start: start + S]
        else:
            codes, mask = self._pad(codes, mask, S - T)
        return {
            "codes": torch.from_numpy(np.ascontiguousarray(codes)),
            "loss_mask": torch.from_numpy(np.ascontiguousarray(mask)),
        }

    def _pad(self, codes: np.ndarray, mask: np.ndarray, pad: int) -> tuple[np.ndarray, np.ndarray]:
        T = codes.shape[1]
        out = np.full((17, T + pad), self.cfg.text_pad_id, dtype=np.int64)
        out[:, :T] = codes
        out[1:9, T:] = SILENCE_TOKENS[:, None]
        out[9:17, T:] = SINE_TOKENS[:, None]
        mout = np.zeros((17, T + pad), dtype=np.float32)
        mout[:, :T] = mask
        return out, mout


def collate(batch: list[dict]) -> dict:
    return {
        "codes": torch.stack([b["codes"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch]),
    }
