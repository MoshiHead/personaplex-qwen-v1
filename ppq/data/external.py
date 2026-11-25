"""External corpus adapters (Stage A monologue data).

LibriTTS-R (CC-BY-4.0): read-aloud audiobook speech at 24 kHz. Utterances are
short, so consecutive utterances from the same speaker are concatenated with
small gaps into ~20-90 s monologues. Loudness-normalized like PersonaPlex.

Grouping is deterministic and independent of the shard layout: every shard
streams the same utterance sequence and computes the same group boundaries,
but only the owning shard loudness-normalizes and concatenates the group.
"""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

TARGET_LUFS = -20.0


def _normalize(wav: np.ndarray, sr: int) -> np.ndarray:
    import pyloudnorm as pyln
    if len(wav) < sr // 10:
        return wav
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(wav)
    except ValueError:
        return wav
    if not np.isfinite(loudness):
        return wav
    return pyln.normalize.loudness(wav, loudness, TARGET_LUFS).astype(np.float32)


def iter_libritts_monologues(max_seconds: float = 100 * 3600,
                             min_seconds: float = 20.0,
                             max_seconds_per: float = 90.0,
                             shard: int | None = None,
                             num_shards: int = 1,
                             repo: str = "blinoff/libritts_r",
                             config: str = "clean",
                             split: str = "train.clean.100"
                             ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (group_index, mono float32 24 kHz wav) for groups owned by this
    shard (group_index % num_shards == shard). With shard=None, yields all."""
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True)
    sr = 24000
    gap_s = 0.3
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    buf: list[np.ndarray] = []
    buf_s = 0.0
    total_s = 0.0
    group_idx = 0
    last_speaker = None

    def flush() -> tuple[int, np.ndarray | None, float]:
        nonlocal buf, buf_s, group_idx
        idx, dur = group_idx, buf_s
        group_idx += 1
        owned = shard is None or idx % num_shards == shard
        out = None
        if owned:
            out = np.concatenate([_normalize(w, sr) for w in buf])
        buf, buf_s = [], 0.0
        return idx, out, dur

    for item in ds:
        audio = item["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        if audio["sampling_rate"] != sr:
            import sphn
            wav = sphn.resample(wav[None], audio["sampling_rate"], sr)[0]
        speaker = item.get("speaker_id", None)
        dur = len(wav) / sr
        if total_s + dur > max_seconds:
            break
        if speaker != last_speaker or buf_s + dur > max_seconds_per:
            if buf_s >= min_seconds:
                idx, out, gdur = flush()
                total_s += gdur
                if out is not None:
                    yield idx, out
            else:
                buf, buf_s = [], 0.0
            last_speaker = speaker
        buf.append(wav)
        buf.append(gap)
        buf_s += dur + gap_s
    if buf_s >= min_seconds:
        idx, out, _ = flush()
        if out is not None:
            yield idx, out
