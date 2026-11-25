"""External corpus adapters (Stage A monologue data).

LibriTTS-R (CC-BY-4.0): read-aloud audiobook speech at 24 kHz. Utterances are
short, so consecutive utterances from the same speaker are concatenated with
small gaps into ~20-90 s monologues. Loudness-normalized like PersonaPlex.
"""
from __future__ import annotations

import logging

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
                             repo: str = "blinoff/libritts_r",
                             config: str = "clean",
                             split: str = "train.clean.100"):
    """Yield mono float32 24 kHz monologues from LibriTTS-R (streaming)."""
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True)
    sr = 24000
    gap = np.zeros(int(0.3 * sr), dtype=np.float32)
    buf: list[np.ndarray] = []
    buf_s = 0.0
    total_s = 0.0
    last_speaker = None
    for item in ds:
        audio = item["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        if audio["sampling_rate"] != sr:
            import sphn
            wav = sphn.resample(wav[None], audio["sampling_rate"], sr)[0]
        speaker = item.get("speaker_id", None)
        wav = _normalize(wav, sr)
        dur = len(wav) / sr
        if total_s + dur > max_seconds:
            break
        if speaker != last_speaker or buf_s + dur > max_seconds_per:
            if buf_s >= min_seconds:
                total_s += buf_s
                yield np.concatenate(buf)
            buf, buf_s = [], 0.0
            last_speaker = speaker
        buf.append(wav)
        buf.append(gap)
        buf_s += dur + 0.3
    if buf_s >= min_seconds:
        yield np.concatenate(buf)
