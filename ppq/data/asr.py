"""Whisper word-timestamp ASR with a per-session/channel JSON cache.

Used for the sources without usable word timestamps: the task sets (full
transcription of detected speech regions) and turn-104h (only the long SRT
segments; short segments keep their SRT text with even word spreading).

Cache layout: <data_root>/asr_cache/<source>/<session>/ch<k>.json with
{"regions": [[s, e], ...], "words": [[word, s, e], ...]} in session time.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"


def cache_path(data_root: str | Path, source: str, session_id: str,
               channel: int) -> Path:
    from .oto_sources import safe_name
    return (Path(data_root) / "asr_cache" / source / safe_name(session_id)
            / f"ch{channel}.json")


def load_cached(path: str | Path) -> dict | None:
    """Full cache dict ({"regions": ..., "words": ...}) or None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        logger.warning("ignoring unreadable ASR cache %s", path)
        return None


def load_cached_words(path: str | Path) -> list[tuple[str, float, float]] | None:
    d = load_cached(path)
    if d is None:
        return None
    try:
        return [(w, float(s), float(e)) for w, s, e in d.get("words", [])]
    except (ValueError, TypeError):
        logger.warning("ignoring malformed ASR cache %s", path)
        return None


def save_cached_words(path: str | Path, regions, words) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({
        "regions": [[float(s), float(e)] for s, e in regions],
        "words": [[w, float(s), float(e)] for w, s, e in words],
    }), encoding="utf-8")
    os.replace(tmp, path)


class WhisperASR:
    """transformers Whisper pipeline returning (word, start_s, end_s)."""

    def __init__(self, model_id: str = DEFAULT_ASR_MODEL, device: str = "cuda"):
        import torch
        from transformers import pipeline
        dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
        if dev >= 0:
            logger.info("Whisper ASR on GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning(
                "CUDA not available - Whisper ASR falling back to CPU, which is "
                "VERY slow. Check CUDA_VISIBLE_DEVICES and the torch build.")
        dtype = torch.float16 if dev >= 0 else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model_id,
                             device=dev, torch_dtype=dtype)

    def transcribe_words(self, wav: np.ndarray, sr: int) -> list[tuple[str, float, float]]:
        out = self.pipe(
            {"raw": np.asarray(wav, dtype=np.float32), "sampling_rate": int(sr)},
            return_timestamps="word", chunk_length_s=30)
        words: list[tuple[str, float, float]] = []
        for ch in out.get("chunks", []):
            w = (ch.get("text") or "").strip()
            s, e = ch.get("timestamp") or (None, None)
            if not w or s is None:
                continue
            if e is None or e <= s:
                e = s + 0.2
            words.append((w, float(s), float(e)))
        return words


def transcribe_regions(engine, wav: np.ndarray, sr: int, regions,
                       accept=None) -> list[tuple[str, float, float]]:
    """Transcribe each (start_s, end_s) region of wav; returned word times are
    offset back to session time. accept(start, end, region) optionally filters
    words (e.g. to the unpadded region span)."""
    words: list[tuple[str, float, float]] = []
    for (s, e) in regions:
        seg = wav[int(s * sr): int(e * sr)]
        if len(seg) < sr // 10:
            continue
        for w, ws, we in engine.transcribe_words(seg, sr):
            if accept is not None and not accept(ws + s, we + s, (s, e)):
                continue
            words.append((w, ws + s, we + s))
    words.sort(key=lambda x: (x[1], x[2]))
    return words
