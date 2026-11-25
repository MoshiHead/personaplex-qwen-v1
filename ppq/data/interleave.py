"""Text-stream interleaver: place word tokens at their spoken frames.

One text token per 80 ms frame (12.5 Hz). Between words, frames are PAD;
the last PAD frame right before a word starts is EPAD (end-of-padding),
matching the moshi training convention.

Note on token budget: at 12.5 fps there are 0.08 s per frame. BPE tokens
encode ~0.75 words on average, and typical speech is ~2.5 words/s, so the
token stream fits comfortably; when a word's tokens would overrun the next
word's start frame, the placement cursor simply shifts the next word later
(speech and text then drift apart slightly, as in moshi training).
"""
from __future__ import annotations

import numpy as np

from ..config import FRAME_RATE_HZ


def words_to_text_stream(
    words: list[tuple[str, float, float]],
    n_frames: int,
    encode,
    pad_id: int,
    epad_id: int,
    fps: float = FRAME_RATE_HZ,
) -> np.ndarray:
    """words: [(word_text, start_s, end_s)] sorted by start. encode: text -> ids.

    Returns int32 [n_frames]."""
    stream = np.full(n_frames, pad_id, dtype=np.int32)
    cursor = 0
    for i, (word, start, _end) in enumerate(words):
        text = word if i == 0 else " " + word
        tokens = encode(text)
        if not tokens:
            continue
        start_frame = int(start * fps)
        if start_frame < cursor:
            start_frame = cursor
        if start_frame > cursor and start_frame > 0:
            # mark the final padding frame right before the word starts
            stream[start_frame - 1] = epad_id
        if start_frame >= n_frames:
            break
        end_frame = min(n_frames, start_frame + len(tokens))
        stream[start_frame:end_frame] = tokens[: end_frame - start_frame]
        cursor = end_frame
    return stream
