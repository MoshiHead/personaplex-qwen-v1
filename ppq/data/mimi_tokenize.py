"""Turn a stitched stereo conversation into 17-stream training codes.

Layout (all undelayed; forward_train applies the delays):
  row 0     : text stream (agent words, PAD/EPAD elsewhere)
  rows 1-8  : agent audio codebooks (Mimi codes of stereo channel 0)
  rows 9-16 : user audio codebooks (Mimi codes of stereo channel 1)

Prefix (matching PersonaPlex inference-time system prompts):
  [voice prompt frames] agent=voice-prompt codes, user=SINE, text=PAD
  [a silence frames]    agent=SILENCE, user=SINE, text=PAD
  [text prompt frames]  agent=SILENCE, user=SINE, text=one prompt token/frame
  [a silence frames]    agent=SILENCE, user=SINE, text=PAD

loss_mask is 1 on supervised rows of the conversation body, 0 on the prefix.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import (
    PPQModelConfig, SILENCE_TOKENS, SINE_TOKENS, SAMPLE_RATE, FRAME_RATE_HZ,
)
from .interleave import words_to_text_stream


def encode_channel(mimi, wav: np.ndarray) -> np.ndarray:
    """wav: float32 mono [N] at 24 kHz -> int codes [8, T]."""
    device = next(mimi.parameters()).device
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(wav)).float()[None, None].to(device)
        codes = mimi.encode(tensor)  # [1, 8, T]
    return codes[0].long().cpu().numpy()


def _tile(tokens: np.ndarray, n: int) -> np.ndarray:
    return np.tile(tokens[:, None], (1, n))  # [8, n]


def assemble_codes(
    agent_codes: np.ndarray,
    user_codes: np.ndarray,
    agent_words: list[tuple[str, float, float]],
    system_prompt_ids: list[int],
    voice_prompt_agent_codes: np.ndarray | None,
    cfg: PPQModelConfig,
    mask_intervals: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble 17-stream codes from pre-encoded channels.

    agent_codes/user_codes: int [8, T] Mimi codes (undelayed). agent_words:
    [(word, start_s, end_s)] relative to the body (frame 0). mask_intervals:
    [(start_s, end_s)] in body time where the loss is zeroed on ALL rows
    (e.g. redacted segments); the audio/text stay as context.

    Returns (codes int32 [17, P+T], loss_mask float16 [17, P+T]).
    """
    T = min(agent_codes.shape[1], user_codes.shape[1])
    agent_codes = agent_codes[:, :T]
    user_codes = user_codes[:, :T]
    a = cfg.audio_silence_frame_cnt

    prefix_parts = []
    if voice_prompt_agent_codes is not None:
        vp = voice_prompt_agent_codes.astype(np.int32)
        prefix_parts.append((vp, _tile(SINE_TOKENS, vp.shape[1]), "pad"))
    prefix_parts.append((_tile(SILENCE_TOKENS, a), _tile(SINE_TOKENS, a), "pad"))
    if len(system_prompt_ids) > 0:
        n = len(system_prompt_ids)
        prefix_parts.append((_tile(SILENCE_TOKENS, n), _tile(SINE_TOKENS, n), "prompt"))
    prefix_parts.append((_tile(SILENCE_TOKENS, a), _tile(SINE_TOKENS, a), "pad"))

    P = sum(part[0].shape[1] for part in prefix_parts)
    total = P + T
    codes = np.full((17, total), cfg.text_pad_id, dtype=np.int32)
    loss_mask = np.zeros((17, total), dtype=np.float16)

    pos = 0
    for agent_part, user_part, kind in prefix_parts:
        n = agent_part.shape[1]
        codes[1:9, pos: pos + n] = agent_part
        codes[9:17, pos: pos + n] = user_part
        if kind == "prompt":
            codes[0, pos: pos + n] = np.asarray(system_prompt_ids, dtype=np.int32)
        pos += n

    text_stream = words_to_text_stream(
        agent_words, T,
        encode=lambda s: _encode_ids(s),
        pad_id=cfg.text_pad_id, epad_id=cfg.text_epad_id,
        fps=FRAME_RATE_HZ,
    )
    codes[0, P:] = text_stream
    codes[1:9, P:] = agent_codes
    codes[9:17, P:] = user_codes
    loss_mask[0, P:] = 1.0      # text stream supervised (PAD down-weighted in loss)
    loss_mask[1:9, P:] = 1.0    # agent audio supervised
    loss_mask[9:17, P:] = 1.0   # user audio rows present; weight set in train config
    for start_s, end_s in mask_intervals or []:
        f0 = max(0, int(start_s * FRAME_RATE_HZ))
        f1 = min(T, int(np.ceil(end_s * FRAME_RATE_HZ)))
        if f1 > f0:
            loss_mask[:, P + f0: P + f1] = 0.0
    return codes, loss_mask


def build_conversation_codes(
    stereo: np.ndarray,
    agent_words: list[tuple[str, float, float]],
    system_prompt_ids: list[int],
    voice_prompt_agent_codes: np.ndarray | None,
    mimi,
    cfg: PPQModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (codes int32 [17, T], loss_mask float16 [17, T])."""
    agent_codes = encode_channel(mimi, stereo[0])
    user_codes = encode_channel(mimi, stereo[1])
    return assemble_codes(agent_codes, user_codes, agent_words, system_prompt_ids,
                          voice_prompt_agent_codes, cfg)


# Set by the worker before calling build_conversation_codes (avoids passing
# the tokenizer through every call site).
_ENCODE_FN = None


def set_text_encoder(fn):
    global _ENCODE_FN
    _ENCODE_FN = fn


def _encode_ids(text: str) -> list[int]:
    assert _ENCODE_FN is not None, "call ppq.data.mimi_tokenize.set_text_encoder first"
    return _ENCODE_FN(text)


def build_monologue_codes(
    speech: np.ndarray,
    mimi,
    cfg: PPQModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage-A monologue sample (e.g. LibriTTS-R): agent speech only, user
    channel digital silence, text stream PAD (text loss masked out)."""
    agent_codes = encode_channel(mimi, speech)
    silence = np.zeros_like(speech)
    user_codes = encode_channel(mimi, silence)
    T = min(agent_codes.shape[1], user_codes.shape[1])
    a = cfg.audio_silence_frame_cnt
    total = a + T
    codes = np.full((17, total), cfg.text_pad_id, dtype=np.int32)
    loss_mask = np.zeros((17, total), dtype=np.float16)
    codes[1:9, :a] = _tile(SILENCE_TOKENS, a)
    codes[9:17, :a] = _tile(SINE_TOKENS, a)
    codes[1:9, a:] = agent_codes[:, :T]
    codes[9:17, a:] = user_codes[:, :T]
    loss_mask[1:9, a:] = 1.0   # supervise agent audio only
    return codes, loss_mask
