"""Stitch per-turn mono audio into a full-duplex stereo conversation.

Channel 0 = agent, channel 1 = user (PersonaPlex convention).

Timeline rules:
  - Normal alternation: next turn starts after the other speaker's last turn
    ends + a gap ~ N(0.2 s, 0.1 s); 10% of gaps are negative (slight overlap).
  - Backchannel: user utterance placed at a random point inside the previous
    agent turn (both channels active).
  - Interrupt: the previous agent turn is truncated at a random interior word
    boundary (with a short fade-out); the user turn starts 0.15-0.5 s later.
  - pause_mid: the user turn is split at a word boundary with a 0.5-1.5 s gap.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .dialogue import Dialogue
from .tts_kokoro import SynthResult, SAMPLE_RATE


@dataclass
class PlacedSegment:
    channel: int
    start: float
    wav: np.ndarray
    words: list[tuple[str, float, float]]  # absolute times

    @property
    def end(self) -> float:
        return self.start + len(self.wav) / SAMPLE_RATE


@dataclass
class StitchResult:
    stereo: np.ndarray  # float32 [2, N], 24 kHz
    agent_words: list[tuple[str, float, float]]
    stats: dict = field(default_factory=dict)


def _fade_out(wav: np.ndarray, fade_s: float = 0.03) -> np.ndarray:
    n = min(len(wav), int(fade_s * SAMPLE_RATE))
    if n > 0:
        wav = wav.copy()
        wav[-n:] *= np.linspace(1.0, 0.0, n)
    return wav


def _split_at_word(res: SynthResult, rng: random.Random) -> tuple[SynthResult, SynthResult]:
    """Split a synthesized turn at a word boundary near the middle."""
    if len(res.words) < 4:
        return res, SynthResult(np.zeros(0, dtype=np.float32), [])
    idx = rng.randint(len(res.words) // 3, 2 * len(res.words) // 3)
    cut_t = res.words[idx][1]
    cut_n = int(cut_t * SAMPLE_RATE)
    first = SynthResult(res.wav[:cut_n], res.words[:idx], res.estimated)
    second_words = [(w, s - cut_t, e - cut_t) for w, s, e in res.words[idx:]]
    second = SynthResult(res.wav[cut_n:], second_words, res.estimated)
    return first, second


def stitch_dialogue(dlg: Dialogue, turn_audio: list[SynthResult],
                    rng: random.Random, tail_s: float = 0.5,
                    user_gain_range: tuple[float, float] = (0.8, 1.0)) -> StitchResult:
    assert len(turn_audio) == len(dlg.turns)
    segments: list[PlacedSegment] = []
    agent_end = 0.0
    user_end = 0.0
    n_back = n_int = n_pause = 0
    overlap_s = 0.0

    def last_agent_segment() -> PlacedSegment | None:
        for seg in reversed(segments):
            if seg.channel == 0:
                return seg
        return None

    for turn, res in zip(dlg.turns, turn_audio):
        dur = len(res.wav) / SAMPLE_RATE
        if turn.backchannel:
            prev = last_agent_segment()
            assert prev is not None, "backchannel without a preceding agent turn"
            span = prev.end - prev.start
            lo = prev.start + 0.15 * span
            hi = prev.start + 0.85 * span
            start = rng.uniform(lo, max(lo, hi - 0.5 * dur))
            segments.append(PlacedSegment(1, start, res.wav,
                                          [(w, s + start, e + start) for w, s, e in res.words]))
            user_end = max(user_end, start + dur)
            n_back += 1
            overlap_s += max(0.0, min(dur, prev.end - start))
            continue
        if turn.interrupt:
            prev = last_agent_segment()
            assert prev is not None, "interrupt without a preceding agent turn"
            words = prev.words
            if len(words) >= 6:
                cut_idx = rng.randint(len(words) // 3, 2 * len(words) // 3)
                cut_t = words[cut_idx][1]
            else:
                cut_t = rng.uniform(prev.start + 0.4 * (prev.end - prev.start),
                                    prev.start + 0.7 * (prev.end - prev.start))
            cut_n = max(0, int((cut_t - prev.start) * SAMPLE_RATE))
            prev.wav = _fade_out(prev.wav[:cut_n])
            prev.words = [(w, s, e) for w, s, e in prev.words if e <= cut_t]
            agent_end = min(agent_end, cut_t)
            start = cut_t + rng.uniform(0.15, 0.5)
            segments.append(PlacedSegment(1, start, res.wav,
                                          [(w, s + start, e + start) for w, s, e in res.words]))
            user_end = max(user_end, start + dur)
            n_int += 1
            continue

        gap = rng.gauss(0.2, 0.1)
        if rng.random() < 0.1:
            gap = -rng.uniform(0.05, 0.3)
            overlap_s += -gap
        start = max(0.0, max(agent_end, user_end) + gap)
        if turn.speaker == "agent":
            segments.append(PlacedSegment(0, start, res.wav,
                                          [(w, s + start, e + start) for w, s, e in res.words]))
            agent_end = start + dur
        elif turn.pause_mid and len(res.words) >= 4:
            first, second = _split_at_word(res, rng)
            segments.append(PlacedSegment(1, start, first.wav,
                                          [(w, s + start, e + start) for w, s, e in first.words]))
            pause = rng.uniform(0.5, 1.5)
            start2 = start + len(first.wav) / SAMPLE_RATE + pause
            segments.append(PlacedSegment(1, start2, second.wav,
                                          [(w, s + start2, e + start2) for w, s, e in second.words]))
            user_end = start2 + len(second.wav) / SAMPLE_RATE
            n_pause += 1
        else:
            segments.append(PlacedSegment(1, start, res.wav,
                                          [(w, s + start, e + start) for w, s, e in res.words]))
            user_end = start + dur

    total_n = int((max(agent_end, user_end) + tail_s) * SAMPLE_RATE) + 1
    stereo = np.zeros((2, total_n), dtype=np.float32)
    user_gain = rng.uniform(*user_gain_range)
    for seg in segments:
        n0 = int(seg.start * SAMPLE_RATE)
        n1 = min(total_n, n0 + len(seg.wav))
        gain = 1.0 if seg.channel == 0 else user_gain
        stereo[seg.channel, n0:n1] += seg.wav[: n1 - n0] * gain

    agent_words = []
    for seg in segments:
        if seg.channel == 0:
            agent_words.extend(seg.words)
    agent_words.sort(key=lambda w: w[1])
    stats = {
        "duration_s": total_n / SAMPLE_RATE,
        "n_backchannels": n_back, "n_interrupts": n_int, "n_mid_pauses": n_pause,
        "overlap_s": overlap_s,
        "words_estimated": any(res.estimated for res in turn_audio),
    }
    return StitchResult(stereo, agent_words, stats)

