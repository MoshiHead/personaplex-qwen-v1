"""Turn real two-speaker sessions into PPQ training samples.

Pipeline per session (see ppq.data.worker --stage real):
  resample to 24 kHz + loudness-normalize each channel -> channel-separation
  check -> chunk at mutual silence (20-80 s) -> per chunk emit TWO role-swap
  samples (agent = ch0 or ch1; a role is skipped when that speaker talks less
  than ``min_agent_speech`` of the chunk) -> Mimi-encode each channel once per
  chunk -> assemble 17-stream codes with the redaction intervals masked out.

Chunk length is capped so that chunk + prefix (voice prompt + system prompt)
always fits inside the training crop (segment_frames = 1250 frames = 100 s).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import PPQModelConfig, SAMPLE_RATE
from . import mimi_tokenize
from .external import normalize_loudness
from .oto_sources import SessionMeta, merge_intervals, safe_name

logger = logging.getLogger(__name__)


@dataclass
class PreparedSession:
    meta: SessionMeta
    stereo: np.ndarray   # float32 [2, N] at SAMPLE_RATE, loudness-normalized
    speech: list         # per channel [(s, e)]
    words: list          # per channel [(word, s, e)]

    @property
    def duration_s(self) -> float:
        return self.stereo.shape[1] / SAMPLE_RATE


@dataclass
class Chunk:
    start_s: float
    end_s: float


def resample_to_24k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return np.asarray(wav, dtype=np.float32)
    import sphn
    return sphn.resample(np.asarray(wav, dtype=np.float32)[None], sr, SAMPLE_RATE)[0]


def prepare_stereo(ch0: np.ndarray, ch1: np.ndarray, sr: int) -> np.ndarray:
    """Resample + loudness-normalize both channels -> float32 [2, N] at 24 kHz."""
    a = normalize_loudness(resample_to_24k(ch0, sr), SAMPLE_RATE)
    b = normalize_loudness(resample_to_24k(ch1, sr), SAMPLE_RATE)
    n = min(len(a), len(b))
    return np.stack([a[:n], b[:n]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Speech activity
# ---------------------------------------------------------------------------

def energy_vad(wav: np.ndarray, sr: int, frame_s: float = 0.05,
               noise_percentile: float = 20.0, thresh_mult: float = 3.5,
               min_region_s: float = 0.15, merge_gap_s: float = 0.3
               ) -> list[tuple[float, float]]:
    """Simple adaptive energy VAD -> speech intervals (seconds)."""
    n = max(1, int(frame_s * sr))
    frames = len(wav) // n
    if frames == 0:
        return []
    rms = np.sqrt(np.mean(wav[:frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    floor = float(np.percentile(rms, noise_percentile))
    thr = max(floor * thresh_mult, 1e-5)
    active = rms > thr
    regions: list[tuple[float, float]] = []
    i = 0
    while i < frames:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < frames and active[j + 1]:
            j += 1
        regions.append((i * n / sr, (j + 1) * n / sr))
        i = j + 1
    regions = merge_intervals(regions, gap_s=merge_gap_s)
    return [(s, e) for s, e in regions if e - s >= min_region_s]


def merge_regions(regions, max_region_s: float = 60.0) -> list[tuple[float, float]]:
    """Group adjacent speech regions into ASR inputs of at most max_region_s."""
    out: list[tuple[float, float]] = []
    cur: tuple[float, float] | None = None
    for s, e in regions:
        if cur is None:
            cur = (s, e)
        elif e - cur[0] <= max_region_s:
            cur = (cur[0], e)
        else:
            out.append(cur)
            cur = (s, e)
    if cur is not None:
        out.append(cur)
    return out


def speech_ratio(speech, start: float, end: float) -> float:
    """Fraction of [start, end] covered by speech intervals."""
    overlap = sum(max(0.0, min(e, end) - max(s, start)) for s, e in speech)
    return overlap / max(1e-9, end - start)


def separation_db(wav0: np.ndarray, wav1: np.ndarray, speech0, speech1,
                  sr: int, window_s: float = 0.1, max_windows: int = 4000) -> float:
    """Median dB ratio between the active channel and the (supposedly silent)
    other channel, over windows where exactly one channel speaks. Low values
    indicate heavy channel bleed. Returns inf when there is too little data."""
    n = max(1, int(window_s * sr))
    frames = min(len(wav0), len(wav1)) // n
    if frames == 0:
        return float("inf")

    def active_at(speech, t):
        return any(s <= t <= e for s, e in speech)

    ratios = []
    step = max(1, frames // max_windows)
    for i in range(0, frames, step):
        t = (i + 0.5) * n / sr
        a0, a1 = active_at(speech0, t), active_at(speech1, t)
        if a0 == a1:
            continue
        r0 = float(np.sqrt(np.mean(wav0[i * n:(i + 1) * n] ** 2) + 1e-12))
        r1 = float(np.sqrt(np.mean(wav1[i * n:(i + 1) * n] ** 2) + 1e-12))
        ratios.append(r0 / max(r1, 1e-8) if a0 else r1 / max(r0, 1e-8))
    if len(ratios) < 20:
        return float("inf")
    return float(20.0 * np.log10(np.median(ratios)))


# ---------------------------------------------------------------------------
# Chunking at mutual silence
# ---------------------------------------------------------------------------

def mutual_silence_cuts(speech0, speech1, dur_s: float,
                        min_silence_s: float = 0.8) -> list[float]:
    """Midpoints of gaps where BOTH channels are silent for >= min_silence_s."""
    both = merge_intervals(list(speech0) + list(speech1))
    cuts = []
    prev_end = 0.0
    for s, e in both:
        if s - prev_end >= min_silence_s:
            cuts.append((prev_end + s) / 2.0)
        prev_end = max(prev_end, e)
    if dur_s - prev_end >= min_silence_s:
        cuts.append((prev_end + dur_s) / 2.0)
    return cuts


def chunk_session(speech0, speech1, dur_s: float, min_chunk_s: float = 20.0,
                  max_chunk_s: float = 80.0, min_silence_s: float = 0.8,
                  min_tail_s: float = 8.0) -> list[Chunk]:
    """Greedy chunks of [min_chunk_s, max_chunk_s], cut at the latest mutual
    silence before the max length (hard cut when there is none)."""
    cuts = mutual_silence_cuts(speech0, speech1, dur_s, min_silence_s)
    chunks: list[Chunk] = []
    start = 0.0
    while dur_s - start > max_chunk_s:
        limit = start + max_chunk_s
        cands = [c for c in cuts if start + min_chunk_s <= c <= limit]
        cut = cands[-1] if cands else limit
        chunks.append(Chunk(start, cut))
        start = cut
    tail = dur_s - start
    if tail >= min_tail_s:
        chunks.append(Chunk(start, dur_s))
    elif chunks and tail > 0 and chunks[-1].end_s - chunks[-1].start_s + tail <= max_chunk_s:
        chunks[-1].end_s = dur_s
    return chunks


# ---------------------------------------------------------------------------
# Sample building
# ---------------------------------------------------------------------------

def filter_words_for_chunk(words, start: float, end: float, redactions
                           ) -> list[tuple[str, float, float]]:
    """Words whose midpoint is inside [start, end) and NOT inside a redaction
    interval, shifted to chunk-relative time."""
    out = []
    for w, s, e in words:
        mid = 0.5 * (s + e)
        if not (start <= mid < end):
            continue
        if any(rs <= mid <= re for rs, re in redactions):
            continue
        out.append((w, s - start, e - start))
    return out


def clip_voice_prompt(wav_ch: np.ndarray, speech_ch, chunk_start: float,
                      chunk_end: float, seconds: float,
                      sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """A `seconds`-long clip of this speaker, preferring speech OUTSIDE the
    chunk (so the voice prompt is not part of the training body)."""
    need = seconds + 0.5
    outside = [(s, e) for s, e in speech_ch
               if e <= chunk_start or s >= chunk_end]
    inside = [(s, e) for s, e in speech_ch
              if not (e <= chunk_start or s >= chunk_end)]
    for pool in (outside, inside):
        pool = sorted(pool, key=lambda iv: iv[1] - iv[0], reverse=True)
        for s, e in pool:
            if e - s >= need:
                mid = 0.5 * (s + e)
                c0 = int((mid - seconds / 2.0) * sr)
                c1 = c0 + int(seconds * sr)
                return wav_ch[max(0, c0):c1].astype(np.float32)
    return None


def build_chunk_samples(prep: PreparedSession, mimi, tokenizer, cfg: PPQModelConfig,
                        *, voice_prompt_mode: str = "none",
                        min_agent_speech: float = 0.15,
                        voice_prompt_seconds: float = 8.0,
                        segment_frames: int = 1250,
                        min_chunk_s: float = 20.0, max_chunk_s: float = 80.0,
                        min_silence_s: float = 0.8):
    """Yields (name, codes, loss_mask, info) for every kept chunk x role."""
    sid = safe_name(prep.meta.session_id)
    chunks = chunk_session(prep.speech[0], prep.speech[1], prep.duration_s,
                           min_chunk_s, max_chunk_s, min_silence_s)
    for ci, ch in enumerate(chunks):
        n0 = int(ch.start_s * SAMPLE_RATE)
        n1 = int(ch.end_s * SAMPLE_RATE)
        seg = prep.stereo[:, n0:n1]
        codes_per_ch = [mimi_tokenize.encode_channel(mimi, seg[0]),
                        mimi_tokenize.encode_channel(mimi, seg[1])]
        for role in (0, 1):
            agent_ch, user_ch = role, 1 - role
            ratio = speech_ratio(prep.speech[agent_ch], ch.start_s, ch.end_s)
            if ratio < min_agent_speech:
                continue
            words = filter_words_for_chunk(prep.words[agent_ch], ch.start_s,
                                           ch.end_s, prep.meta.redactions)
            prompt_ids = tokenizer.encode_system_prompt(
                prep.meta.system_prompts[agent_ch])
            vp_codes = None
            if voice_prompt_mode == "same_speaker":
                clip = clip_voice_prompt(prep.stereo[agent_ch], prep.speech[agent_ch],
                                         ch.start_s, ch.end_s, voice_prompt_seconds)
                if clip is not None and len(clip) >= SAMPLE_RATE // 2:
                    vp_codes = mimi_tokenize.encode_channel(mimi, clip)
            masks = [(s - ch.start_s, e - ch.start_s) for s, e in prep.meta.redactions
                     if e > ch.start_s and s < ch.end_s]
            codes, loss_mask = mimi_tokenize.assemble_codes(
                codes_per_ch[agent_ch], codes_per_ch[user_ch], words, prompt_ids,
                vp_codes, cfg, mask_intervals=masks)
            if codes.shape[1] > segment_frames:
                logger.warning("skipping %s c%d r%d: %d frames > segment_frames=%d",
                               sid, ci, role, codes.shape[1], segment_frames)
                continue
            name = f"real-{prep.meta.source}-{sid}-c{ci:04d}-r{role}"
            info = {"session_id": prep.meta.session_id, "chunk_idx": ci,
                    "agent_channel": agent_ch,
                    "duration_s": round(ch.end_s - ch.start_s, 3),
                    "agent_speech_ratio": round(ratio, 3),
                    "n_agent_words": len(words)}
            yield name, codes, loss_mask, info


