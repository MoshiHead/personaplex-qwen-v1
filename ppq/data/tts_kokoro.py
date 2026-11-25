"""Kokoro-82M TTS wrapper producing 24 kHz mono audio + word timestamps.

Kokoro (Apache-2.0) runs per dialogue turn. Word-level timestamps come from
kokoro's token results (misaki MToken start_ts/end_ts, in seconds); if the
installed kokoro version does not expose them we fall back to an even
distribution over the segment duration and set ``estimated=True``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# A small diverse set of Kokoro v1.0 voices (American English).
AGENT_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah"]
USER_VOICES = [
    "am_adam", "am_michael", "af_alloy", "af_aoede", "af_jessica",
    "af_kore", "af_nova", "af_river", "af_sky", "am_echo", "am_eric",
    "am_fenrir", "am_liam", "am_onyx", "am_puck", "bf_alice", "bf_emma",
    "bm_george", "bm_lewis",
]


@dataclass
class SynthResult:
    wav: np.ndarray           # float32 mono, 24 kHz
    words: list[tuple[str, float, float]]  # (word, start_s, end_s), relative to wav
    estimated: bool = False   # True if word timestamps were not provided by kokoro


class KokoroTTS:
    def __init__(self, lang_code: str = "a"):
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code=lang_code)

    def synth(self, text: str, voice: str, speed: float = 1.0) -> SynthResult:
        wavs: list[np.ndarray] = []
        words: list[tuple[str, float, float]] = []
        cursor = 0.0
        saw_tokens = False
        for result in self.pipeline(text, voice=voice, speed=speed):
            audio = getattr(result, "audio", None)
            if audio is None:  # older API yields tuples (graphemes, phonemes, audio)
                audio = result[2]
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            tokens = getattr(result, "tokens", None) or []
            for tok in tokens:
                w = getattr(tok, "text", None)
                s = getattr(tok, "start_ts", None)
                e = getattr(tok, "end_ts", None)
                if w and w.strip() and s is not None and e is not None:
                    words.append((w.strip(), cursor + float(s), cursor + float(e)))
                    saw_tokens = True
            wavs.append(audio)
            cursor += len(audio) / SAMPLE_RATE
        wav = np.concatenate(wavs) if wavs else np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        if not saw_tokens:
            words = _even_word_times(text, len(wav) / SAMPLE_RATE)
            return SynthResult(wav, words, estimated=True)
        return SynthResult(wav, words)


def _even_word_times(text: str, dur: float) -> list[tuple[str, float, float]]:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return []
    step = dur / len(words)
    return [(w, i * step, (i + 1) * step) for i, w in enumerate(words)]
