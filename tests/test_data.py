"""Data pipeline unit tests (interleaver, stitcher, dataset)."""
import json
import random

import numpy as np
import pytest

from ppq.config import PPQModelConfig
from ppq.data.dialogue import Dialogue, Turn
from ppq.data.interleave import words_to_text_stream
from ppq.data.stitch import stitch_dialogue
from ppq.data.tts_kokoro import SynthResult, SAMPLE_RATE


def fake_encode(text: str) -> list[int]:
    # 1 token per character word-ish, deterministic
    return [ord(c) % 100 + 1 for c in text[:3]]


def test_interleave_basic():
    words = [("hello", 0.0, 0.4), ("world", 0.8, 1.2)]
    stream = words_to_text_stream(words, 40, fake_encode, pad_id=98, epad_id=99, fps=12.5)
    # "hello" -> tokens at frames 0..2 (encode takes first 3 chars)
    assert stream[0] == ord("h") % 100 + 1
    assert stream[2] == ord("l") % 100 + 1
    # frames 3..9 padding; frame 9 is the last pad before "world" at frame 10 -> EPAD
    assert (stream[3:9] == 98).all()
    assert stream[9] == 99
    # "world" -> " world" tokens (first 3 chars: ' ', 'w', 'o') at frames 10..12
    assert stream[10] == ord(" ") % 100 + 1
    assert stream[11] == ord("w") % 100 + 1
    assert stream[12] == ord("o") % 100 + 1
    assert (stream[13:] == 98).all()


def test_interleave_overflow_shifts_cursor():
    # words spoken faster than the token budget: cursor shifts placement later
    words = [("a", 0.0, 0.05), ("b", 0.08, 0.12)]
    stream = words_to_text_stream(words, 20, fake_encode, pad_id=98, epad_id=99, fps=12.5)
    # "a" -> 1 token at frame 0; " b" -> 2 tokens would start at frame 1 (no gap -> no EPAD)
    assert stream[0] == ord("a") % 100 + 1
    assert stream[1] == ord(" ") % 100 + 1
    assert stream[2] == ord("b") % 100 + 1


def make_dialogue() -> Dialogue:
    return Dialogue(
        system_prompt="You are a test assistant.",
        turns=[
            Turn("agent", "This is a fairly long agent turn that can be overlapped easily."),
            Turn("user", "mm-hmm", backchannel=True),
            Turn("agent", "Another long agent turn that is going to be interrupted midway."),
            Turn("user", "wait stop", interrupt=True),
            Turn("user", "Let me think about that for a second ... okay I have an idea now.", pause_mid=True),
            Turn("agent", "Sure, go ahead."),
        ],
    )


def make_turn_audio(dlg: Dialogue) -> list[SynthResult]:
    rng = np.random.default_rng(0)
    out = []
    for turn in dlg.turns:
        words = turn.text.split()
        dur = max(0.8, 0.25 * len(words))
        n = int(dur * SAMPLE_RATE)
        wav = (0.1 * rng.standard_normal(n)).astype(np.float32)
        step = dur / len(words)
        wt = [(w, i * step, (i + 1) * step) for i, w in enumerate(words)]
        out.append(SynthResult(wav, wt))
    return out


def test_stitch_events():
    dlg = make_dialogue()
    audio = make_turn_audio(dlg)
    rng = random.Random(0)
    res = stitch_dialogue(dlg, audio, rng)
    assert res.stereo.shape[0] == 2
    assert res.stats["n_backchannels"] == 1
    assert res.stats["n_interrupts"] == 1
    assert res.stats["n_mid_pauses"] == 1
    # Agent words exist and are sorted
    starts = [w[1] for w in res.agent_words]
    assert starts == sorted(starts)
    # The interrupted agent turn (turn idx 2) must have fewer words than synthesized
    full_words = len(dlg.turns[2].text.split())
    agent2_words = [w for w, s, e in res.agent_words if w in dlg.turns[2].text.split()]
    assert len(agent2_words) < full_words + len(dlg.turns[0].text.split())


def test_dataset_crop_and_pad(tmp_path):
    from ppq.data.dataset import TokenShardDataset
    cfg = PPQModelConfig()
    T = 100
    codes = np.random.randint(0, 2048, (17, T), dtype=np.int32)
    mask = np.ones((17, T), dtype=np.float16)
    path = tmp_path / "s.npz"
    np.savez_compressed(path, codes=codes, loss_mask=mask)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"path": str(path), "frames": T}) + "\n")

    ds = TokenShardDataset([manifest], cfg, segment_frames=50, seed=0, epoch_multiplier=2)
    assert len(ds) == 2
    item = ds[0]
    assert item["codes"].shape == (17, 50)
    assert item["loss_mask"].shape == (17, 50)

    ds_pad = TokenShardDataset([manifest], cfg, segment_frames=150, seed=0)
    item = ds_pad[0]
    assert item["codes"].shape == (17, 150)
    assert (item["loss_mask"][:, 100:] == 0).all()
    assert item["codes"][0, 100] == cfg.text_pad_id
