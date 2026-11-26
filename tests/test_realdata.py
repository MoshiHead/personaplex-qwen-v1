"""Tests for the real-conversation (HF otoSpeech) data pipeline:
assemble_codes refactor, SRT parsing, redaction normalization, tar range
reader, chunking/role swap, dataset source weights, validation helpers and
the real-stage worker resume behaviour. All CPU/offline.
"""
import io
import json
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import ppq.checkpoint
import ppq.text_tokenizer
from ppq.config import FRAME_RATE_HZ, PPQModelConfig, SAMPLE_RATE
from ppq.data import mimi_tokenize, oto_sources, realconv, worker
from ppq.progress import read_jsonl


class FakeMimi(torch.nn.Module):
    """Deterministic content-dependent codes in [0, 2048)."""

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def encode(self, tensor):
        t = max(1, int(np.ceil(tensor.shape[-1] / 1920)))
        base = int(tensor.shape[-1]) % 2048
        rows = torch.arange(8).view(1, 8, 1)
        cols = torch.arange(t).view(1, 1, t)
        return ((base + 7 * rows + cols) % 2048).long()


class FakeTokenizer:
    def __init__(self, *a, **k):
        pass

    def encode(self, text):
        return [ord(c) % 100 + 1 for c in text[:3]]

    def encode_system_prompt(self, text):
        return [7, 7, 7, 7]


@pytest.fixture(autouse=True)
def text_encoder():
    mimi_tokenize.set_text_encoder(FakeTokenizer().encode)


def _cfg():
    return PPQModelConfig()


# ---------------------------------------------------------------------------
# assemble_codes (extracted from build_conversation_codes)
# ---------------------------------------------------------------------------

def test_assemble_codes_matches_build_conversation_codes():
    cfg = _cfg()
    mimi = FakeMimi()
    rng = np.random.default_rng(0)
    stereo = (0.05 * rng.standard_normal((2, SAMPLE_RATE * 5))).astype(np.float32)
    words = [("hello", 0.5, 0.9), ("world", 1.2, 1.7)]
    vp = (np.arange(80).reshape(8, 10) % 2048).astype(np.int64)
    prompt = [7, 7, 7, 7]

    ref_codes, ref_mask = mimi_tokenize.build_conversation_codes(
        stereo, words, prompt, vp, mimi, cfg)
    agent_codes = mimi_tokenize.encode_channel(mimi, stereo[0])
    user_codes = mimi_tokenize.encode_channel(mimi, stereo[1])
    for mi in (None, []):
        codes, mask = mimi_tokenize.assemble_codes(
            agent_codes, user_codes, words, prompt, vp, cfg, mask_intervals=mi)
        assert np.array_equal(codes, ref_codes)
        assert np.array_equal(mask, ref_mask)


def test_assemble_codes_mask_intervals():
    cfg = _cfg()
    T = 100
    agent = np.zeros((8, T), dtype=np.int64)
    user = np.ones((8, T), dtype=np.int64)
    codes, mask = mimi_tokenize.assemble_codes(
        agent, user, [], [], None, cfg,
        mask_intervals=[(1.0, 2.0), (3.04, 3.04)])  # 2nd: f0 == f1 -> skipped
    assert codes.dtype == np.int32 and mask.dtype == np.float16
    a = cfg.audio_silence_frame_cnt
    P = a + a  # no voice prompt, no text prompt
    f0, f1 = int(1.0 * FRAME_RATE_HZ), int(np.ceil(2.0 * FRAME_RATE_HZ))
    assert (mask[:, P + f0: P + f1] == 0).all()   # redacted span: all rows
    assert (mask[:, P:P + f0] == 1).all()
    assert (mask[:, P + f1:] == 1).all()
    # audio codes stay as context (not touched by masking)
    assert (codes[1:9, P:] == 0).all() and (codes[9:17, P:] == 1).all()


# ---------------------------------------------------------------------------
# oto_sources: SRT parsing, intervals, redactions, transcripts, parsers
# ---------------------------------------------------------------------------

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:03,000
[Normal Turn] hello there friend

2
00:00:03,500 --> 00:00:04,000
[Channel Bleed] yeah

3
00:00:05,000 --> 00:00:12,000
[Normal Turn] this is a much longer segment that keeps going

4
00:00:12,500 --> 00:00:13,000
[Acknowledgement Backchannel] mm-hmm
"""


def test_parse_srt():
    segs = oto_sources.parse_srt(SRT_SAMPLE)
    assert len(segs) == 4
    assert segs[0].start_s == 1.0 and segs[0].end_s == 3.0
    assert segs[0].label == "Normal Turn" and segs[0].text == "hello there friend"
    assert segs[1].label == "Channel Bleed"


def test_srt_words_short_speech_only():
    segs = oto_sources.parse_srt(SRT_SAMPLE)
    words = oto_sources.srt_words(segs, max_segment_s=4.0)
    # seg 1 (speech, 2 s) + seg 4 (backchannel, 0.5 s); bleed + long excluded
    assert [w for w, _, _ in words] == ["hello", "there", "friend", "mm-hmm"]
    assert words[0][1] == pytest.approx(1.0)      # even spread starts at seg start
    assert words[2][2] == pytest.approx(3.0)      # ... and ends at seg end


def test_long_srt_segments():
    segs = oto_sources.parse_srt(SRT_SAMPLE)
    assert oto_sources.long_srt_segments(segs, min_segment_s=4.0) == [(5.0, 12.0)]


def test_normalize_redactions():
    raw = [
        {"start_sec": 5.0, "end_sec": 2.0},        # reversed -> swapped
        {"start": 1.0, "end": 3.0},
        (3.5, 4.5),                                # inside the first two
        (7.0, 7.0),                                # degenerate -> dropped
        {"start_sec": None, "end_sec": 9.0},       # missing bound -> dropped
    ]
    assert oto_sources.normalize_redactions(raw) == [(1.0, 5.0)]
    assert oto_sources.normalize_redactions(None) == []


def test_merge_intervals_and_intervals_from_words():
    assert oto_sources.merge_intervals([(0, 1), (1.2, 2)], gap_s=0.3) == [(0.0, 2.0)]
    assert oto_sources.merge_intervals([(0, 1), (2, 3)], gap_s=0.3) == [(0.0, 1.0), (2.0, 3.0)]
    words = [("a", 0.0, 0.4), ("b", 0.6, 1.0), ("c", 5.0, 5.4)]
    assert oto_sources.intervals_from_words(words) == [(0.0, 1.0), (5.0, 5.4)]


def test_transcript_words_ms():
    tr = {"segments": [
        {"tokens": [
            {"written": " Hello", "starttime": 1000, "endtime": 1400},
            {"written": "", "starttime": 0, "endtime": 1},
            {"written": " world", "starttime": 1500, "endtime": 2000},
        ]},
        {"text": "fallback segment here", "starttime": 3000, "endtime": 4500},
    ]}
    words = oto_sources.transcript_words(tr)
    assert words[0] == ("Hello", 1.0, 1.4)         # ms -> s
    assert words[1] == ("world", 1.5, 2.0)
    # segment without tokens -> even spread of its text
    assert [w for w, _, _ in words[2:]] == ["fallback", "segment", "here"]
    assert words[2][1] == pytest.approx(3.0)
    assert words[-1][2] == pytest.approx(4.5)


def test_parse_task_session():
    spec = oto_sources.SOURCES["task-20h"]
    meta = {
        "duration": 61.5,
        "task": {"task_name": "picture-description_task"},
        "speakers": [
            {"speaker_id": "s1", "task_role_name": "describer", "gender": "f"},
            {"speaker_id": "s2", "task_role_name": "guesser", "gender": "m", "channel": 1},
        ],
        "redacted_segments": [{"start_sec": 10.0, "end_sec": 12.0}],
    }
    sm = oto_sources.parse_task_session(spec, "sess1", meta)
    assert sm.source == "task-20h" and sm.session_id == "sess1"
    assert sm.duration_s == 61.5
    assert sm.speakers[0]["role"] == "describer"
    assert "picture description task" in sm.system_prompts[0]
    assert "describer" in sm.system_prompts[0]
    assert sm.redactions == [(10.0, 12.0)]


def test_parse_turn_session():
    spec = oto_sources.SOURCES["turn-104h"]
    segs = oto_sources.parse_srt(SRT_SAMPLE)
    meta = {"conversation_type": "Casual", "speaker_1_actor_id": "a1",
            "speaker_2_actor_id": "a2"}
    sm = oto_sources.parse_turn_session(spec, "42", meta, segs, [])
    assert sm.speakers[0]["speaker_id"] == "a1"
    assert "casual conversation" in sm.system_prompts[0]
    # speech = speech-label intervals only (Channel Bleed excluded)
    assert sm.speech[0] == [(1.0, 3.0), (5.0, 12.0), (12.5, 13.0)]
    assert sm.speech[1] == []
    assert sm.duration_s == 13.0
    assert sm.srt_segments[0] == segs


# ---------------------------------------------------------------------------
# hf_tar: range reader + tar index
# ---------------------------------------------------------------------------

def _make_tar(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_tar_index_and_read_member(tmp_path):
    from ppq.data import hf_tar
    members = {
        "a/sess1.flac": b"FLAC" * 1000,
        "a/sess1.json": b"{}",
        "b/sess2.transcript_speaker_1.json": b'{"segments": []}',
    }
    blob = _make_tar(members)

    def fetch(off, n):
        return blob[off:off + n]

    cache = tmp_path / "idx.json"
    idx = hf_tar.tar_index(len(blob), fetch, cache)
    assert set(idx) == set(members)
    for name, data in members.items():
        assert hf_tar.read_member(fetch, idx, name) == data
    # the JSON cache is written and used on the second pass (fetch not called)
    assert cache.exists()

    def boom(off, n):
        raise AssertionError("fetch called despite the index cache")

    assert hf_tar.tar_index(len(blob), boom, cache) == idx


def test_http_range_file_seek_read():
    from ppq.data import hf_tar
    blob = bytes(range(256)) * 100  # 25600 bytes, one 1 MiB chunk
    fetched = []

    def fetch(off, n):
        fetched.append((off, n))
        return blob[off:off + n]

    f = hf_tar.HTTPRangeFile(len(blob), fetch)
    assert f.read(10) == blob[:10]
    f.seek(5000)
    assert f.read(100) == blob[5000:5100]
    f.seek(-50, 2)  # SEEK_END
    assert f.read(100) == blob[-50:]
    assert f.read(1) == b""
    # everything lives in one 1 MiB chunk -> exactly one fetch (LRU hits)
    assert len(fetched) == 1


# ---------------------------------------------------------------------------
# realconv: chunking, word filtering, voice prompt, separation
# ---------------------------------------------------------------------------

def test_mutual_silence_cuts():
    cuts = realconv.mutual_silence_cuts([(1.0, 20.0)], [(25.0, 48.0)], 50.0,
                                        min_silence_s=0.8)
    assert cuts == [pytest.approx(0.5), pytest.approx(22.5), pytest.approx(49.0)]


def test_chunk_session():
    # short session -> a single chunk
    chunks = realconv.chunk_session([(1.0, 20.0)], [(25.0, 48.0)], 50.0)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0.0, 50.0)]
    # no mutual silence inside the window -> hard cuts at max_chunk_s
    chunks = realconv.chunk_session([(5.0, 200.0)], [], 200.0,
                                    min_chunk_s=20.0, max_chunk_s=80.0)
    assert [(c.start_s, c.end_s) for c in chunks] == [
        (0.0, 80.0), (80.0, 160.0), (160.0, 200.0)]
    # prefers the latest mutual silence before the max length
    chunks = realconv.chunk_session([(1.0, 20.0)], [(25.0, 48.0)], 100.0,
                                    min_chunk_s=20.0, max_chunk_s=80.0)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0.0, 74.0), (74.0, 100.0)]
    # a tail shorter than min_tail_s is dropped
    chunks = realconv.chunk_session([(1.0, 25.0), (35.0, 100.0)], [], 112.0,
                                    min_chunk_s=20.0, max_chunk_s=80.0)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0.0, 30.0), (30.0, 106.0)]


def test_filter_words_for_chunk():
    words = [("a", 1.0, 1.4), ("b", 5.0, 5.4), ("c", 9.0, 9.4), ("d", 11.0, 11.4)]
    out = realconv.filter_words_for_chunk(words, 0.0, 10.0, [(5.0, 5.5)])
    assert [w for w, _, _ in out] == ["a", "c"]  # b redacted, d outside
    out2 = realconv.filter_words_for_chunk(words, 8.0, 12.0, [])
    assert [w for w, _, _ in out2] == ["c", "d"]
    assert out2[0][1] == pytest.approx(1.0)      # shifted to chunk-relative time


def test_clip_voice_prompt_prefers_outside_chunk():
    sr = SAMPLE_RATE
    wav = np.arange(sr * 40, dtype=np.float32)  # position-encoded 40 s
    speech = [(2.0, 12.0), (20.0, 38.0)]
    clip = realconv.clip_voice_prompt(wav, speech, 15.0, 39.0, 8.0, sr)
    assert clip is not None and len(clip) == 8 * sr
    center = 0.5 * (clip[0] + clip[-1]) / sr
    assert 2.0 <= center <= 12.0                 # taken from outside the chunk
    # falls back to inside-chunk speech when nothing outside fits
    clip2 = realconv.clip_voice_prompt(wav, [(20.0, 38.0)], 0.0, 40.0, 8.0, sr)
    assert clip2 is not None and len(clip2) == 8 * sr
    # nothing long enough -> None
    assert realconv.clip_voice_prompt(wav, [(1.0, 3.0)], 0.0, 40.0, 8.0, sr) is None


def test_separation_db():
    sr = SAMPLE_RATE
    n = sr * 30
    rng = np.random.default_rng(0)
    wav0 = np.zeros(n, dtype=np.float32)
    wav1 = np.zeros(n, dtype=np.float32)
    wav0[: 10 * sr] = 0.1 * rng.standard_normal(10 * sr)
    wav1[20 * sr:] = 0.1 * rng.standard_normal(10 * sr)
    s0, s1 = [(0.0, 10.0)], [(20.0, 30.0)]
    assert realconv.separation_db(wav0, wav1, s0, s1, sr) > 30.0
    # heavy bleed: the same signal on both channels -> ~0 dB
    assert realconv.separation_db(wav0, wav0.copy(), s0, s1, sr) < 6.0
    # too little data -> inf (caller skips the separation check)
    assert realconv.separation_db(wav0[:sr], wav1[:sr], s0, s1, sr) == float("inf")


def _prepared_session(dur_s=50.0, source="task-20h", sid="sess/1"):
    sr = SAMPLE_RATE
    n = int(dur_s * sr)
    rng = np.random.default_rng(1)
    stereo = np.zeros((2, n), dtype=np.float32)
    stereo[0, int(1 * sr): int(20 * sr)] = 0.05 * rng.standard_normal(19 * sr)
    stereo[1, int(25 * sr): int(48 * sr)] = 0.05 * rng.standard_normal(23 * sr)
    speech = [[(1.0, 20.0)], [(25.0, 48.0)]]
    words = [[("hello", 2.0, 2.4), ("there", 3.0, 3.4), ("redacted", 6.0, 6.4)],
             [("answer", 30.0, 30.4), ("here", 31.0, 31.4)]]
    meta = oto_sources.SessionMeta(
        source=source, session_id=sid, duration_s=dur_s,
        speakers=[{"speaker_id": "s1"}, {"speaker_id": "s2"}],
        system_prompts=["prompt zero", "prompt one"],
        redactions=[(5.5, 7.0)])
    return realconv.PreparedSession(meta=meta, stereo=stereo, speech=speech, words=words)


def test_build_chunk_samples_role_swap():
    cfg = _cfg()
    out = list(realconv.build_chunk_samples(
        _prepared_session(), FakeMimi(), FakeTokenizer(), cfg,
        voice_prompt_mode="none"))
    assert [name for name, *_ in out] == [
        "real-task-20h-sess_1-c0000-r0", "real-task-20h-sess_1-c0000-r1"]
    for name, codes, mask, info in out:
        assert codes.dtype == np.int32 and mask.dtype == np.float16
        assert codes.shape == mask.shape and codes.shape[0] == 17
        assert codes.shape[1] <= 1250
    # r0: agent = ch0; the redacted word is excluded and its span zeroed
    name, codes, mask, info = out[0]
    assert info["agent_channel"] == 0 and info["n_agent_words"] == 2
    a = cfg.audio_silence_frame_cnt
    P = a + 4 + a  # silence + prompt ids + silence (no voice prompt)
    f0, f1 = int(5.5 * FRAME_RATE_HZ), int(np.ceil(7.0 * FRAME_RATE_HZ))
    assert (mask[:, P + f0: P + f1] == 0).all()
    assert (mask[1:9, P:P + f0] == 1).all()
    # r1: agent = ch1 -> different codes and prompt
    assert out[1][3]["agent_channel"] == 1
    assert not np.array_equal(out[0][1], out[1][1])


def test_build_chunk_samples_min_agent_speech_skips_role():
    out = list(realconv.build_chunk_samples(
        _prepared_session(), FakeMimi(), FakeTokenizer(), _cfg(),
        voice_prompt_mode="none", min_agent_speech=0.4))
    # ch0 speaks 19/50 = 0.38 < 0.4, ch1 speaks 23/50 = 0.46 -> only r1 survives
    assert [name for name, *_ in out] == ["real-task-20h-sess_1-c0000-r1"]


def test_build_chunk_samples_same_speaker_voice_prompt():
    cfg = _cfg()
    prep = _prepared_session()
    none = list(realconv.build_chunk_samples(prep, FakeMimi(), FakeTokenizer(), cfg,
                                             voice_prompt_mode="none"))
    vp = list(realconv.build_chunk_samples(prep, FakeMimi(), FakeTokenizer(), cfg,
                                           voice_prompt_mode="same_speaker",
                                           voice_prompt_seconds=8.0))
    assert len(vp) == len(none) == 2
    # the voice-prompt prefix adds frames (8 s clip = 100 frames)
    assert vp[0][1].shape[1] == none[0][1].shape[1] + 100


# ---------------------------------------------------------------------------
# dataset source weights + validation helpers
# ---------------------------------------------------------------------------

def _write_sample(path: Path, T: int = 100, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 2048, (17, T)).astype(np.int32)
    codes[0] = rng.integers(0, 151936, T).astype(np.int32)
    mask = np.ones((17, T), dtype=np.float16)
    np.savez_compressed(path, codes=codes, loss_mask=mask)


def test_dataset_source_weights(tmp_path):
    from ppq.data.dataset import TokenShardDataset
    cfg = _cfg()
    entries = []
    for i, src in enumerate(["a", "a", "b"]):
        p = tmp_path / f"s{i}.npz"
        _write_sample(p, seed=i)
        entries.append({"path": str(p), "frames": 100, "source": src})
    man = tmp_path / "m.jsonl"
    man.write_text("".join(json.dumps(e) + "\n" for e in entries))

    # weight 0 disables a source entirely
    ds = TokenShardDataset([man], cfg, source_weights={"a": 1.0, "b": 0.0})
    picked = {ds._pick_entry(i)["path"] for i in range(50)}
    assert picked == {entries[0]["path"], entries[1]["path"]}

    # sources not listed default to weight 1.0
    ds2 = TokenShardDataset([man], cfg, source_weights={"b": 5.0})
    picked2 = {ds2._pick_entry(i)["path"] for i in range(200)}
    assert picked2 == {e["path"] for e in entries}

    # all-zero weights are rejected
    with pytest.raises(ValueError):
        TokenShardDataset([man], cfg, source_weights={"a": 0.0, "b": 0.0})


def test_check_sample_and_manifest(tmp_path):
    from ppq.data import validate
    cfg = _cfg()
    good = tmp_path / "good.npz"
    _write_sample(good)
    assert validate.check_sample(good, cfg) == []

    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, codes=np.zeros((16, 10), dtype=np.int32),
                        loss_mask=np.ones((16, 10), dtype=np.float16))
    assert validate.check_sample(bad, cfg)  # wrong stream count

    man = tmp_path / "m.jsonl"
    man.write_text(json.dumps({"path": str(good), "frames": 100,
                               "source": "real-task-20h", "name": "x"}) + "\n")
    rep = validate.check_manifest(man, cfg)
    assert rep["entries"] == 1 and rep["errors"] == [] and rep["duplicate_names"] == 0


def test_loader_smoke(tmp_path):
    from ppq.config import TrainConfig
    from ppq.data import validate
    cfg = _cfg()
    lines = []
    for i in range(4):
        p = tmp_path / f"s{i}.npz"
        _write_sample(p, T=1250, seed=i)
        lines.append(json.dumps({"path": str(p), "frames": 1250,
                                 "source": "real-task-20h"}))
    man = tmp_path / "m.jsonl"
    man.write_text("".join(line + "\n" for line in lines))
    tcfg = TrainConfig(model=cfg, segment_frames=1250)
    losses = validate.loader_smoke([man], tcfg, n_batches=2, batch_size=2)
    assert np.isfinite(losses["total"]) and losses["total"] > 0


# ---------------------------------------------------------------------------
# worker --stage real: resume behaviour with faked HF/Mimi/ASR boundaries
# ---------------------------------------------------------------------------

def _real_args(tmp_path: Path, out_name: str, **kw):
    a = types.SimpleNamespace(
        stage="real", source="task-20h", shard=0, num_shards=1,
        scripts="", out=str(tmp_path / out_name),
        mimi_weight="fake", tokenizer="fake", model_config="",
        voice_prompt="", max_hours=1.0,
        log_dir=str(tmp_path / f"logs-{out_name}"),
        seed=0, max_attempts=3, retry_failed=False, max_items=0,
        data_root=str(tmp_path / f"root-{out_name}"), hf_token_env="HF_TOKEN",
        device="cpu", asr_model="fake",
        min_chunk_s=20.0, max_chunk_s=80.0, min_silence_s=0.8,
        min_agent_speech=0.15, min_separation_db=12.0,
        voice_prompt_seconds=8.0, voice_prompt_mode="",
        segment_frames=1250, long_segment_s=4.0, keep_raw=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _flac_bytes(dur_s: float = 30.0) -> bytes:
    import soundfile as sf
    sr = SAMPLE_RATE
    n = int(dur_s * sr)
    rng = np.random.default_rng(7)
    data = np.zeros((n, 2), dtype=np.float32)
    data[: 14 * sr, 0] = 0.05 * rng.standard_normal(14 * sr)
    data[16 * sr:, 1] = 0.05 * rng.standard_normal(14 * sr)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="FLAC")
    return buf.getvalue()


@pytest.fixture
def real_fakes(monkeypatch):
    flac = _flac_bytes()

    def meta_for(sid):
        return oto_sources.SessionMeta(
            source="task-20h", session_id=sid, duration_s=30.0,
            speakers=[{"speaker_id": "s1"}, {"speaker_id": "s2"}],
            system_prompts=["prompt zero", "prompt one"], redactions=[])

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def session_map(self):
            return {"s1": {}, "s2": {}}

    monkeypatch.setattr(oto_sources, "list_sessions",
                        lambda spec, root, token: ["s1", "s2"])
    monkeypatch.setattr(oto_sources, "TarStore", FakeStore)
    monkeypatch.setattr(oto_sources, "get_hf_token", lambda env: None)
    monkeypatch.setattr(oto_sources, "load_tar_session",
                        lambda spec, store, entry, sid: (meta_for(sid), flac))
    monkeypatch.setattr(ppq.checkpoint, "get_mimi", lambda *a, **k: FakeMimi())
    monkeypatch.setattr(ppq.text_tokenizer, "QwenTextTokenizer", FakeTokenizer)
    speech = [[(0.5, 14.0)], [(16.0, 29.5)]]
    words = [[("hello", 1.0, 1.4), ("there", 2.0, 2.4)],
             [("answer", 20.0, 20.4), ("here", 21.0, 21.4)]]
    monkeypatch.setattr(worker, "_session_speech_words",
                        lambda *a, **k: (speech, words))


def _real_manifest(out: Path) -> list:
    return read_jsonl(out / "manifest-real-task-20h-0000.jsonl")


def test_real_worker_resume_is_identical(tmp_path, real_fakes):
    out_a = tmp_path / "out_a"
    worker.run_real(_real_args(tmp_path, "out_a", max_items=1))  # "crash" after 1
    assert len(_real_manifest(out_a)) == 2       # one session x two roles
    worker.run_real(_real_args(tmp_path, "out_a"))               # resume
    man_a = _real_manifest(out_a)
    assert len(man_a) == 4

    out_b = tmp_path / "out_b"
    worker.run_real(_real_args(tmp_path, "out_b"))               # fresh reference
    man_b = _real_manifest(out_b)
    assert [e["name"] for e in man_a] == [e["name"] for e in man_b]
    for ea, eb in zip(man_a, man_b):
        da, db = np.load(ea["path"]), np.load(eb["path"])
        assert np.array_equal(da["codes"], db["codes"]), ea["name"]
        assert np.array_equal(da["loss_mask"], db["loss_mask"]), ea["name"]

    # third run: fully idempotent (session markers -> no new manifest lines)
    worker.run_real(_real_args(tmp_path, "out_a"))
    assert len(_real_manifest(out_a)) == 4

    prog = json.loads((tmp_path / "logs-out_a"
                       / "progress-real-task-20h-0000.json").read_text(encoding="utf-8"))
    assert prog["status"] == "finished" and prog["done"] == 2 and prog["total"] == 2
