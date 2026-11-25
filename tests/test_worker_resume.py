"""Resume behaviour of the data worker (synth stage) with fake TTS/Mimi:
interrupt mid-run -> resume produces identical output; torn manifests and
half-written npz files are redone; permanent failures are tracked and skipped.
"""
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import ppq.checkpoint
import ppq.data.tts_kokoro
import ppq.text_tokenizer
from ppq.data import worker
from ppq.data.dialogue import Dialogue, Turn
from ppq.data.tts_kokoro import SynthResult
from ppq.progress import read_jsonl


class FakeMimi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def encode(self, tensor):
        t = max(1, int(np.ceil(tensor.shape[-1] / 1920)))
        return torch.zeros(1, 8, t, dtype=torch.long)


class FakeTokenizer:
    def __init__(self, *a, **k):
        pass

    def encode(self, text):
        return [1, 2, 3]

    def encode_system_prompt(self, text):
        return [7, 7, 7, 7]


class FakeTTS:
    def synth(self, text, voice):
        if "BOOM" in text:
            raise RuntimeError("tts boom")
        words = text.split()
        per = 0.12
        dur = per * len(words)
        n = max(1, int(dur * 24000))
        seed = abs(hash((text, voice))) % (2**31)
        wav = np.random.RandomState(seed).uniform(-0.1, 0.1, n).astype(np.float32)
        return SynthResult(wav=wav,
                           words=[(w, i * per, (i + 1) * per) for i, w in enumerate(words)])


def _make_dialogue(i: int, boom: bool = False) -> Dialogue:
    user_first = "BOOM hello there" if boom else f"user question number {i}"
    return Dialogue(
        system_prompt=f"You are a helpful test assistant number {i}.",
        turns=[
            Turn("user", user_first),
            Turn("agent", f"answer to question {i} with a few words"),
            Turn("user", f"follow up question {i} please"),
            Turn("agent", f"final answer {i} with more words here"),
        ],
    )


def _write_scripts(path: Path, n: int, boom_idx: int = -1) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(_make_dialogue(i, boom=(i == boom_idx)).to_json() + "\n")


def _args(tmp_path: Path, out_name: str, **kw) -> types.SimpleNamespace:
    a = types.SimpleNamespace(
        stage="synth", shard=0, num_shards=2,
        scripts=str(tmp_path / "scripts.jsonl"),
        out=str(tmp_path / out_name),
        mimi_weight="fake", tokenizer="fake", model_config="",
        voice_prompt="", log_dir=str(tmp_path / f"logs-{out_name}"),
        seed=0, max_attempts=3, retry_failed=False, max_items=0,
        max_hours=1.0,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(ppq.text_tokenizer, "QwenTextTokenizer", FakeTokenizer)
    monkeypatch.setattr(ppq.checkpoint, "get_mimi", lambda *a, **k: FakeMimi())
    monkeypatch.setattr(ppq.data.tts_kokoro, "KokoroTTS", FakeTTS)


def _manifest(out: Path) -> list[dict]:
    return read_jsonl(out / "manifest-0000.jsonl")


def test_resume_after_interrupt_is_identical(tmp_path, fakes):
    _write_scripts(tmp_path / "scripts.jsonl", 6)
    # shard 0 of 2 owns global indices 0, 2, 4 -> 3 dialogues

    out_a = tmp_path / "out_a"
    worker.run_synth(_args(tmp_path, "out_a", max_items=1))  # "crash" after 1
    assert len(_manifest(out_a)) == 1
    worker.run_synth(_args(tmp_path, "out_a"))               # resume to completion
    man_a = _manifest(out_a)
    assert len(man_a) == 3

    out_b = tmp_path / "out_b"
    worker.run_synth(_args(tmp_path, "out_b"))               # fresh reference run
    man_b = _manifest(out_b)

    assert [e["name"] for e in man_a] == [e["name"] for e in man_b]
    for ea, eb in zip(man_a, man_b):
        da, db = np.load(ea["path"]), np.load(eb["path"])
        assert np.array_equal(da["codes"], db["codes"]), ea["name"]
        assert np.array_equal(da["loss_mask"], db["loss_mask"]), ea["name"]

    prog = json.loads((tmp_path / "logs-out_a" / "progress-synth-0000.json")
                      .read_text(encoding="utf-8"))
    assert prog["status"] == "finished" and prog["done"] == 3 and prog["total"] == 3


def test_torn_manifest_and_npz_are_redone(tmp_path, fakes):
    _write_scripts(tmp_path / "scripts.jsonl", 6)
    out = tmp_path / "out"
    worker.run_synth(_args(tmp_path, "out"))
    man = _manifest(out)
    assert len(man) == 3

    # simulate crash: last manifest line torn + the npz half-written
    mpath = out / "manifest-0000.jsonl"
    lines = mpath.read_text(encoding="utf-8").splitlines()
    mpath.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20], encoding="utf-8")
    last = man[-1]
    with open(last["path"], "wb") as f:
        f.write(b"garbage")

    worker.run_synth(_args(tmp_path, "out"))  # must redo the torn item
    man2 = _manifest(out)
    assert len(man2) == 3
    redone = [e for e in man2 if e["name"] == last["name"]][0]
    d = np.load(redone["path"])
    assert d["codes"].shape[0] == 17


def test_failure_tracking_and_retry(tmp_path, fakes):
    _write_scripts(tmp_path / "scripts.jsonl", 6, boom_idx=2)  # gi 2 is shard 0's
    out = tmp_path / "out"
    log_dir = tmp_path / "logs-out"

    worker.run_synth(_args(tmp_path, "out", max_attempts=1))
    assert len(_manifest(out)) == 2  # conv-0000002 failed permanently
    failed = read_jsonl(log_dir / "failed-synth-0000.jsonl")
    assert len(failed) == 1 and failed[0]["name"] == "conv-0000002"

    worker.run_synth(_args(tmp_path, "out", max_attempts=1))  # no retry
    assert len(_manifest(out)) == 2
    assert len(read_jsonl(log_dir / "failed-synth-0000.jsonl")) == 1

    worker.run_synth(_args(tmp_path, "out", max_attempts=2, retry_failed=True))
    assert len(_manifest(out)) == 2  # still failing, but attempted again
    assert len(read_jsonl(log_dir / "failed-synth-0000.jsonl")) == 2
    prog = json.loads((log_dir / "progress-synth-0000.json").read_text(encoding="utf-8"))
    assert prog["status"] == "finished" and prog["failed"] == 1 and prog["done"] == 2
    assert prog["remaining"] == 0


def test_second_worker_instance_exits(tmp_path, fakes):
    import os
    _write_scripts(tmp_path / "scripts.jsonl", 6)
    log_dir = tmp_path / "logs-out"
    log_dir.mkdir(parents=True)
    # fake a live lock owner: the parent process is alive and not us
    (log_dir / "worker-synth-0000.pid").write_text(str(os.getppid()), encoding="utf-8")
    worker.run_synth(_args(tmp_path, "out"))  # should print and return immediately
    assert not (tmp_path / "out" / "manifest-0000.jsonl").exists()
