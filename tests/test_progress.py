"""Unit tests for ppq.progress: tracker math, atomic writes, JSONL repair,
PID locks, and the notebook status renderers."""
import json
import os
import time

from ppq.progress import (
    PidLock, ProgressTracker, atomic_write_json, fmt_duration,
    load_progress_files, pid_alive, pipeline_all_done, read_jsonl,
    render_data_progress, repair_jsonl, training_status,
)


def test_fmt_duration():
    assert fmt_duration(None) == "?"
    assert fmt_duration(float("nan")) == "?"
    assert fmt_duration(0) == "0s"
    assert fmt_duration(65) == "1m05s"
    assert fmt_duration(3661) == "1h01m01s"


def test_tracker_math_and_persistence(tmp_path):
    tr = ProgressTracker(tmp_path / "p.json", stage="synth-0000", total=10,
                         prior_done=4)
    assert tr.remaining == 6 and abs(tr.percent - 40.0) < 1e-9
    tr.add(audio_seconds=36.0)
    tr.add(failed=True)
    assert tr.done == 5 and tr.failed == 1 and tr.session_done == 1
    assert tr.remaining == 4 and abs(tr.percent - 60.0) < 1e-9
    assert abs(tr.audio_hours - 0.01) < 1e-9
    assert tr.rate_per_s() > 0  # two add() calls -> measurable rate
    tr.save()
    d = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert d["done"] == 5 and d["total"] == 10 and d["failed"] == 1
    assert d["remaining"] == 4 and d["status"] == "running"
    assert "eta" in d and "updated_at" in d


def test_atomic_write_json_leaves_valid_file(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1})
    (tmp_path / "x.json.tmp").write_text("{torn")  # simulated torn write
    atomic_write_json(p, {"a": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2}


def test_repair_jsonl(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n{"a": 4', encoding="utf-8")
    assert repair_jsonl(p) == 1
    assert len(read_jsonl(p)) == 3
    assert list(tmp_path.glob("m.jsonl.broken-*"))  # backup kept
    assert repair_jsonl(p) == 0  # clean now
    assert repair_jsonl(tmp_path / "missing.jsonl") == 0


def test_pid_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(-1)
    assert not pid_alive(0)


def test_pid_lock(tmp_path):
    path = tmp_path / "w.pid"
    lock = PidLock(path)
    assert lock.acquire()
    assert lock.owner_alive()
    # stale lock from a dead pid can be taken over
    path.write_text("99999999", encoding="utf-8")
    lock2 = PidLock(path)
    assert lock2.acquire()
    lock2.release()
    assert not path.exists()


def test_render_and_all_done(tmp_path):
    tr = ProgressTracker(tmp_path / "progress-synth-0000.json",
                         stage="synth-0000", total=2, prior_done=1)
    tr.add()
    tr.save("finished")
    table = render_data_progress(tmp_path, ("synth",))
    assert "synth-0000" in table and "TOTAL" in table
    assert pipeline_all_done(tmp_path, ("synth",))
    assert not pipeline_all_done(tmp_path, ("synth", "libritts"))
    rows = load_progress_files(tmp_path, ("libritts",))
    assert rows == []


def test_training_status(tmp_path):
    run = tmp_path / "run"
    (run / "init").mkdir(parents=True)
    assert "missing" in training_status(run)
    (run / "init" / "COMPLETE").write_text("ok")
    out = training_status(run)
    assert "init checkpoint : DONE" in out
    assert "stage_a         : not started" in out
