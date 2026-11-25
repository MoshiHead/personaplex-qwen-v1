"""Shared progress tracking, logging and file utilities for the PPQ
data and training pipelines.

Conventions:
  - Progress JSON files:  <log_dir>/progress-<stage>[-<shard>].json
  - Worker log files:     <log_dir>/worker-<stage>-<shard>.log
  - Training log file:    <stage_dir>/logs/train.log
  - Failed items:         <log_dir>/failed-<stage>-<shard>.jsonl
All small files are written atomically (temp file + rename) so a crash can
never leave a torn file behind.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def fmt_duration(seconds: float | None) -> str:
    """Human-friendly duration, e.g. '1h53m04s'. '?' when unknown."""
    if seconds is None or seconds != seconds or seconds < 0:  # None / NaN / negative
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, indent=1))


def repair_jsonl(path: str | Path) -> int:
    """Drop trailing lines that are not valid JSON (torn by a crash).
    The original file is backed up next to it. Returns dropped line count."""
    path = Path(path)
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    valid = len(lines)
    while valid > 0:
        try:
            json.loads(lines[valid - 1])
            break
        except json.JSONDecodeError:
            valid -= 1
    dropped = len(lines) - valid
    if dropped:
        backup = path.with_name(path.name + f".broken-{int(time.time())}")
        os.replace(path, backup)
        path.write_text("".join(l + "\n" for l in lines[:valid]), encoding="utf-8")
        logger.warning("repair_jsonl: dropped %d torn line(s) from %s (backup: %s)",
                       dropped, path.name, backup.name)
    return dropped


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pid_alive(pid: int) -> bool:
    """Cross-platform check whether a process exists (safe on Windows, where
    os.kill(pid, 0) would actually terminate the process)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PidLock:
    """Best-effort single-instance lock via a PID file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def acquire(self) -> bool:
        if self.path.exists():
            try:
                pid = int(self.path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pid = -1
            if pid > 0 and pid != os.getpid() and pid_alive(pid):
                return False
        atomic_write_text(self.path, str(os.getpid()))
        return True

    def release(self) -> None:
        try:
            if self.path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def owner_alive(self) -> bool:
        try:
            return pid_alive(int(self.path.read_text(encoding="utf-8").strip()))
        except (ValueError, OSError):
            return False


def session_header(log_path: str | Path, title: str) -> None:
    """Append a visible session separator to a log file (append mode)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {title} @ {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} =====\n")


def setup_file_logger(name: str, log_path: str | Path,
                      level: int = logging.INFO,
                      reset: bool = False) -> logging.Logger:
    """Logger writing to a file (append) AND the console. With reset=True,
    any previously attached FileHandlers (e.g. from an earlier call with a
    different path in the same process) are removed first."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False
    if reset:
        for h in [h for h in log.handlers if isinstance(h, logging.FileHandler)]:
            log.removeHandler(h)
            h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
               for h in log.handlers):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


class ProgressTracker:
    """Tracks done/failed/total for one job and persists a progress JSON.

    `done` counts successfully finished items (including items finished in
    previous sessions), `failed` counts permanently failed items. Completion
    is (done + failed) / total.
    """

    def __init__(self, path: str | Path, stage: str, total: int,
                 prior_done: int = 0, prior_failed: int = 0,
                 unit: str = "items"):
        self.path = Path(path)
        self.stage = stage
        self.total = int(total)
        self.done = int(prior_done)
        self.failed = int(prior_failed)
        self.session_done = 0
        self.unit = unit
        self.audio_hours = 0.0
        self.status = "running"
        self.started_at = time.time()
        self._times: collections.deque = collections.deque(maxlen=200)

    def add(self, n: int = 1, audio_seconds: float = 0.0, failed: bool = False) -> None:
        now = time.time()
        if failed:
            self.failed += n
        else:
            self.done += n
            self.session_done += n
        for _ in range(n):
            self._times.append(now)
        self.audio_hours += audio_seconds / 3600.0

    @property
    def processed(self) -> int:
        return self.done + self.failed

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.processed)

    @property
    def percent(self) -> float:
        return 100.0 * self.processed / max(1, self.total)

    def rate_per_s(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    def eta_seconds(self) -> float | None:
        r = self.rate_per_s()
        if r <= 0:
            return None
        return self.remaining / r

    def snapshot(self, status: str | None = None) -> dict:
        if status is not None:
            self.status = status
        return {
            "stage": self.stage,
            "status": self.status,
            "done": self.done,
            "failed": self.failed,
            "total": self.total,
            "remaining": self.remaining,
            "percent": round(self.percent, 2),
            "session_done": self.session_done,
            "rate_per_s": round(self.rate_per_s(), 4),
            "eta_s": (None if self.eta_seconds() is None else round(self.eta_seconds())),
            "eta": fmt_duration(self.eta_seconds()),
            "audio_hours": round(self.audio_hours, 2),
            "unit": self.unit,
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_ts": time.time(),
        }

    def save(self, status: str | None = None) -> None:
        atomic_write_json(self.path, self.snapshot(status))

    def log_line(self, log: logging.Logger, every: int = 1) -> None:
        if self.session_done % every != 0:
            return
        log.info(
            "[%s] done %d/%d (%.1f%%) | this session %d | failed %d | remaining %d "
            "| %.3f %s/s | ETA %s | audio %.1f h",
            self.stage, self.done, self.total, self.percent, self.session_done,
            self.failed, self.remaining, self.rate_per_s(), self.unit,
            fmt_duration(self.eta_seconds()), self.audio_hours)


# ---------------------------------------------------------------------------
# Notebook-facing helpers: read all progress files and render status tables.
# ---------------------------------------------------------------------------

STALL_SECONDS = 600  # no progress update for 10 min -> "STALLED?"


def load_progress_files(log_dir: str | Path, stages: tuple[str, ...] | None = None) -> list[dict]:
    out = []
    for p in sorted(Path(log_dir).glob("progress-*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if stages and not any(d.get("stage", "").startswith(s) for s in stages):
            continue
        d["_file"] = p.name
        out.append(d)
    return out


def render_data_progress(log_dir: str | Path, stages: tuple[str, ...] | None = None) -> str:
    rows = load_progress_files(log_dir, stages)
    if not rows:
        return f"no progress files yet in {log_dir}"
    lines = [f"{'file':<34} {'status':<9} {'done':>8} {'total':>8} {'%':>6} "
             f"{'failed':>6} {'rate/s':>7} {'ETA':>10} {'audio_h':>8}  note"]
    tot_done = tot_total = tot_failed = 0
    etas = []
    for d in rows:
        note = ""
        if d.get("status") == "running" and time.time() - d.get("updated_ts", 0) > STALL_SECONDS:
            note = "STALLED?"
        if d.get("status") == "failed":
            note = (note + " FAILED").strip()
        lines.append(
            f"{d['_file']:<34} {d.get('status', '?'):<9} {d.get('done', 0):>8} "
            f"{d.get('total', 0):>8} {d.get('percent', 0):>5.1f}% {d.get('failed', 0):>6} "
            f"{d.get('rate_per_s', 0):>7.3f} {d.get('eta', '?'):>10} {d.get('audio_hours', 0):>8.1f}  {note}")
        tot_done += d.get("done", 0)
        tot_total += d.get("total", 0)
        tot_failed += d.get("failed", 0)
        if d.get("eta_s"):
            etas.append(d["eta_s"])
    pct = 100.0 * tot_done / max(1, tot_total)
    lines.append("-" * 110)
    lines.append(f"{'TOTAL':<34} {'':<9} {tot_done:>8} {tot_total:>8} {pct:>5.1f}% "
                 f"{tot_failed:>6} {'':>7} {fmt_duration(max(etas) if etas else None):>10}")
    return "\n".join(lines)


def pipeline_all_done(log_dir: str | Path, stages: tuple[str, ...]) -> bool:
    rows = load_progress_files(log_dir, stages)
    seen = {s: False for s in stages}
    for d in rows:
        for s in stages:
            if d.get("stage", "").startswith(s):
                seen[s] = True
        if d.get("status") not in ("finished", "failed"):
            return False
    return bool(rows) and all(seen.values())


def watch(log_dir: str | Path, stages: tuple[str, ...], interval_s: float = 60.0) -> None:
    """Notebook helper: refresh the progress table until all stages finish.
    Safe to interrupt with the stop button; re-run any time."""
    from IPython.display import clear_output
    while True:
        clear_output(wait=True)
        print(render_data_progress(log_dir, stages))
        print(f"\n(updated {time.strftime('%H:%M:%S')}; interrupt the cell to stop watching)")
        if pipeline_all_done(log_dir, stages):
            print("\nALL DONE")
            return
        time.sleep(interval_s)


def render_train_progress(stage_dir: str | Path) -> str:
    p = Path(stage_dir) / "progress.json"
    if not p.exists():
        return f"no progress yet in {stage_dir}"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"progress file unreadable in {stage_dir}"
    return (
        f"stage {d.get('stage')} [{d.get('status')}]  step {d.get('step')}/{d.get('max_steps')} "
        f"({d.get('percent', 0):.1f}%)  remaining {d.get('remaining_steps')}  "
        f"{d.get('sec_per_step', 0):.2f}s/step  ETA {d.get('eta', '?')}  "
        f"loss {d.get('total_loss', float('nan')):.3f}  lr {d.get('lr', 0):.2e}  "
        f"audio seen {d.get('audio_hours_seen', 0):.1f} h  updated {d.get('updated_at')}")


def watch_train(stage_dir: str | Path, interval_s: float = 60.0) -> None:
    """Notebook helper: refresh training progress until finished/failed."""
    from IPython.display import clear_output
    while True:
        clear_output(wait=True)
        print(render_train_progress(stage_dir))
        print(f"\n(updated {time.strftime('%H:%M:%S')}; interrupt the cell to stop watching)")
        p = Path(stage_dir) / "progress.json"
        if p.exists():
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("status") in ("finished", "failed"):
                    print("\nSTAGE DONE")
                    return
            except json.JSONDecodeError:
                pass
        time.sleep(interval_s)


def training_status(run_dir: str | Path) -> str:
    """Overview of init + stage_a + stage_b for the training notebook."""
    run_dir = Path(run_dir)
    lines = []
    init_ok = (run_dir / "init" / "COMPLETE").exists()
    lines.append(f"init checkpoint : {'DONE' if init_ok else 'missing'}")
    for stage in ("stage_a", "stage_b"):
        sdir = run_dir / stage
        final_ok = (sdir / "checkpoints" / "final" / "COMPLETE").exists()
        prog = sdir / "progress.json"
        if final_ok:
            lines.append(f"{stage:<16}: DONE (final checkpoint complete)")
        elif prog.exists():
            try:
                d = json.loads(prog.read_text(encoding="utf-8"))
                lines.append(f"{stage:<16}: {d.get('status')} at step "
                             f"{d.get('step')}/{d.get('max_steps')} ({d.get('percent', 0):.1f}%)")
            except json.JSONDecodeError:
                lines.append(f"{stage:<16}: in progress (progress file unreadable)")
        else:
            lines.append(f"{stage:<16}: not started")
    return "\n".join(lines)


