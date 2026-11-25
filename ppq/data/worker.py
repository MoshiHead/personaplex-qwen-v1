"""Sharded data-pipeline worker. The data notebook launches one process per
GPU, each handling a disjoint subset of the work.

Stages:
  synth    : scripts JSONL -> Kokoro TTS -> stereo stitch -> mimi tokenize -> npz shards
  libritts : LibriTTS-R utterances -> monologue npz shards (Stage A data)

Resume contract:
  - Items are named by their GLOBAL index (conv-0000123 / libritts-0000045),
    so names stay stable if scripts.jsonl grows or --num-shards changes.
  - The manifest is the source of truth for "done". An .npz without a
    manifest entry (crash between write and manifest append) is redone.
  - .npz files are written to a temp name and atomically renamed.
  - Failures are recorded in failed-<stage>-<shard>.jsonl with an attempt
    count; an item is skipped for good after --max-attempts (reset with
    --retry-failed).
  - Every shard writes <log_dir>/worker-<stage>-<shard>.log (append) and
    <log_dir>/progress-<stage>-<shard>.json (done/total/%/ETA).

Example:
  CUDA_VISIBLE_DEVICES=0 python -m ppq.data.worker --stage synth \
      --shard 0 --num-shards 8 --scripts /workspace/ppq_data/scripts.jsonl \
      --out /workspace/ppq_data/tokens --voice-prompt /workspace/ppq_data/voice_prompt.wav
"""
from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np

from ..config import PPQModelConfig, SAMPLE_RATE
from ..progress import (
    PidLock, ProgressTracker, read_jsonl, repair_jsonl,
    session_header, setup_file_logger, fmt_duration,
)
from .dialogue import Dialogue


def _save_sample(out_dir: Path, name: str, codes: np.ndarray, mask: np.ndarray,
                 source: str) -> dict:
    """Atomic npz write (temp file + rename)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    tmp = out_dir / f"{name}.tmp.npz"
    np.savez_compressed(tmp, codes=codes.astype(np.int32),
                        loss_mask=mask.astype(np.float16))
    os.replace(tmp, path)
    return {"path": str(path), "name": name, "frames": int(codes.shape[1]), "source": source}


def _load_done_set(manifest_path: Path) -> set[str]:
    """Names of items fully written (present in the manifest)."""
    repair_jsonl(manifest_path)
    done = set()
    if manifest_path.exists():
        for e in read_jsonl(manifest_path):
            if "name" in e:
                done.add(e["name"])
            elif "path" in e:  # legacy entries without explicit name
                done.add(Path(e["path"]).stem)
    return done


def _load_failed(failed_path: Path, done: set[str]) -> dict[str, int]:
    """name -> number of failed attempts (excluding items that later succeeded)."""
    attempts: dict[str, int] = {}
    if failed_path.exists():
        repair_jsonl(failed_path)
        for e in read_jsonl(failed_path):
            name = e.get("name")
            if name:
                attempts[name] = attempts.get(name, 0) + 1
    for name in done:
        attempts.pop(name, None)
    return attempts


def _record_failure(failed_path: Path, name: str, global_idx: int, error: str) -> None:
    with open(failed_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "global_idx": global_idx,
                            "error": error[-2000:]}) + "\n")


def _log_dir(args) -> Path:
    if args.log_dir:
        return Path(args.log_dir)
    return Path(args.out).parent / "logs"


def run_synth(args):
    log_dir = _log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = PidLock(log_dir / f"worker-synth-{args.shard:04d}.pid")
    if not lock.acquire():
        print(f"[synth shard {args.shard}] another worker is already running; exiting.")
        return
    try:
        _run_synth_locked(args, log_dir)
    finally:
        lock.release()


def _run_synth_locked(args, log_dir: Path):
    from ..text_tokenizer import QwenTextTokenizer
    from ..checkpoint import get_mimi
    from . import mimi_tokenize
    from .tts_kokoro import KokoroTTS
    from .stitch import stitch_dialogue

    log_path = log_dir / f"worker-synth-{args.shard:04d}.log"
    session_header(log_path, f"synth shard {args.shard}/{args.num_shards}")
    log = setup_file_logger("ppq.data.worker", log_path)

    cfg = PPQModelConfig.from_json(args.model_config) if args.model_config else PPQModelConfig()
    device = "cuda"
    mimi = get_mimi(args.mimi_weight, device)
    tokenizer = QwenTextTokenizer(args.tokenizer, cfg)
    mimi_tokenize.set_text_encoder(tokenizer.encode)
    tts = KokoroTTS()

    # Voice prompt agent codes (computed once).
    vp_codes = None
    if args.voice_prompt:
        import sphn
        vp_wav, vp_sr = sphn.read(args.voice_prompt)
        if vp_sr != SAMPLE_RATE:
            vp_wav = sphn.resample(vp_wav, vp_sr, SAMPLE_RATE)
        vp_codes = mimi_tokenize.encode_channel(mimi, vp_wav[0])

    repair_jsonl(args.scripts)
    with open(args.scripts, "r", encoding="utf-8") as f:
        scripts = [json.loads(line) for line in f if line.strip()]
    # Global index -> stable shard assignment and stable file names.
    owned = [(gi, s) for gi, s in enumerate(scripts) if gi % args.num_shards == args.shard]
    total = len(owned)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) / f"shard-{args.shard:04d}"
    manifest_path = Path(args.out) / f"manifest-{args.shard:04d}.jsonl"
    failed_path = log_dir / f"failed-synth-{args.shard:04d}.jsonl"
    done = _load_done_set(manifest_path)
    failed = _load_failed(failed_path, done)
    owned_names = {f"conv-{gi:07d}" for gi, _ in owned}
    prior_done = len(owned_names & done)
    prior_failed = sum(1 for n in owned_names if failed.get(n, 0) >= args.max_attempts)

    tracker = ProgressTracker(log_dir / f"progress-synth-{args.shard:04d}.json",
                              stage=f"synth-{args.shard:04d}", total=total,
                              prior_done=prior_done, prior_failed=prior_failed,
                              unit="dlg")
    tracker.save("running")
    log.info("shard %d/%d: %d dialogues | %d already done | %d permanently failed | %d to do",
             args.shard, args.num_shards, total, prior_done, prior_failed, tracker.remaining)

    t0_stats = {"backchannels": 0, "interrupts": 0, "mid_pauses": 0, "words_estimated": 0}
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for gi, script in owned:
            name = f"conv-{gi:07d}"
            if name in done:
                continue
            if not args.retry_failed and failed.get(name, 0) >= args.max_attempts:
                continue
            # Deterministic per-dialogue RNG: resume produces identical audio.
            rng = random.Random(args.seed * 1_000_003 + gi)
            try:
                dlg = Dialogue.from_dict(script)
                turn_audio = []
                for turn in dlg.turns:
                    voice = dlg.agent_voice if turn.speaker == "agent" else dlg.user_voice
                    turn_audio.append(tts.synth(turn.text, voice))
                stitch = stitch_dialogue(dlg, turn_audio, rng)
                prompt_ids = tokenizer.encode_system_prompt(dlg.system_prompt)
                codes, mask = mimi_tokenize.build_conversation_codes(
                    stitch.stereo, stitch.agent_words, prompt_ids, vp_codes, mimi, cfg)
                entry = _save_sample(out_dir, name, codes, mask, source="synth")
                entry.update({
                    "global_idx": gi,
                    "duration_s": stitch.stats["duration_s"],
                    "backchannels": stitch.stats["n_backchannels"],
                    "interrupts": stitch.stats["n_interrupts"],
                    "mid_pauses": stitch.stats["n_mid_pauses"],
                    "words_estimated": stitch.stats["words_estimated"],
                })
                manifest.write(json.dumps(entry) + "\n")
                manifest.flush()
                done.add(name)
                t0_stats["backchannels"] += stitch.stats["n_backchannels"]
                t0_stats["interrupts"] += stitch.stats["n_interrupts"]
                t0_stats["mid_pauses"] += stitch.stats["n_mid_pauses"]
                t0_stats["words_estimated"] += int(stitch.stats["words_estimated"])
                tracker.add(audio_seconds=stitch.stats["duration_s"])
            except Exception:
                err = traceback.format_exc()
                log.error("FAILED %s (attempt %d):\n%s", name,
                          failed.get(name, 0) + 1, err)
                _record_failure(failed_path, name, gi, err)
                failed[name] = failed.get(name, 0) + 1
                if failed[name] >= args.max_attempts:
                    tracker.add(failed=True)
            tracker.save()
            tracker.log_line(log)
            if args.max_items and tracker.session_done >= args.max_items:
                log.info("stopping after --max-items=%d new items", args.max_items)
                tracker.save("paused")
                return
    tracker.save("finished")
    log.info("shard %d FINISHED: %d done, %d failed, %s wall, stats %s",
             args.shard, tracker.done, tracker.failed,
             fmt_duration(__import__("time").time() - tracker.started_at), t0_stats)


def run_libritts(args):
    log_dir = _log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = PidLock(log_dir / f"worker-libritts-{args.shard:04d}.pid")
    if not lock.acquire():
        print(f"[libritts shard {args.shard}] another worker is already running; exiting.")
        return
    try:
        _run_libritts_locked(args, log_dir)
    finally:
        lock.release()


def _run_libritts_locked(args, log_dir: Path):
    from ..checkpoint import get_mimi
    from . import mimi_tokenize
    from .external import iter_libritts_monologues

    log_path = log_dir / f"worker-libritts-{args.shard:04d}.log"
    session_header(log_path, f"libritts shard {args.shard}/{args.num_shards}")
    log = setup_file_logger("ppq.data.worker", log_path)

    cfg = PPQModelConfig.from_json(args.model_config) if args.model_config else PPQModelConfig()
    mimi = get_mimi(args.mimi_weight, "cuda")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) / f"shard-{args.shard:04d}"
    manifest_path = Path(args.out) / f"manifest-libritts-{args.shard:04d}.jsonl"
    failed_path = log_dir / f"failed-libritts-{args.shard:04d}.jsonl"
    done = _load_done_set(manifest_path)
    # Rough total estimate: 100 h at ~45 s/group -> used only for the % display.
    est_total = int(args.max_hours * 3600 / 45.0 / max(1, args.num_shards))
    tracker = ProgressTracker(log_dir / f"progress-libritts-{args.shard:04d}.json",
                              stage=f"libritts-{args.shard:04d}", total=est_total,
                              prior_done=len(done), unit="seg")
    tracker.save("running")
    log.info("shard %d/%d: %d segments already done; target %.0f h",
             args.shard, args.num_shards, len(done), args.max_hours / max(1, args.num_shards))

    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for gi, mono in iter_libritts_monologues(
                max_seconds=args.max_hours * 3600, shard=args.shard,
                num_shards=args.num_shards):
            name = f"libritts-{gi:07d}"
            if name in done:
                continue
            try:
                codes, mask = mimi_tokenize.build_monologue_codes(mono, mimi, cfg)
                entry = _save_sample(out_dir, name, codes, mask, source="libritts")
                entry["global_idx"] = gi
                manifest.write(json.dumps(entry) + "\n")
                manifest.flush()
                done.add(name)
                tracker.add(audio_seconds=len(mono) / SAMPLE_RATE)
            except Exception:
                err = traceback.format_exc()
                log.error("FAILED %s:\n%s", name, err)
                _record_failure(failed_path, name, gi, err)
                tracker.add(failed=True)
            tracker.save()
            tracker.log_line(log)
            if args.max_items and tracker.session_done >= args.max_items:
                tracker.save("paused")
                return
    tracker.save("finished")
    log.info("shard %d FINISHED: %d segments, %.1f h audio",
             args.shard, tracker.done, tracker.audio_hours)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["synth", "libritts"], required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--scripts", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--mimi-weight", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--model-config", type=str, default="")
    p.add_argument("--voice-prompt", type=str, default="")
    p.add_argument("--max-hours", type=float, default=100.0)  # libritts only
    p.add_argument("--log-dir", type=str, default="")         # default: <out>/../logs
    p.add_argument("--seed", type=int, default=0)             # base seed for per-dialogue RNG
    p.add_argument("--max-attempts", type=int, default=3)     # give up on an item after N failures
    p.add_argument("--retry-failed", action="store_true")     # re-attempt items that hit --max-attempts
    p.add_argument("--max-items", type=int, default=0)        # debug: stop after N new items
    args = p.parse_args(argv)
    if args.stage == "synth":
        if not args.scripts:
            p.error("--scripts required for stage synth")
        run_synth(args)
    else:
        run_libritts(args)


if __name__ == "__main__":
    main()
