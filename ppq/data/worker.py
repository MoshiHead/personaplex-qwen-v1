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
import time
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


# ---------------------------------------------------------------------------
# Real-conversation stages (otoearth HF datasets)
# ---------------------------------------------------------------------------

def _session_speech_words(spec, meta, sid, stereo, data_root, get_asr_engine, args):
    """Per-channel (speech intervals, word list) for one prepared stereo session.

    video: from the bundled transcripts. task: Whisper over energy-VAD regions
    (cached). turn: SRT speech intervals; words = short SRT segments (even
    spread) + Whisper for long segments (cached)."""
    from . import asr as asr_mod
    from .oto_sources import (intervals_from_words, long_srt_segments,
                              merge_intervals, srt_words)
    from .realconv import energy_vad, merge_regions

    if spec.kind == "video":
        return meta.speech, meta.words
    speech, words = [[], []], [[], []]
    for ch in (0, 1):
        cpath = asr_mod.cache_path(data_root, spec.name, sid, ch)
        cached = asr_mod.load_cached_words(cpath)
        if cached is None:
            if spec.kind == "task":
                regions = merge_regions(energy_vad(stereo[ch], SAMPLE_RATE),
                                        max_region_s=60.0)
                w = asr_mod.transcribe_regions(get_asr_engine(), stereo[ch],
                                               SAMPLE_RATE, regions)
            else:  # turn: only the long SRT segments
                regions = long_srt_segments(meta.srt_segments[ch], args.long_segment_s)
                w = asr_mod.transcribe_regions(
                    get_asr_engine(), stereo[ch], SAMPLE_RATE, regions,
                    accept=lambda ws, we, r: r[0] <= 0.5 * (ws + we) <= r[1])
            asr_mod.save_cached_words(cpath, regions, w)
            cached = w
        if spec.kind == "task":
            words[ch] = cached
            speech[ch] = merge_intervals(intervals_from_words(cached))
        else:
            merged = srt_words(meta.srt_segments[ch], args.long_segment_s) + cached
            merged.sort(key=lambda x: (x[1], x[2]))
            words[ch] = merged
            speech[ch] = meta.speech[ch]
    return speech, words


def run_asr(args):
    log_dir = _log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = PidLock(log_dir / f"worker-asr-{args.source}-{args.shard:04d}.pid")
    if not lock.acquire():
        print(f"[asr {args.source} shard {args.shard}] another worker is already running; exiting.")
        return
    try:
        _run_asr_locked(args, log_dir)
    finally:
        lock.release()


def _run_asr_locked(args, log_dir: Path):
    """Warm the Whisper word cache for the task sets (turn-104h ASR runs
    inline in the real stage to avoid downloading its 145 GB twice)."""
    import io

    import soundfile as sf

    from . import asr as asr_mod
    from .oto_sources import SOURCES, TarStore, get_hf_token, list_sessions, load_tar_session
    from .realconv import energy_vad, merge_regions

    spec = SOURCES[args.source]
    stage = f"asr-{args.source}-{args.shard:04d}"
    log_path = log_dir / f"worker-{stage}.log"
    session_header(log_path, f"asr {args.source} shard {args.shard}/{args.num_shards}")
    log = setup_file_logger("ppq.data.worker", log_path)

    data_root = Path(args.data_root) if args.data_root else Path(args.out).parent
    token = get_hf_token(args.hf_token_env)
    sessions = list_sessions(spec, data_root, token)
    owned = [sid for i, sid in enumerate(sessions) if i % args.num_shards == args.shard]

    failed_path = log_dir / f"failed-{stage}.jsonl"
    failed = _load_failed(failed_path, set())
    tracker = ProgressTracker(log_dir / f"progress-{stage}.json", stage=stage,
                              total=len(owned),
                              prior_failed=sum(1 for s in owned
                                               if failed.get(s, 0) >= args.max_attempts),
                              unit="sess")
    tracker.save("running")

    if spec.asr_mode != "full":
        log.info("source %s has asr_mode=%s; ASR runs inline in the real stage - nothing to do",
                 spec.name, spec.asr_mode)
        tracker.save("finished")
        return

    engine = asr_mod.WhisperASR(args.asr_model, args.device)
    log.info("ASR engine device: %s", getattr(engine.pipe, "device", "?"))
    store = TarStore(spec, data_root, token)
    smap = store.session_map()
    log.info("shard %d/%d: %d sessions to transcribe", args.shard, args.num_shards, len(owned))

    def _cache(sid: str, ch: int) -> Path:
        return asr_mod.cache_path(data_root, spec.name, sid, ch)

    def _cached(sid: str) -> bool:
        return all(asr_mod.load_cached(_cache(sid, ch)) is not None for ch in (0, 1))

    def _cached_duration(sid: str) -> float:
        ends = [float(r[1]) for ch in (0, 1)
                for r in (asr_mod.load_cached(_cache(sid, ch)) or {}).get("regions", [])]
        return max(ends, default=0.0)

    # count sessions already cached by earlier runs so progress reflects reality
    for sid in owned:
        if _cached(sid):
            tracker.add(audio_seconds=_cached_duration(sid))
    if tracker.done:
        tracker.save()
        tracker.log_line(log)

    # Retry passes: a session that fails transiently (network blip, expired
    # signed URL) is retried up to max_attempts times within this run, so
    # "finished" really means every session is cached or permanently failed.
    for pass_no in range(1, args.max_attempts + 1):
        pending = [sid for sid in owned
                   if not _cached(sid)
                   and (args.retry_failed or failed.get(sid, 0) < args.max_attempts)]
        if not pending:
            break
        if pass_no > 1:
            log.info("retry pass %d/%d: %d sessions still pending",
                     pass_no, args.max_attempts, len(pending))
            time.sleep(30.0)
        for sid in pending:
            log.info("transcribing session %s (done %d/%d)", sid, tracker.done, len(owned))
            try:
                meta, flac = load_tar_session(spec, store, smap[sid], sid)
                data, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=True)
                for ch in (0, 1):
                    if asr_mod.load_cached(_cache(sid, ch)) is not None:
                        continue
                    mono = np.ascontiguousarray(data[:, ch])
                    regions = merge_regions(energy_vad(mono, sr), max_region_s=60.0)
                    w = asr_mod.transcribe_regions(engine, mono, sr, regions)
                    asr_mod.save_cached_words(_cache(sid, ch), regions, w)
                    tracker.save()  # heartbeat: keep the watchdog's updated_ts fresh
                tracker.add(audio_seconds=meta.duration_s)
            except Exception:
                err = traceback.format_exc()
                log.error("FAILED session %s (attempt %d):\n%s", sid,
                          failed.get(sid, 0) + 1, err)
                _record_failure(failed_path, sid, -1, err)
                failed[sid] = failed.get(sid, 0) + 1
                if failed[sid] >= args.max_attempts:
                    tracker.add(failed=True)
            tracker.save()
            tracker.log_line(log)
            if args.max_items and tracker.session_done >= args.max_items:
                tracker.save("paused")
                return
    tracker.save("finished")
    log.info("shard %d FINISHED: %d sessions transcribed, %d failed",
             args.shard, tracker.done, tracker.failed)


def _real_process_session(args, spec, sid, store, smap, mimi, tokenizer, cfg,
                          data_root, get_asr_engine, out_dir, manifest,
                          done_samples, marker_dir) -> float:
    """Load one session, build all its chunk/role samples, write the session
    marker. Returns the session duration in seconds."""
    import io
    import shutil

    import soundfile as sf

    from . import realconv
    from .oto_sources import load_tar_session, load_turn_session, safe_name

    if spec.kind == "turn":
        meta, wav_paths = load_turn_session(spec, sid, data_root,
                                            get_hf_token_cached(args))
        chans, sr = [], None
        for p in wav_paths:
            w, sr = sf.read(str(p), dtype="float32", always_2d=True)
            chans.append(np.ascontiguousarray(w[:, 0]))
        stereo = realconv.prepare_stereo(chans[0], chans[1], sr)
    else:
        meta, flac = load_tar_session(spec, store, smap[sid], sid)
        data, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=True)
        stereo = realconv.prepare_stereo(
            np.ascontiguousarray(data[:, 0]),
            np.ascontiguousarray(data[:, 1]), sr)
    speech, words = _session_speech_words(spec, meta, sid, stereo, data_root,
                                          get_asr_engine, args)
    sep = realconv.separation_db(stereo[0], stereo[1], speech[0], speech[1],
                                 SAMPLE_RATE)
    if sep < args.min_separation_db:
        raise ValueError(
            f"channel separation {sep:.1f} dB < {args.min_separation_db} dB")
    prep = realconv.PreparedSession(meta=meta, stereo=stereo,
                                    speech=speech, words=words)
    vp_mode = args.voice_prompt_mode or spec.voice_prompt_mode
    wrote = []
    for name, codes, mask, info in realconv.build_chunk_samples(
            prep, mimi, tokenizer, cfg, voice_prompt_mode=vp_mode,
            min_agent_speech=args.min_agent_speech,
            voice_prompt_seconds=args.voice_prompt_seconds,
            segment_frames=args.segment_frames,
            min_chunk_s=args.min_chunk_s, max_chunk_s=args.max_chunk_s,
            min_silence_s=args.min_silence_s):
        if name in done_samples:
            wrote.append(name)
            continue
        entry = _save_sample(out_dir, name, codes, mask, source=f"real-{spec.name}")
        entry.update(info)
        manifest.write(json.dumps(entry) + "\n")
        manifest.flush()
        done_samples.add(name)
        wrote.append(name)
    marker_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(sid)
    tmp = marker_dir / f"{safe}.done.tmp"
    tmp.write_text(json.dumps({"session_id": sid, "samples": wrote}) + "\n",
                   encoding="utf-8")
    os.replace(tmp, marker_dir / f"{safe}.done")
    if spec.kind == "turn" and not args.keep_raw:
        shutil.rmtree(data_root / "raw" / spec.name / sid, ignore_errors=True)
    return prep.duration_s


def get_hf_token_cached(args):
    from .oto_sources import get_hf_token
    return get_hf_token(args.hf_token_env)


def run_real(args):
    log_dir = _log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = PidLock(log_dir / f"worker-real-{args.source}-{args.shard:04d}.pid")
    if not lock.acquire():
        print(f"[real {args.source} shard {args.shard}] another worker is already running; exiting.")
        return
    try:
        _run_real_locked(args, log_dir)
    finally:
        lock.release()


def _run_real_locked(args, log_dir: Path):
    from ..checkpoint import get_mimi
    from ..text_tokenizer import QwenTextTokenizer
    from . import asr as asr_mod
    from . import mimi_tokenize
    from .oto_sources import SOURCES, TarStore, get_hf_token, list_sessions, safe_name

    spec = SOURCES[args.source]
    stage = f"real-{args.source}-{args.shard:04d}"
    log_path = log_dir / f"worker-{stage}.log"
    session_header(log_path, f"real {args.source} shard {args.shard}/{args.num_shards}")
    log = setup_file_logger("ppq.data.worker", log_path)

    cfg = PPQModelConfig.from_json(args.model_config) if args.model_config else PPQModelConfig()
    mimi = get_mimi(args.mimi_weight, args.device)
    tokenizer = QwenTextTokenizer(args.tokenizer, cfg)
    mimi_tokenize.set_text_encoder(tokenizer.encode)
    token = get_hf_token(args.hf_token_env)
    data_root = Path(args.data_root) if args.data_root else Path(args.out).parent

    sessions = list_sessions(spec, data_root, token)
    owned = [sid for i, sid in enumerate(sessions) if i % args.num_shards == args.shard]
    total = len(owned)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) / f"shard-{args.shard:04d}"
    manifest_path = Path(args.out) / f"manifest-real-{args.source}-{args.shard:04d}.jsonl"
    failed_path = log_dir / f"failed-{stage}.jsonl"
    marker_dir = data_root / "markers" / f"real-{args.source}"
    done_samples = _load_done_set(manifest_path)
    failed = _load_failed(failed_path, done_samples)
    done_sessions = ({p.stem for p in marker_dir.glob("*.done")}
                     if marker_dir.exists() else set())
    prior_done = sum(1 for s in owned if safe_name(s) in done_sessions)
    prior_failed = sum(1 for s in owned if failed.get(s, 0) >= args.max_attempts)

    tracker = ProgressTracker(log_dir / f"progress-{stage}.json", stage=stage,
                              total=total, prior_done=prior_done,
                              prior_failed=prior_failed, unit="sess")
    tracker.save("running")
    log.info("shard %d/%d: %d sessions | %d already done | %d permanently failed | %d to do",
             args.shard, args.num_shards, total, prior_done, prior_failed, tracker.remaining)

    store, smap = None, None
    if spec.kind in ("video", "task"):
        store = TarStore(spec, data_root, token)
        smap = store.session_map()
    asr_engine = None

    def get_asr_engine():
        nonlocal asr_engine
        if asr_engine is None:
            asr_engine = asr_mod.WhisperASR(args.asr_model, args.device)
        return asr_engine

    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for sid in owned:
            safe = safe_name(sid)
            if safe in done_sessions:
                continue
            if not args.retry_failed and failed.get(sid, 0) >= args.max_attempts:
                continue
            log.info("processing session %s (done %d/%d)", sid, tracker.done, total)
            try:
                dur_s = _real_process_session(
                    args, spec, sid, store, smap, mimi, tokenizer, cfg,
                    data_root, get_asr_engine, out_dir, manifest,
                    done_samples, marker_dir)
                tracker.add(audio_seconds=dur_s)
            except Exception:
                err = traceback.format_exc()
                log.error("FAILED session %s (attempt %d):\n%s", sid,
                          failed.get(sid, 0) + 1, err)
                _record_failure(failed_path, sid, -1, err)
                failed[sid] = failed.get(sid, 0) + 1
                if failed[sid] >= args.max_attempts:
                    tracker.add(failed=True)
            tracker.save()
            tracker.log_line(log)
            if args.max_items and tracker.session_done >= args.max_items:
                log.info("stopping after --max-items=%d new sessions", args.max_items)
                tracker.save("paused")
                return
    tracker.save("finished")
    log.info("shard %d FINISHED: %d sessions done, %d failed, %.1f h audio",
             args.shard, tracker.done, tracker.failed, tracker.audio_hours)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["synth", "libritts", "asr", "real"], required=True)
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
    # real-conversation stages (asr / real)
    p.add_argument("--source", type=str, default="")          # key in oto_sources.SOURCES
    p.add_argument("--data-root", type=str, default="")       # default: parent of --out
    p.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--asr-model", type=str, default="openai/whisper-large-v3-turbo")
    p.add_argument("--min-chunk-s", type=float, default=20.0)
    p.add_argument("--max-chunk-s", type=float, default=80.0)
    p.add_argument("--min-silence-s", type=float, default=0.8)
    p.add_argument("--min-agent-speech", type=float, default=0.15)
    p.add_argument("--min-separation-db", type=float, default=12.0)
    p.add_argument("--voice-prompt-seconds", type=float, default=8.0)
    p.add_argument("--voice-prompt-mode", type=str, default="")  # "" -> source default
    p.add_argument("--segment-frames", type=int, default=1250)
    p.add_argument("--long-segment-s", type=float, default=4.0)  # turn-104h ASR threshold
    p.add_argument("--keep-raw", action="store_true")         # keep downloaded raw audio
    args = p.parse_args(argv)
    if args.stage == "synth":
        if not args.scripts:
            p.error("--scripts required for stage synth")
        run_synth(args)
    elif args.stage == "libritts":
        run_libritts(args)
    else:
        if not args.source:
            p.error(f"--source required for stage {args.stage}")
        run_asr(args) if args.stage == "asr" else run_real(args)


if __name__ == "__main__":
    main()
