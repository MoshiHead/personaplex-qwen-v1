"""Sharded data-pipeline worker. The data notebook launches one process per
GPU, each handling a disjoint subset of the scripts.

Stages:
  synth    : scripts JSONL -> Kokoro TTS -> stereo stitch -> mimi tokenize -> npz shards
  libritts : LibriTTS-R utterances -> monologue npz shards (Stage A data)

Example:
  CUDA_VISIBLE_DEVICES=0 python -m ppq.data.worker --stage synth \
      --shard 0 --num-shards 8 --scripts /workspace/ppq_data/scripts.jsonl \
      --out /workspace/ppq_data/tokens --voice-prompt /workspace/ppq_data/voice_prompt.wav
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

from ..config import PPQModelConfig, SAMPLE_RATE
from .dialogue import Dialogue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ppq.data.worker")


def _save_sample(out_dir: Path, name: str, codes: np.ndarray, mask: np.ndarray,
                 source: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, codes=codes.astype(np.int32),
                        loss_mask=mask.astype(np.float16))
    return {"path": str(path), "frames": int(codes.shape[1]), "source": source}


def run_synth(args):
    from ..text_tokenizer import QwenTextTokenizer
    from ..checkpoint import get_mimi
    from . import mimi_tokenize
    from .tts_kokoro import KokoroTTS
    from .stitch import stitch_dialogue

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

    with open(args.scripts, "r", encoding="utf-8") as f:
        scripts = [json.loads(line) for line in f if line.strip()]
    scripts = scripts[args.shard:: args.num_shards]
    logger.info("shard %d/%d: %d dialogues", args.shard, args.num_shards, len(scripts))

    out_dir = Path(args.out) / f"shard-{args.shard:04d}"
    manifest_path = Path(args.out) / f"manifest-{args.shard:04d}.jsonl"
    rng = random.Random(args.seed + args.shard)
    stats_acc = {"backchannels": 0, "interrupts": 0, "mid_pauses": 0, "words_estimated": 0}
    n_done = n_skip = 0
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for i, script in enumerate(scripts):
            name = f"conv-{args.shard:04d}-{i:06d}"
            if (out_dir / f"{name}.npz").exists():
                n_skip += 1
                continue
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
                    "duration_s": stitch.stats["duration_s"],
                    "backchannels": stitch.stats["n_backchannels"],
                    "interrupts": stitch.stats["n_interrupts"],
                    "mid_pauses": stitch.stats["n_mid_pauses"],
                    "words_estimated": stitch.stats["words_estimated"],
                })
                manifest.write(json.dumps(entry) + "\n")
                manifest.flush()
                stats_acc["backchannels"] += stitch.stats["n_backchannels"]
                stats_acc["interrupts"] += stitch.stats["n_interrupts"]
                stats_acc["mid_pauses"] += stitch.stats["n_mid_pauses"]
                stats_acc["words_estimated"] += int(stitch.stats["words_estimated"])
                n_done += 1
                if n_done % 10 == 0:
                    logger.info("shard %d: %d done, %d skipped, stats %s",
                                args.shard, n_done, n_skip, stats_acc)
            except Exception:
                logger.exception("failed to process dialogue %d of shard %d", i, args.shard)
    logger.info("shard %d finished: %d written, %d skipped, stats %s",
                args.shard, n_done, n_skip, stats_acc)


def run_libritts(args):
    from ..checkpoint import get_mimi
    from . import mimi_tokenize
    from .external import iter_libritts_monologues

    cfg = PPQModelConfig.from_json(args.model_config) if args.model_config else PPQModelConfig()
    mimi = get_mimi(args.mimi_weight, "cuda")
    out_dir = Path(args.out) / f"libritts-{args.shard:04d}"
    manifest_path = Path(args.out) / f"manifest-libritts-{args.shard:04d}.jsonl"
    n_done = 0
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for i, mono in enumerate(iter_libritts_monologues(
                max_seconds=args.max_hours * 3600, min_seconds=20.0, max_seconds_per=90.0)):
            if i % args.num_shards != args.shard:
                continue
            name = f"libritts-{args.shard:04d}-{i:07d}"
            if (out_dir / f"{name}.npz").exists():
                continue
            try:
                codes, mask = mimi_tokenize.build_monologue_codes(mono, mimi, cfg)
                entry = _save_sample(out_dir, name, codes, mask, source="libritts")
                manifest.write(json.dumps(entry) + "\n")
                manifest.flush()
                n_done += 1
                if n_done % 50 == 0:
                    logger.info("libritts shard %d: %d written", args.shard, n_done)
            except Exception:
                logger.exception("failed libritts sample %d", i)
    logger.info("libritts shard %d finished: %d written", args.shard, n_done)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["synth", "libritts"], required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--scripts", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--mimi-weight", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--model-config", type=str, default="")
    p.add_argument("--voice-prompt", type=str, default="")
    p.add_argument("--max-hours", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.stage == "synth":
        assert args.scripts, "--scripts required for synth stage"
        run_synth(args)
    else:
        run_libritts(args)


if __name__ == "__main__":
    main()

