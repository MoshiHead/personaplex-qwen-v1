"""Offline full-duplex generation against a recorded user utterance.

  python -m ppq.serve.offline --ppq-checkpoint /workspace/ppq_runs/run1/checkpoints/final \
      --mimi-weight /workspace/models/mimi.safetensors \
      --tokenizer Qwen/Qwen3-1.7B-Base \
      --user-wav user_question.wav --voice-prompt voice_prompt.wav \
      --text-prompt "You are a helpful assistant." --out agent_reply.wav
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from ..checkpoint import load_pretrained, get_mimi
from ..config import SAMPLE_RATE
from ..text_tokenizer import QwenTextTokenizer
from ..train.evaluate import generate_offline, write_wav, response_delays

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppq.serve.offline")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ppq-checkpoint", required=True)
    p.add_argument("--mimi-weight", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--user-wav", required=True)
    p.add_argument("--voice-prompt", default="")
    p.add_argument("--text-prompt", default="You are a helpful voice assistant.")
    p.add_argument("--out", default="agent_reply.wav")
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--temp-text", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import sphn
    model = load_pretrained(args.ppq_checkpoint, device=args.device)
    mimi = get_mimi(args.mimi_weight, args.device)
    tokenizer = QwenTextTokenizer(args.tokenizer, model.cfg)

    user_wav, usr_sr = sphn.read(args.user_wav)
    if usr_sr != SAMPLE_RATE:
        user_wav = sphn.resample(user_wav, usr_sr, SAMPLE_RATE)
    user_wav = user_wav[0].astype(np.float32)
    vp_wav = None
    if args.voice_prompt:
        vp_wav, vp_sr = sphn.read(args.voice_prompt)
        if vp_sr != SAMPLE_RATE:
            vp_wav = sphn.resample(vp_wav, vp_sr, SAMPLE_RATE)
        vp_wav = vp_wav[0].astype(np.float32)

    agent_wav, transcript = generate_offline(
        model, mimi, tokenizer, user_wav, vp_wav, args.text_prompt,
        device=args.device, max_seconds=args.max_seconds,
        temp=args.temp, temp_text=args.temp_text)
    write_wav(args.out, agent_wav)
    delays = response_delays(agent_wav, user_wav)
    logger.info("agent transcript: %s", transcript)
    if delays:
        logger.info("response delays (s): mean %.3f median %.3f n %d",
                    float(np.mean(delays)), float(np.median(delays)), len(delays))
    logger.info("wrote %s (%.1fs)", args.out, len(agent_wav) / SAMPLE_RATE)


if __name__ == "__main__":
    main()
