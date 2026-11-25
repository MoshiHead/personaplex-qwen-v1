"""Evaluation helpers: per-codebook losses on held-out data, offline
generation against a recorded user utterance, and response-delay metrics."""
from __future__ import annotations

import logging
import wave

import numpy as np
import torch

from ..config import PPQModelConfig, FRAME_SIZE, SAMPLE_RATE, FRAME_RATE_HZ
from ..lmgen import PPQLMGen
from ..text_tokenizer import StreamingTextDecoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_per_codebook(model, loader, n_batches: int, device, mcfg: PPQModelConfig) -> dict:
    """Unweighted CE per stream on held-out samples."""
    model.eval()
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    it = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        codes = batch["codes"].to(device)
        mask = batch["loss_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.forward_train(codes)
        # text
        sel = out.text_mask[:, 0] & (mask[:, 0] > 0)
        if bool(sel.any()):
            ce = torch.nn.functional.cross_entropy(
                out.text_logits[:, 0][sel].float(), codes[:, 0][sel])
            sums["text"] = sums.get("text", 0.0) + float(ce)
            counts["text"] = counts.get("text", 0) + 1
        for cb in range(out.logits.shape[1]):
            sel = out.mask[:, cb] & (mask[:, cb + 1] > 0)
            if not bool(sel.any()):
                continue
            ce = torch.nn.functional.cross_entropy(
                out.logits[:, cb][sel].float(), codes[:, cb + 1][sel])
            key = f"cb{cb}"
            sums[key] = sums.get(key, 0.0) + float(ce)
            counts[key] = counts.get(key, 0) + 1
    model.train()
    return {k: sums[k] / max(1, counts[k]) for k in sums}


@torch.no_grad()
def generate_offline(model, mimi, tokenizer, user_wav: np.ndarray,
                     voice_prompt_wav: np.ndarray | None,
                     text_prompt: str, device, max_seconds: float = 30.0,
                     tail_seconds: float = 6.0, temp: float = 0.8, temp_text: float = 0.7,
                     top_k: int = 250, top_k_text: int = 25) -> tuple[np.ndarray, str]:
    """Run a full-duplex episode offline: the recorded user audio is streamed
    in, the agent is sampled. Returns (agent wav float32 24 kHz, transcript)."""
    import sphn
    from ..config import SILENCE_TOKENS
    from ..lmgen import PPQLMGen

    lm_gen = PPQLMGen(model, device=device, check=False,
                      temp=temp, temp_text=temp_text, top_k=top_k, top_k_text=top_k_text)
    decoder = StreamingTextDecoder(tokenizer)
    agent_pcm: list[np.ndarray] = []
    user_codes = mimi.encode(
        torch.from_numpy(np.ascontiguousarray(user_wav)).float()[None, None].to(device))
    n_user_frames = user_codes.shape[2]
    tail_frames = int(tail_seconds * FRAME_RATE_HZ)
    total_frames = min(int(max_seconds * FRAME_RATE_HZ), n_user_frames + tail_frames)

    silence = torch.zeros(1, 1, FRAME_SIZE, device=device)
    with lm_gen.streaming(1), mimi.streaming(1):
        if voice_prompt_wav is not None:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            write_wav(tmp, voice_prompt_wav)
            lm_gen.load_voice_prompt(tmp)
            os.unlink(tmp)
        lm_gen.text_prompt_tokens = tokenizer.encode_system_prompt(text_prompt) if text_prompt else []
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()
        for t in range(total_frames):
            if t < n_user_frames:
                frame_codes = user_codes[:, :, t: t + 1]
            else:
                frame_codes = mimi.encode(silence)
            tokens = lm_gen.step(frame_codes)
            if tokens is None:
                continue
            pcm = mimi.decode(tokens[:, 1:9])
            agent_pcm.append(pcm[0, 0].cpu().numpy())
            piece = decoder.decode_step(int(tokens[0, 0, 0]))
    wav = np.concatenate(agent_pcm) if agent_pcm else np.zeros(0, dtype=np.float32)
    text = tokenizer.decode(decoder.ids)
    return wav, text


def write_wav(path: str, wav: np.ndarray, sr: int = SAMPLE_RATE):
    pcm = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm16.tobytes())


def response_delays(agent_wav: np.ndarray, user_wav: np.ndarray,
                    energy_threshold: float = 0.01, min_gap_s: float = 0.2) -> list[float]:
    """For each user speech segment, the delay from its end to the start of
    the next agent speech segment (negative = barged in)."""
    def segments(wav: np.ndarray) -> list[tuple[float, float]]:
        n = len(wav) // FRAME_SIZE
        if n == 0:
            return []
        e = wav[: n * FRAME_SIZE].reshape(n, FRAME_SIZE).std(axis=1)
        active = e > energy_threshold
        segs = []
        start = None
        for i, a in enumerate(active):
            if a and start is None:
                start = i
            elif not a and start is not None:
                if (i - start) / FRAME_RATE_HZ >= min_gap_s:
                    segs.append((start / FRAME_RATE_HZ, i / FRAME_RATE_HZ))
                start = None
        if start is not None:
            segs.append((start / FRAME_RATE_HZ, len(active) / FRAME_RATE_HZ))
        return segs

    delays = []
    user_segs = segments(user_wav)
    agent_segs = segments(agent_wav)
    for us, ue in user_segs:
        for as_, ae in agent_segs:
            if ae > us and as_ > us - 1.0:
                delays.append(as_ - ue)
                break
    return delays
