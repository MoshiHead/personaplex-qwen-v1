"""PPQ WebSocket server — same wire protocol as the PersonaPlex server, so the
stock PersonaPlex web client works unmodified.

Differences vs moshi/server.py:
  - PPQLMModel checkpoint instead of the Helium model.
  - Qwen tokenizer + incremental UTF-8-safe transcript decoding (PAD/EPAD
    tokens are skipped).
  - Voice prompts are .wav files; requested names that don't exist (including
    the stock client's built-in *.pt entries) are mapped to --default-voice.

  python -m ppq.serve.server --ppq-checkpoint ... --mimi-weight ... \
      --tokenizer Qwen/Qwen3-1.7B-Base --voice-prompt-dir voices/ \
      --default-voice af_heart.wav --static /path/to/dist --host 0.0.0.0 --port 8998
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import time

import aiohttp
from aiohttp import web
import numpy as np
import sphn
import torch

from moshi.utils.logging import setup_logger, ColorizedLog
from moshi.utils.connection import create_ssl_context, get_lan_ip


def seed_all(seed: int):
    import random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def torch_auto_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

from ..checkpoint import load_pretrained, get_mimi
from ..config import FRAME_SIZE, SAMPLE_RATE, FRAME_RATE_HZ
from ..lmgen import PPQLMGen
from ..text_tokenizer import QwenTextTokenizer, StreamingTextDecoder

logger = setup_logger(__name__)


@dataclass
class ServerState:
    mimi: object
    other_mimi: object
    lm_gen: PPQLMGen
    tokenizer: QwenTextTokenizer
    voice_prompt_dir: str | None
    default_voice: str | None
    device: torch.device
    lock: asyncio.Lock

    def __init__(self, mimi, other_mimi, tokenizer, lm, device,
                 voice_prompt_dir=None, default_voice=None):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.tokenizer = tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.default_voice = default_voice
        self.frame_size = int(mimi.sample_rate / mimi.frame_rate)
        self.lm_gen = PPQLMGen(lm, device=device)
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def resolve_voice_prompt(self, requested: str | None) -> str | None:
        if self.voice_prompt_dir is None:
            return None
        if requested:
            path = os.path.join(self.voice_prompt_dir, requested)
            if os.path.exists(path) and not path.endswith(".pt"):
                return path
            logger.warning(
                "voice prompt '%s' not available (PPQ uses .wav prompts); "
                "falling back to default voice", requested)
        if self.default_voice is None:
            raise FileNotFoundError("no default voice configured (--default-voice)")
        return os.path.join(self.voice_prompt_dir, self.default_voice)

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote
        peer_port = request.transport.get_extra_info("peername")[1]
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        voice_prompt_path = self.resolve_voice_prompt(request.query.get("voice_prompt"))
        if self.lm_gen.voice_prompt != voice_prompt_path:
            self.lm_gen.load_voice_prompt(voice_prompt_path)
        text_prompt = request.query.get("text_prompt", "")
        self.lm_gen.text_prompt_tokens = (
            self.tokenizer.encode_system_prompt(text_prompt) if len(text_prompt) > 0 else [])
        seed = int(request.query["seed"]) if "seed" in request.query else None
        decoder = StreamingTextDecoder(self.tokenizer)
        decoder.reset()

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED,
                                        aiohttp.WSMsgType.CLOSE):
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    message = message.data
                    if not isinstance(message, bytes) or len(message) == 0:
                        continue
                    if message[0] == 1:  # audio
                        opus_reader.append_bytes(message[1:])
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                all_pcm_data = pcm if all_pcm_data is None else np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk).to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        opus_writer.append_pcm(main_pcm[0, 0].cpu().numpy())
                        text_piece = decoder.decode_step(int(tokens[0, 0, 0]))
                        if text_piece:
                            await ws.send_bytes(b"\x02" + bytes(text_piece, encoding="utf8"))

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            decoder.reset()

            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True

            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
        clog.log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str,
                        help="Path to web client static dir; 'none' disables static serving.")
    parser.add_argument("--ppq-checkpoint", required=True, type=str,
                        help="PPQ checkpoint dir (config.json + model.safetensors).")
    parser.add_argument("--mimi-weight", required=True, type=str)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base", type=str,
                        help="HF repo id or local path of the Qwen tokenizer.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--voice-prompt-dir", type=str, default=None,
                        help="Directory with .wav voice prompts (agent voice).")
    parser.add_argument("--default-voice", type=str, default=None,
                        help="Filename inside --voice-prompt-dir used when the client's "
                             "requested voice is unavailable.")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--ssl", type=str, default=None,
                        help="Directory with key.pem and cert.pem for https.")
    args = parser.parse_args()

    device = torch_auto_device(args.device)
    seed_all(42424242)

    logger.info("loading mimi")
    mimi = get_mimi(args.mimi_weight, device)
    other_mimi = get_mimi(args.mimi_weight, device)
    logger.info("loading PPQ model from %s", args.ppq_checkpoint)
    lm = load_pretrained(args.ppq_checkpoint, device=device)
    tokenizer = QwenTextTokenizer(args.tokenizer, lm.cfg)
    if args.voice_prompt_dir is not None:
        assert os.path.isdir(args.voice_prompt_dir), args.voice_prompt_dir
        if args.default_voice is not None:
            assert os.path.exists(os.path.join(args.voice_prompt_dir, args.default_voice))

    state = ServerState(
        mimi=mimi, other_mimi=other_mimi, tokenizer=tokenizer, lm=lm, device=device,
        voice_prompt_dir=args.voice_prompt_dir, default_voice=args.default_voice)
    state.lm_gen.temp = args.temp
    state.lm_gen.temp_text = args.temp_text
    logger.info("warming up the model")
    state.warmup()

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    static_path = None if args.static in (None, "none") else args.static
    if static_path is not None:
        assert os.path.exists(static_path), f"static path missing: {static_path}"

        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info("Access the Web UI at %s://%s:%d", protocol, host_ip, args.port)
    web.run_app(app, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()

