"""Regenerate the two DGX notebooks:
  notebooks/ppq_data_pipeline.ipynb
  notebooks/ppq_training.ipynb

Run: python tools/make_notebooks.py
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip().splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ===========================================================================
# Data pipeline notebook
# ===========================================================================

DATA_CELLS = []

DATA_CELLS.append(md("""
# PPQ Data Pipeline

Run this notebook on the DGX (8x H100/A100). It:
1. Sets up paths and dependencies (kokoro TTS, vLLM in its own venv).
2. Generates dialogue scripts with an instruct LLM (vLLM), then **stops the server**.
3. Launches one data worker per GPU for TTS + stereo stitching + Mimi tokenization.
4. Prepares Stage-A monologue data from LibriTTS-R (CC-BY-4.0).
5. Builds the stock PersonaPlex web client.

**Crash-safe resume everywhere.** Re-running any cell is safe:
- finished work is skipped (manifest is the source of truth, output names are
  stable global indices, `.npz` writes are atomic, torn manifests are repaired);
- per-shard append-only logs live in `DATA_ROOT/logs/worker-<stage>-<shard>.log`;
- every shard writes `logs/progress-<stage>-<shard>.json` with done/total/%/ETA,
  rendered by the *watch* cells below;
- dialogues that keep failing are retried up to `MAX_ATTEMPTS` times, then
  skipped and listed in `logs/failed-<stage>-<shard>.jsonl` (re-attempt them
  by launching the workers with `--retry-failed`).

For a **fresh start**, set `RESUME = False` and `FRESH_START_CONFIRM = "FRESH"`.
"""))

DATA_CELLS.append(code("""
import os, sys, json, subprocess, time
from pathlib import Path

# ---- EDIT THESE ----
PPQ_REPO = "/workspace/personaplex-qwen"            # this repo (contains ppq/)
MOSHI_REPO = "/workspace/personaplex"              # original personaplex repo (moshi pkg inside)
MIMI_WEIGHT = "/workspace/models/mimi.safetensors" # moshi mimi codec weights
QWEN_TOKENIZER = "/workspace/models/qwen3-1.7b-base"  # local snapshot of Qwen/Qwen3-1.7B-Base
# Script-generator model: a local snapshot dir OR a HF repo id ("Qwen/Qwen3-8B",
# auto-downloaded by vLLM). On a 32 GB card, "Qwen/Qwen3-4B" also works well.
GEN_MODEL = "/workspace/models/qwen3-8b"
DATA_ROOT = Path("/workspace/ppq_data")
N_GPUS = 2                   # how many GPUs this box has (works with 1)
TARGET_SCRIPTS = 35000   # ~35k x ~80 s ~= 800 h of dialogue
MAX_ATTEMPTS = 3         # per-dialogue retry limit before it is skipped for good

RESUME = True                # False = archive previous outputs and start over
FRESH_START_CONFIRM = ""     # type "FRESH" here when RESUME = False
# --------------------

os.environ["PYTHONPATH"] = f"{MOSHI_REPO}/moshi:{PPQ_REPO}:" + os.environ.get("PYTHONPATH", "")
for p in (f"{MOSHI_REPO}/moshi", PPQ_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

TOKENS_DIR = DATA_ROOT / "tokens"
SCRIPTS = DATA_ROOT / "scripts.jsonl"
LOG_DIR = DATA_ROOT / "logs"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if not RESUME:
    assert FRESH_START_CONFIRM == "FRESH", "type FRESH to confirm a fresh start"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for p in (TOKENS_DIR, SCRIPTS, LOG_DIR):
        if p.exists():
            target = DATA_ROOT / f"archive-{p.name}-{stamp}"
            p.rename(target)
            print("archived", p, "->", target)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

print("data root:", DATA_ROOT, "| logs:", LOG_DIR)
"""))

DATA_CELLS.append(md("""
## 1. Dependencies

kokoro (Apache-2.0) + misaki for TTS, espeak-ng for g2p, and a separate venv
for vLLM (its numpy pin conflicts with moshi).
"""))

DATA_CELLS.append(code("""
WORK = Path("/workspace")
KOKORO_DIR = WORK / "kokoro"
if not KOKORO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/hexgrad/kokoro.git", str(KOKORO_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(KOKORO_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "misaki[en]", "soundfile", "pyloudnorm", "sphn", "safetensors",
    "datasets", "huggingface_hub", "requests"], check=True)
subprocess.run(["bash", "-c",
    "command -v espeak-ng || (apt-get update -qq && apt-get install -y -qq espeak-ng)"],
    check=False)
import kokoro, misaki  # noqa
print("kokoro ok:", kokoro.__file__)
"""))

DATA_CELLS.append(code("""
VENV = Path("/workspace/venvs/vllm_gen")
if not (VENV / "bin" / "python").exists():
    subprocess.run(["python3", "-m", "venv", str(VENV)], check=True)
    subprocess.run([str(VENV / "bin/pip"), "install", "-q", "vllm", "openai"], check=True)
print("vllm venv:", VENV)
"""))

DATA_CELLS.append(md("""
## 2. Dialogue script generation

Starts a vLLM server (on the last GPU) if the scripts file is incomplete,
generates the missing scripts, then the next cell **stops the server again**.
Resumable: re-running appends only the missing scripts; torn final lines from
an interrupted run are repaired automatically.
"""))

DATA_CELLS.append(code("""
import socket

def _script_count() -> int:
    if not SCRIPTS.exists():
        return 0
    with open(SCRIPTS, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def port_open(port: int = 8000) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def resolve_model(spec: str) -> str:
    # Local snapshot dir if it exists, else a HF repo id (vLLM downloads it).
    p = Path(spec)
    if p.exists():
        assert (p / "config.json").exists(), \
            f"{p} has no config.json - not a model snapshot (incomplete download?)"
        return str(p)
    if not spec.startswith("/") and "/" in spec:
        print(f"'{spec}' not found locally - vLLM will download it from Hugging Face")
        return spec
    raise FileNotFoundError(
        f"model not found: {spec}\\n"
        "Either download it first, e.g.:\\n"
        f"  huggingface-cli download Qwen/Qwen3-8B --local-dir {spec}\\n"
        "or set GEN_MODEL = 'Qwen/Qwen3-8B' in the config cell and vLLM will "
        "download it automatically.")

if _script_count() >= TARGET_SCRIPTS:
    print("scripts already complete:", _script_count())
elif port_open(8000):
    print("vLLM server already running")
else:
    model = resolve_model(GEN_MODEL)
    # Last GPU; with a single GPU it shares GPU 0 - fine, nothing else runs
    # during script generation (the server is stopped before the audio workers).
    vllm_gpu = N_GPUS - 1
    srv_log = open(LOG_DIR / "vllm-server.log", "ab", buffering=0)
    proc = subprocess.Popen([
        str(VENV / "bin/python"), "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--served-model-name", "qwen-gen",
        "--tensor-parallel-size", "1", "--max-model-len", "8192",
        # vLLM runs alone in this phase, so give it most of the card; 0.45 of a
        # 32 GB card is too small for an 8B bf16 model (~16 GB of weights).
        "--gpu-memory-utilization", "0.85", "--port", "8000",
    ], stdout=srv_log, stderr=subprocess.STDOUT, start_new_session=True,
       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(vllm_gpu)})
    (LOG_DIR / "vllm-server.pid").write_text(str(proc.pid))
    print("vLLM starting (pid", proc.pid, ") on GPU", vllm_gpu, "- waiting for readiness...")
    for _ in range(360):   # up to 30 min (first run may download the model)
        if port_open(8000):
            break
        if proc.poll() is not None:   # server died - fail fast with the log tail
            tail = "".join((LOG_DIR / "vllm-server.log")
                           .read_text(errors="replace").splitlines(keepends=True)[-30:])
            raise RuntimeError(f"vLLM exited during startup. Log tail:\\n{tail}")
        time.sleep(5)
    assert port_open(8000), "vLLM did not start; see logs/vllm-server.log"
    print("vLLM server up")
"""))

DATA_CELLS.append(code("""
from ppq.progress import setup_file_logger
from ppq.data.script_gen import generate_scripts_openai

existing = _script_count()
if existing >= TARGET_SCRIPTS:
    print("scripts already complete:", existing)
else:
    setup_file_logger("ppq.data.script_gen", LOG_DIR / "script-gen.log", reset=True)
    print(f"{existing} scripts exist; generating {TARGET_SCRIPTS - existing} more")
    print("(% progress + ETA are logged to logs/script-gen.log and printed below)")
    generate_scripts_openai(TARGET_SCRIPTS - existing, SCRIPTS, model="qwen-gen",
                            base_url="http://127.0.0.1:8000/v1", seed=existing)
print("total scripts:", _script_count())
"""))

DATA_CELLS.append(code("""
# Stop the vLLM server - script generation is done, free the GPU.
import signal

pid_file = LOG_DIR / "vllm-server.pid"
stopped = False
if pid_file.exists():
    try:
        os.killpg(int(pid_file.read_text().strip()), signal.SIGTERM)
        stopped = True
    except ProcessLookupError:
        pass
    pid_file.unlink(missing_ok=True)
if not stopped:
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
print("vLLM server stopped" if stopped else "pkill issued for vllm.entrypoints")
"""))

DATA_CELLS.append(code("""
from ppq.data.dialogue import Dialogue

n = n_back = n_int = n_pause = bad = 0
with open(SCRIPTS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = Dialogue.from_dict(json.loads(line))
        except Exception:
            bad += 1
            continue
        n += 1
        n_back += sum(t.backchannel for t in d.turns)
        n_int += sum(t.interrupt for t in d.turns)
        n_pause += sum(t.pause_mid for t in d.turns)
print(f"{n} valid dialogues ({bad} bad lines skipped) | "
      f"backchannels {n_back} | interrupts {n_int} | mid-pauses {n_pause}")
"""))

DATA_CELLS.append(md("""
## 3. TTS + stereo stitching + Mimi tokenization (one worker per GPU)

First build the agent voice prompt (first seconds of a Kokoro voice), then
launch one detached worker per GPU. Each worker:
- processes scripts `i % N_GPUS == shard` and writes `conv-<global index>.npz`
  (names stay stable even if scripts.jsonl grows or the shard count changes);
- appends every finished dialogue to its manifest (the source of truth);
- writes `logs/worker-synth-<shard>.log` and `logs/progress-synth-<shard>.json`.

Re-running the launch cell is safe: shards with a live worker are left alone,
finished shards are skipped by the workers themselves, and a crashed shard
simply resumes where its manifest ends.
"""))

DATA_CELLS.append(code("""
import soundfile as sf
from ppq.data.tts_kokoro import KokoroTTS
from ppq.config import PPQModelConfig

cfg = PPQModelConfig()
VOICE_PROMPT = DATA_ROOT / "voice_prompt_agent.wav"
if not VOICE_PROMPT.exists():
    tts = KokoroTTS()
    res = tts.synth(
        "Hello there! It is lovely to chat with you today. "
        "I am always happy to help with whatever is on your mind, "
        "so feel free to ask me anything at all.", voice="af_heart")
    n = int(cfg.voice_prompt_duration * 24000)
    sf.write(VOICE_PROMPT, res.wav[:n], 24000)
print("voice prompt:", VOICE_PROMPT, f"({cfg.voice_prompt_duration}s)")
"""))

DATA_CELLS.append(code("""
from ppq.progress import pid_alive, pipeline_all_done, render_data_progress

def launch_workers(stage: str, extra_args=()):
    launched, running = [], []
    for shard in range(N_GPUS):
        pid_file = LOG_DIR / f"worker-{stage}-{shard:04d}.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
            except ValueError:
                pid = -1
            if pid > 0 and pid_alive(pid):
                running.append(shard)
                continue
        console = open(LOG_DIR / f"worker-{stage}-{shard:04d}.console.log", "ab", buffering=0)
        cmd = [sys.executable, "-m", "ppq.data.worker",
               "--stage", stage, "--shard", str(shard), "--num-shards", str(N_GPUS),
               "--out", str(TOKENS_DIR), "--log-dir", str(LOG_DIR),
               "--mimi-weight", MIMI_WEIGHT, "--tokenizer", QWEN_TOKENIZER,
               "--max-attempts", str(MAX_ATTEMPTS), "--seed", "0", *extra_args]
        subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)},
                         stdout=console, stderr=subprocess.STDOUT,
                         start_new_session=True)
        launched.append(shard)
    print(f"{stage}: launched shards {launched} | already running {running}")
    if running:
        print("(safe: live workers are never duplicated)")

if pipeline_all_done(LOG_DIR, ("synth",)):
    print("synth stage already complete")
    print(render_data_progress(LOG_DIR, ("synth",)))
else:
    launch_workers("synth", ["--scripts", str(SCRIPTS),
                             "--voice-prompt", str(VOICE_PROMPT)])
"""))

DATA_CELLS.append(code("""
# Live progress table - refreshes until all synth shards finish.
# Interrupt the cell to stop watching (workers keep running); re-run any time.
from ppq.progress import watch
watch(LOG_DIR, ("synth",), interval_s=60)
"""))

DATA_CELLS.append(code("""
# Last lines of each shard log (failures show up here with a full traceback).
for shard in range(N_GPUS):
    p = LOG_DIR / f"worker-synth-{shard:04d}.log"
    if p.exists():
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"--- shard {shard} (last 3 lines) ---")
        print("\\n".join(lines[-3:]))
"""))

DATA_CELLS.append(md("""
## 4. Stage A: LibriTTS-R monologues (~100 h, CC-BY-4.0)

Same worker pattern as the synth stage. The % shown for libritts is an
estimate (the corpus streams until the hour budget is reached); audio hours
in the table are exact.
"""))

DATA_CELLS.append(code("""
if pipeline_all_done(LOG_DIR, ("libritts",)):
    print("libritts stage already complete")
    print(render_data_progress(LOG_DIR, ("libritts",)))
else:
    launch_workers("libritts", ["--max-hours", "100"])
"""))

DATA_CELLS.append(code("""
from ppq.progress import watch
watch(LOG_DIR, ("libritts",), interval_s=60)
"""))

DATA_CELLS.append(md("""
## 5. Aggregate stats
"""))

DATA_CELLS.append(code("""
from ppq.progress import read_jsonl, repair_jsonl

n_synth = n_libritts = 0
total_s = 0.0
bc = ic = mp = words = 0
for m in sorted(TOKENS_DIR.glob("manifest*.jsonl")):
    repair_jsonl(m)
    for e in read_jsonl(m):
        if e.get("source") == "libritts":
            n_libritts += 1
            total_s += e["frames"] / 12.5
        else:
            n_synth += 1
            total_s += e.get("duration_s", e["frames"] / 12.5)
            bc += e.get("backchannels", 0)
            ic += e.get("interrupts", 0)
            mp += e.get("mid_pauses", 0)
            words += e.get("words_estimated", 0)
print(f"synth dialogues     : {n_synth}")
print(f"libritts monologues : {n_libritts}")
print(f"total audio         : {total_s / 3600:.1f} h")
print(f"backchannels {bc} | interrupts {ic} | mid-pauses {mp} | est words {words}")
"""))

DATA_CELLS.append(md("""
## 6. Web client

Build the stock PersonaPlex web client (unchanged - the PPQ server speaks the
same WebSocket protocol).
"""))

DATA_CELLS.append(code("""
CLIENT = Path(MOSHI_REPO) / "client"
if not (CLIENT / "dist").exists():
    subprocess.run(["npm", "install"], cwd=CLIENT, check=True)
    subprocess.run(["npm", "run", "build"], cwd=CLIENT, check=True)
print("client dist:", CLIENT / "dist")
"""))


# ===========================================================================
# Training notebook
# ===========================================================================

TRAIN_CELLS = []

TRAIN_CELLS.append(md("""
# PPQ Training

Run after `ppq_data_pipeline.ipynb` on a multi-GPU box (defaults are tuned
for 2x 32 GB cards, e.g. RTX 5090; scale `N_GPUS`/`batch_size` up on bigger
hardware). Stages:
1. Build the initial checkpoint: Qwen backbone weights + transplanted depformer.
2. Stage A — audio-side warmup (backbone frozen).
3. Stage B — full fine-tuning.
4. Evaluation (per-codebook loss + offline generation).
5. Serving with the stock PersonaPlex web UI.

**Crash-safe resume, on by default.** Each stage has its own directory
(`stage_a/`, `stage_b/`); re-running a launch cell:
- exits immediately when `checkpoints/final` is complete;
- resumes from the newest complete step checkpoint (model + optimizer +
  scheduler + RNG) otherwise — a crash costs at most `save_every_minutes` of work;
- never starts a second copy while a run is still alive.

Training logs (append-only) are in `<stage>/logs/train.log`; live
step/%/ETA status is in `<stage>/progress.json`, rendered by the watch cells.
To restart a stage from scratch, delete its `checkpoints/` directory.

> Note: after a resume the data order is reshuffled (reseeded), so resumed
> runs are equivalent, not bit-identical. Optimizer/scheduler/RNG states are
> restored exactly.
"""))

TRAIN_CELLS.append(code("""
import os, sys, json, subprocess
from pathlib import Path

# ---- EDIT THESE ----
PPQ_REPO = "/workspace/personaplex-qwen"
MOSHI_REPO = "/workspace/personaplex"
MIMI_WEIGHT = "/workspace/models/mimi.safetensors"
QWEN_BASE = "/workspace/models/qwen3-1.7b-base"        # local Qwen3-1.7B-Base snapshot
PERSONAPLEX_CKPT = "/workspace/models/personaplex/model.sft"  # personaplex LM (depformer donor)
DATA_ROOT = Path("/workspace/ppq_data")
TOKENS_DIR = DATA_ROOT / "tokens"
RUN_DIR = Path("/workspace/ppq_runs/run1")
CLIENT_DIST = Path(MOSHI_REPO) / "client" / "dist"
N_GPUS = 2                   # works with 1; increase on bigger boxes
# --------------------

os.environ["PYTHONPATH"] = f"{MOSHI_REPO}/moshi:{PPQ_REPO}:" + os.environ.get("PYTHONPATH", "")
for p in (f"{MOSHI_REPO}/moshi", PPQ_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

STAGE_A_DIR = RUN_DIR / "stage_a"
STAGE_B_DIR = RUN_DIR / "stage_b"
RUN_DIR.mkdir(parents=True, exist_ok=True)
"""))

TRAIN_CELLS.append(code("""
# Where are we? (safe to re-run any time)
from ppq.progress import training_status
print(training_status(RUN_DIR))
"""))

TRAIN_CELLS.append(md("""
## 1. Initial checkpoint
Qwen3-1.7B-Base backbone + transplanted PersonaPlex depformer; fresh audio
embeddings, depformer_in, depformer_text_emb (trained in stage A).
"""))

TRAIN_CELLS.append(code("""
import torch
from ppq.config import PPQModelConfig
from ppq.checkpoint import build_model_from_scratch, is_complete, save_pretrained

INIT_DIR = RUN_DIR / "init"
if is_complete(INIT_DIR):
    print("init checkpoint already complete:", INIT_DIR)
else:
    cfg = PPQModelConfig()
    model = build_model_from_scratch(cfg, qwen_dir=QWEN_BASE,
                                     depformer_source=PERSONAPLEX_CKPT)
    save_pretrained(model, INIT_DIR)   # atomic + COMPLETE marker
    del model
    print("init checkpoint saved:", INIT_DIR)
"""))

TRAIN_CELLS.append(md("""
## 2. Stage A — audio-side warmup
Backbone layers, final norm and text head frozen; everything else trains.
"""))

TRAIN_CELLS.append(code("""
from ppq.config import PPQModelConfig, TrainConfig

STAGE_A_DIR.mkdir(parents=True, exist_ok=True)
tcfg_a = TrainConfig(
    model=PPQModelConfig(),
    data_dir=str(TOKENS_DIR),
    output_dir=str(STAGE_A_DIR),
    init_checkpoint=str(RUN_DIR / "init"),
    stage="a_audio_warmup",
    # 32 GB cards: batch_size=2 per GPU. On 80 GB cards use batch_size=4,
    # grad_accum=1. Effective batch = batch_size x grad_accum x N_GPUS.
    batch_size=2, grad_accum=2,
    max_steps=400,               # ~= 400 x (4 x N_GPUS) x 100 s of audio seen
    save_every=200, save_every_minutes=30, keep_last_k=3,
    log_every=10,
    fsdp=False,                  # DDP; resume of optimizer state needs fsdp=False
    gradient_checkpointing=False,
)
tcfg_a.to_json(STAGE_A_DIR / "config.json")
print(STAGE_A_DIR / "config.json")
"""))

TRAIN_CELLS.append(code("""
from ppq.checkpoint import is_complete
from ppq.progress import pid_alive

def launch_stage(stage_dir: Path):
    logs = stage_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pid_file = logs / "torchrun.pid"
    if is_complete(stage_dir / "checkpoints" / "final"):
        print(stage_dir.name, "already finished (final checkpoint complete)")
        return
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and pid_alive(pid):
            print(stage_dir.name, "is already running - just watch it below")
            return
    console = open(logs / "console.log", "ab", buffering=0)
    proc = subprocess.Popen([
        "torchrun", "--nproc_per_node", str(N_GPUS), "--max-restarts", "3",
        "-m", "ppq.train.train", "--config", str(stage_dir / "config.json"),
    ], stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
    pid_file.write_text(str(proc.pid))
    print(stage_dir.name, "launched (pid", proc.pid,
          ") - auto-resume picks up from the last checkpoint")

launch_stage(STAGE_A_DIR)
"""))

TRAIN_CELLS.append(code("""
# Live training progress - refreshes until the stage finishes.
# Interrupt the cell to stop watching (training keeps running); re-run any time.
from ppq.progress import watch_train
watch_train(STAGE_A_DIR, interval_s=60)
"""))

TRAIN_CELLS.append(md("""
## 3. Stage B — full fine-tuning
All weights trainable (backbone at 0.5x LR, depformer at 1.5x LR), starting
from Stage A's final checkpoint. Auto-resumes from its own checkpoints.
"""))

TRAIN_CELLS.append(code("""
assert is_complete(STAGE_A_DIR / "checkpoints" / "final"), "finish Stage A first"

STAGE_B_DIR.mkdir(parents=True, exist_ok=True)
tcfg_b = TrainConfig(
    model=PPQModelConfig(),
    data_dir=str(TOKENS_DIR),
    output_dir=str(STAGE_B_DIR),
    init_checkpoint=str(STAGE_A_DIR / "checkpoints" / "final"),
    stage="b_full",
    # 32 GB cards: batch_size=2 + grad checkpointing. 80 GB cards: batch_size=4.
    batch_size=2, grad_accum=2,
    max_steps=1000,
    lr=5e-6, backbone_lr_mult=0.5, depformer_lr_mult=1.5,
    lora_rank=64,
    save_every=250, save_every_minutes=30, keep_last_k=3,
    log_every=10,
    fsdp=False,
    gradient_checkpointing=True,
)
tcfg_b.to_json(STAGE_B_DIR / "config.json")
print(STAGE_B_DIR / "config.json")
"""))

TRAIN_CELLS.append(code("""
launch_stage(STAGE_B_DIR)
"""))

TRAIN_CELLS.append(code("""
from ppq.progress import watch_train
watch_train(STAGE_B_DIR, interval_s=120)
"""))

TRAIN_CELLS.append(md("""
## 4. Evaluation
Per-codebook cross-entropy on held-out segments, then an offline full-duplex
generation against a recorded user utterance (or synthesized one).
"""))

TRAIN_CELLS.append(code("""
import glob, torch
from ppq.checkpoint import load_pretrained
from ppq.data.dataset import TokenShardDataset, collate
from ppq.train.evaluate import eval_per_codebook

FINAL_B = STAGE_B_DIR / "checkpoints" / "final"
assert is_complete(FINAL_B), "finish Stage B first"

device = "cuda"
model = load_pretrained(FINAL_B, device=device)
manifests = sorted(glob.glob(str(TOKENS_DIR / "manifest*.jsonl")))[:1]
ds = TokenShardDataset(manifests, model.cfg, segment_frames=1250, seed=999)
loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate, drop_last=True)
losses = eval_per_codebook(model, loader, n_batches=8, device=device, mcfg=model.cfg)
print({k: round(v, 3) for k, v in losses.items()})
del model
torch.cuda.empty_cache()
"""))

TRAIN_CELLS.append(code("""
import numpy as np
from ppq.checkpoint import get_mimi, load_pretrained
from ppq.text_tokenizer import QwenTextTokenizer
from ppq.train.evaluate import generate_offline, write_wav, response_delays

device = "cuda"
model = load_pretrained(FINAL_B, device=device)
mimi = get_mimi(MIMI_WEIGHT, device)
tokenizer = QwenTextTokenizer(QWEN_BASE, model.cfg)

USER_WAV = DATA_ROOT / "test_user.wav"   # record your own 24 kHz wav, or synthesize:
if not USER_WAV.exists():
    from ppq.data.tts_kokoro import KokoroTTS
    import soundfile as sf
    res = KokoroTTS().synth(
        "Hey! Tell me a fun fact about the ocean in one or two sentences.",
        voice="am_adam")
    sf.write(USER_WAV, res.wav, 24000)

import sphn
user_wav, sr = sphn.read(str(USER_WAV))
if sr != 24000:
    user_wav = sphn.resample(user_wav, sr, 24000)
vp_wav, vp_sr = sphn.read(str(DATA_ROOT / "voice_prompt_agent.wav"))
if vp_sr != 24000:
    vp_wav = sphn.resample(vp_wav, vp_sr, 24000)

agent_wav, transcript = generate_offline(
    model, mimi, tokenizer, user_wav[0].astype(np.float32),
    vp_wav[0].astype(np.float32),
    "You are a cheerful, helpful assistant. You speak naturally.",
    device=device, max_seconds=30.0)
write_wav(str(RUN_DIR / "agent_reply.wav"), agent_wav)
print("transcript:", transcript)
print("response delays (s):", response_delays(agent_wav, user_wav[0]))
"""))

TRAIN_CELLS.append(md("""
## 5. Serving

Stock PersonaPlex WebSocket protocol + the stock web client. Point your
reverse proxy / browser at the printed port.
"""))

TRAIN_CELLS.append(code("""
cmd = [
    sys.executable, "-m", "ppq.serve.server",
    "--ppq-checkpoint", str(STAGE_B_DIR / "checkpoints" / "final"),
    "--mimi-weight", MIMI_WEIGHT,
    "--tokenizer", QWEN_BASE,
    "--voice-prompt-dir", str(DATA_ROOT),   # dir containing voice_prompt_agent.wav
    "--default-voice", "voice_prompt_agent.wav",
    "--static", str(CLIENT_DIST),
    "--port", "8998",
]
print(" ".join(cmd))
# subprocess.Popen(cmd)   # uncomment to launch
"""))


# ===========================================================================

NB_DIR.mkdir(parents=True, exist_ok=True)
for name, cells in (("ppq_data_pipeline.ipynb", DATA_CELLS),
                    ("ppq_training.ipynb", TRAIN_CELLS)):
    path = NB_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook(cells), f, indent=1, ensure_ascii=False)
    print("wrote", path, f"({len(cells)} cells)")

