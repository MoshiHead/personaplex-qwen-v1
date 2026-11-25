# PersonaPlex-Qwen (PPQ)

An Apache-2.0 full-duplex voice agent: the PersonaPlex architecture with the
Helium backbone replaced by **Qwen3-1.7B-Base** (Apache-2.0).

## Architecture

- **Backbone**: Qwen3-1.7B-Base (24 layers, dim 2048, GQA 16Q/8KV heads,
  head_dim 128, RoPE theta 1M, QK-norm) — HF-compatible, loading verified
  bit-exactly against `transformers.Qwen3ForCausalLM` in `tests/test_qwen_equivalence.py`.
- **Audio path** (identical to PersonaPlex): 17 streams = 1 text stream
  (Qwen tokenizer, PAD/EPAD controls) + 8 agent Mimi codebooks + 8 user Mimi
  codebooks, per-stream delays `[0, 0, 1..1, 0, 1..1]`, and a depth transformer
  (6 layers, dim 1024, one input projection/embedding/head per 80 ms step).
- **Mimi codec** (24 kHz, 12.5 Hz frames, 8 codebooks, card 2048): unchanged,
  loaded from the original moshi weights.
- **Moshi-faithful inference**: `PPQLMGen` reuses moshi's `LMGen` streaming
  state machine (delays, initial tokens, system prompts) unchanged, so
  PersonaPlex checkpoint behavior carries over. Verified against the stock
  moshi `LMModel + LMGen` in the tests.

## Layout

```
ppq/            model, lmgen, tokenizer, checkpoint, transplant
ppq/data/       script gen, Kokoro TTS, stereo stitch, mimi tokenize, dataset
ppq/train/      loss, train loop (DDP / FSDP2), evaluation
ppq/serve/      WebSocket server (stock PersonaPlex protocol) + offline CLI
ppq/progress.py progress tracking, ETA, append-mode logging, PID locks, JSONL repair
tests/          equivalence + streaming + data + transplant + training-step + resume tests
notebooks/      ppq_data_pipeline.ipynb, ppq_training.ipynb
```

## Quick start (on the DGX)

```bash
pip install -r requirements.txt
export PYTHONPATH=/workspace/personaplex/moshi:/workspace/personaplex-qwen:$PYTHONPATH
```

1. **Data**: run `notebooks/ppq_data_pipeline.ipynb` — generates dialogue
   scripts with an instruct LLM, synthesizes with Kokoro, stitches full-duplex
   stereo conversations (backchannels / interrupts / mid-turn pauses), and
   pre-tokenizes with Mimi. One worker per GPU, fully resumable.
2. **Training**: run `notebooks/ppq_training.ipynb` — builds the initial
   checkpoint (Qwen backbone + transplanted depformer), then Stage A
   (audio-side warmup, backbone frozen) and Stage B (full fine-tune).
3. **Serving**: `python -m ppq.serve.server --ppq-checkpoint <ckpt>
   --mimi-weight <mimi.safetensors> --tokenizer <qwen-tokenizer>
   --voice-prompt-dir voices/ --default-voice af_heart.wav --static client/dist`
   and open the stock PersonaPlex web UI.

## Progress, logging and crash-safe resume

Both notebooks are built for long, interruptible runs:

- **Data pipeline**: the manifest is the source of truth. Samples are named by
  stable global indices (`conv-0000123.npz`), written atomically (temp file +
  rename), and each dialogue's stitch RNG is seeded by its index, so resuming
  after a crash reproduces byte-identical audio. Torn manifests/JSONL are
  repaired automatically; failing dialogues are retried up to `--max-attempts`
  (tracked in `logs/failed-*.jsonl`, re-arm with `--retry-failed`). Per-shard
  logs and `progress-*.json` files (done/total/%/ETA) live in
  `DATA_ROOT/logs/`; notebook *watch* cells render a live table, and stale
  workers are flagged as `STALLED?`.
- **Training**: each stage writes to its own directory. Checkpoints are atomic
  (`COMPLETE` markers) and bundle model + optimizer + scheduler + RNG states;
  a `LAST` pointer drives auto-resume (on by default), `keep_last_k` prunes
  old checkpoints, and `save_every_minutes` bounds lost work. A finished stage
  (`checkpoints/final`) exits immediately when relaunched. Launch with
  `torchrun --max-restarts 3` so node hiccups self-heal. After a resume the
  data order is reshuffled (reseeded), so resumed runs are equivalent rather
  than bit-identical.

## Tests

```bash
pytest tests/              # 29 tests, CPU-friendly
```

Tests without GPU/HF deps cover: Qwen backbone construction vs HF config,
training/streaming logit alignment (delay-aware) vs the stock moshi reference,
depformer transplant from a PersonaPlex-style checkpoint (with 8->16 per-step
expansion), the text interleaver, the stereo stitcher, the dataset, and a toy
training step. The HF-equivalence test (`test_qwen_equivalence.py`) requires
downloading `Qwen/Qwen3-1.7B-Base` and runs on GPU.

## Licenses

- Qwen3-1.7B-Base: Apache-2.0
- Mimi codec weights (moshi): CC-BY-4.0
- Kokoro-82M TTS: Apache-2.0
- LibriTTS-R: CC-BY-4.0
- PersonaPlex depformer weights: NVIDIA license — check terms before
  redistributing the transplanted depformer weights; everything else here is
  Apache-2.0 code.
