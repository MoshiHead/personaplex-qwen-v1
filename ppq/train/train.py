"""PPQ training entry point.

Single GPU:
  python -m ppq.train.train --config train_config.json
Multi-GPU (DDP, or FSDP2 with "fsdp": true):
  torchrun --nproc_per_node=8 --max-restarts 3 -m ppq.train.train --config train_config.json

Run directory layout (one dir per stage, e.g. <run>/stage_a):
  <stage_dir>/init/                       initial checkpoint (or set "init_checkpoint")
  <stage_dir>/checkpoints/step-NNNNNN/    periodic PPQ checkpoints (COMPLETE marker)
  <stage_dir>/checkpoints/LAST            pointer file to the newest complete step checkpoint
  <stage_dir>/checkpoints/final/          written when training completes
  <stage_dir>/logs/train.log              append-only training log
  <stage_dir>/progress.json               step/percent/ETA status for the notebook

Auto-resume (default): if a complete step checkpoint exists, training resumes
from it (model, optimizer, scheduler, RNG states). If checkpoints/final is
complete, training exits immediately. To restart a stage from scratch, delete
its checkpoints/ directory (and progress.json).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..checkpoint import is_complete, load_pretrained, save_pretrained
from ..config import TrainConfig
from ..data.dataset import TokenShardDataset, collate
from ..progress import atomic_write_json, fmt_duration, session_header, setup_file_logger
from .loss import compute_losses


def setup_distributed() -> tuple[int, int, int, bool]:
    """Returns (rank, world_size, local_rank, is_distributed)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ["LOCAL_RANK"])
        # More ranks than physical GPUs is fine (ranks share GPUs); only the
        # device ordinal must stay in range. Return the device ordinal as
        # local_rank so downstream device placement uses it too.
        n_dev = max(1, torch.cuda.device_count())
        local = local % n_dev
        torch.cuda.set_device(local)
        torch.distributed.init_process_group(backend="nccl")
        return rank, world, local, world > 1
    return 0, 1, 0, False


def freeze_for_stage(model, stage: str, logger):
    if stage == "a_audio_warmup":
        frozen, trainable = 0, 0
        for name, p in model.named_parameters():
            if name.startswith(("backbone.layers", "backbone.norm", "text_linear")):
                p.requires_grad = False
                frozen += p.numel()
            else:
                trainable += p.numel()
        logger.info("Stage A: froze %.1fM backbone params, training %.1fM params",
                    frozen / 1e6, trainable / 1e6)
    elif stage != "b_full":
        raise ValueError(f"unknown stage: {stage}")


def maybe_apply_lora(model, tcfg: TrainConfig):
    if tcfg.lora_rank <= 0:
        return model
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(
        r=tcfg.lora_rank, lora_alpha=tcfg.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def make_optimizer(model, tcfg: TrainConfig):
    groups = {"backbone": [], "depformer": [], "other": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("backbone.", "text_emb.", "text_linear.")):
            groups["backbone"].append(p)
        elif name.startswith(("depformer", "linears.")):
            groups["depformer"].append(p)
        else:
            groups["other"].append(p)
    return torch.optim.AdamW([
        {"params": groups["backbone"], "lr": tcfg.lr * tcfg.backbone_lr_mult},
        {"params": groups["depformer"], "lr": tcfg.lr * tcfg.depformer_lr_mult},
        {"params": groups["other"], "lr": tcfg.lr},
    ], weight_decay=tcfg.weight_decay, betas=(0.9, 0.95), eps=1e-8)


def lr_lambda(step: int, tcfg: TrainConfig) -> float:
    if step < tcfg.warmup_steps:
        return (step + 1) / tcfg.warmup_steps
    progress = (step - tcfg.warmup_steps) / max(1, tcfg.max_steps - tcfg.warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------
# Checkpoint save / resume helpers
# ---------------------------------------------------------------------------

def _torch_save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_training_checkpoint(raw_model, optimizer, scheduler, step: int,
                             out_dir: Path, logger) -> Path:
    """Save model (atomic, COMPLETE marker) + optimizer + scheduler + RNG
    states into checkpoints/step-NNNNNN, then update the LAST pointer and
    prune old step checkpoints."""
    ckpt_dir = out_dir / "checkpoints" / f"step-{step:06d}"
    save_pretrained(raw_model, ckpt_dir)  # model only; atomic
    _torch_save_atomic(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    _torch_save_atomic(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    state = {
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
        "np_rng": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    _torch_save_atomic(state, ckpt_dir / "train_state.pt")
    # LAST pointer (cheap text file, avoids copying the checkpoint).
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "checkpoints" / "LAST.tmp"
    tmp.write_text(ckpt_dir.name, encoding="utf-8")
    os.replace(tmp, out_dir / "checkpoints" / "LAST")
    logger.info("saved checkpoint %s", ckpt_dir)
    return ckpt_dir


def prune_checkpoints(out_dir: Path, keep_last_k: int, logger) -> None:
    steps = sorted((out_dir / "checkpoints").glob("step-*"))
    excess = steps[:-keep_last_k] if keep_last_k > 0 else []
    for d in excess:
        logger.info("pruning old checkpoint %s", d.name)
        shutil.rmtree(d, ignore_errors=True)


def resolve_start(out_dir: Path, tcfg: TrainConfig, logger):
    """Returns (mode, init_dir, resume_state). mode in finished/resume/fresh."""
    final_dir = out_dir / "checkpoints" / "final"
    if is_complete(final_dir):
        logger.info("checkpoints/final is complete -> stage already finished, exiting.")
        return "finished", final_dir, None
    last_ptr = out_dir / "checkpoints" / "LAST"
    if tcfg.auto_resume and last_ptr.exists():
        ckpt_dir = out_dir / "checkpoints" / last_ptr.read_text(encoding="utf-8").strip()
        state_path = ckpt_dir / "train_state.pt"
        if is_complete(ckpt_dir) and state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            logger.info("auto-resuming from %s at step %d", ckpt_dir, state["step"])
            return "resume", ckpt_dir, state
        logger.warning("LAST points to incomplete checkpoint %s; starting fresh", ckpt_dir)
    init_dir = Path(tcfg.init_checkpoint) if tcfg.init_checkpoint else out_dir / "init"
    logger.info("starting fresh from init checkpoint %s", init_dir)
    return "fresh", init_dir, None


def write_progress(stage_dir: Path, tcfg: TrainConfig, step: int, status: str,
                   losses: dict, lr: float, sec_per_step: float,
                   audio_seconds_seen: float, started_at: float) -> None:
    remaining = max(0, tcfg.max_steps - step)
    eta = remaining * sec_per_step if sec_per_step > 0 else None
    atomic_write_json(stage_dir / "progress.json", {
        "stage": tcfg.stage,
        "status": status,
        "step": step,
        "max_steps": tcfg.max_steps,
        "percent": round(100.0 * step / max(1, tcfg.max_steps), 2),
        "remaining_steps": remaining,
        "sec_per_step": round(sec_per_step, 3),
        "eta_s": None if eta is None else round(eta),
        "eta": fmt_duration(eta),
        "total_loss": round(losses.get("total", float("nan")), 4),
        "losses": {k: round(v, 4) for k, v in losses.items()},
        "lr": lr,
        "audio_hours_seen": round(audio_seconds_seen / 3600.0, 2),
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def main(config_path: str | None = None):
    if config_path is None:
        p = argparse.ArgumentParser()
        p.add_argument("--config", required=True)
        config_path = p.parse_args().config
    tcfg = TrainConfig.from_json(config_path)
    rank, world, local_rank, distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tcfg.seed + rank)

    out_dir = Path(tcfg.output_dir)
    log_path = out_dir / "logs" / "train.log"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        session_header(log_path, f"train stage={tcfg.stage} world={world}")
        logger = setup_file_logger("ppq.train", log_path, reset=True)
    else:
        import logging
        logger = logging.getLogger(f"ppq.train.rank{rank}")
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.WARNING)
    logger.info("config %s | stage %s | world %d | device %s",
                config_path, tcfg.stage, world, device)

    # --- resolve start point (fresh / resume / already finished) ---
    mode, init_dir, resume_state = resolve_start(out_dir, tcfg, logger)
    if mode == "finished":
        write_progress(out_dir, tcfg, tcfg.max_steps, "finished", {}, 0.0, 0.0,
                       0.0, time.time())
        if distributed:
            torch.distributed.destroy_process_group()
        return
    if mode == "resume" and tcfg.fsdp:
        raise NotImplementedError("optimizer-state resume is not supported with fsdp=true")
    start_step = int(resume_state["step"]) if resume_state is not None else 0
    if start_step >= tcfg.max_steps:
        logger.info("checkpoint already at step %d >= max_steps %d; saving final.",
                    start_step, tcfg.max_steps)

    # --- data (seeded by resume position so resume sees a fresh data order) ---
    manifests = sorted(glob.glob(str(Path(tcfg.data_dir) / "manifest*.jsonl")))
    assert manifests, f"no manifests found in {tcfg.data_dir}"
    dataset = TokenShardDataset(manifests, tcfg.model, segment_frames=tcfg.segment_frames,
                                seed=tcfg.seed + start_step + rank,
                                source_weights=tcfg.source_weights or None)
    if tcfg.source_weights:
        counts: dict[str, int] = {}
        for e in dataset.entries:
            counts[e.get("source", "")] = counts.get(e.get("source", ""), 0) + 1
        logger.info("source_weights=%s | entries per source=%s",
                    tcfg.source_weights, counts)
    sampler = None
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True,
            seed=tcfg.seed + start_step)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=tcfg.batch_size, sampler=sampler,
        shuffle=(sampler is None and tcfg.shuffle), num_workers=tcfg.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True)

    # --- model ---
    # With LoRA the saved state dict has peft "base_model.model." prefixes, so
    # on resume we load the base weights from the ORIGINAL init checkpoint,
    # re-apply LoRA, then overlay the full saved state dict.
    load_from = init_dir
    lora_overlay = None
    if mode == "resume" and tcfg.lora_rank > 0:
        load_from = Path(tcfg.init_checkpoint) if tcfg.init_checkpoint else out_dir / "init"
        lora_overlay = Path(init_dir) / "model.safetensors"
    logger.info("loading initial model from %s", load_from)
    # Load in bf16 (the native checkpoint dtype). fp32 "master weights" double
    # params+grads and, with fp32 Adam states on ~1.9B trainable params, need
    # ~37 GB - can never fit a 32 GB card. Full-bf16 (params+grads+states,
    # fp32 loss upcast in compute_losses) needs ~18 GB static.
    model = load_pretrained(load_from, device="cpu", dtype=torch.bfloat16)
    model.cfg = tcfg.model  # pick up any config overrides
    freeze_for_stage(model, tcfg.stage, logger)
    model = maybe_apply_lora(model, tcfg)
    if lora_overlay is not None:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(lora_overlay)), strict=True)
        logger.info("overlaid LoRA resume weights from %s", lora_overlay)
    model.backbone.gradient_checkpointing = tcfg.gradient_checkpointing
    model.to(device)
    model.train()

    if distributed and tcfg.fsdp:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.backbone.layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model.backbone, mp_policy=mp)
        fully_shard(model)
    elif distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if hasattr(model, "module") else model

    optimizer = make_optimizer(raw_model, tcfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, tcfg))

    if resume_state is not None:
        ckpt_dir = Path(init_dir)
        optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt",
                                             map_location="cpu", weights_only=False))
        scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt",
                                             map_location="cpu", weights_only=False))
        torch.set_rng_state(resume_state["torch_rng"])
        random.setstate(resume_state["py_rng"])
        np.random.set_state(resume_state["np_rng"])
        if "cuda_rng" in resume_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        logger.info("restored optimizer/scheduler/RNG states (step %d)", start_step)

    # --- loop ---
    data_iter = iter(loader)
    t0 = time.time()
    last_save_t = t0
    step_times: collections.deque = collections.deque(maxlen=200)
    audio_seconds_seen = 0.0
    last_losses: dict = {}
    if rank == 0 and start_step > 0:
        write_progress(out_dir, tcfg, start_step, "running", {}, 0.0, 0.0, 0.0, t0)
    for step in range(start_step, tcfg.max_steps):
        step_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        accum: dict[str, float] = {}
        for _ in range(tcfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            codes = batch["codes"].to(device, non_blocking=True)
            mask = batch["loss_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                lm_out = model.forward_train(codes)
                losses = compute_losses(lm_out, codes, mask, tcfg, raw_model.cfg)
                loss = losses["total"] / tcfg.grad_accum
            loss.backward()
            audio_seconds_seen += float(codes.shape[1]) * float(codes.shape[0]) / 12.5
            for k, v in losses.items():
                accum[k] = accum.get(k, 0.0) + float(v.detach()) / tcfg.grad_accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in raw_model.parameters() if p.requires_grad], tcfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step_times.append(time.time() - step_t0)
        last_losses = accum

        if rank == 0 and (step + 1) % tcfg.log_every == 0:
            elapsed = time.time() - t0
            sec_per_step = (sum(step_times) / len(step_times)) if step_times else 0.0
            remaining = tcfg.max_steps - (step + 1)
            eta = fmt_duration(remaining * sec_per_step if sec_per_step else None)
            msg = " ".join(f"{k}={v:.3f}" for k, v in sorted(accum.items()) if k != "total")
            logger.info(
                "step %d/%d (%.1f%%) total=%.3f %s lr=%.2e | %.2fs/step | remaining %d "
                "steps | ETA %s | audio seen %.1f h | elapsed %.0fs",
                step + 1, tcfg.max_steps, 100.0 * (step + 1) / tcfg.max_steps,
                accum["total"], msg, scheduler.get_last_lr()[0], sec_per_step,
                remaining, eta, audio_seconds_seen / 3600.0, elapsed)
            write_progress(out_dir, tcfg, step + 1, "running", accum,
                           scheduler.get_last_lr()[0], sec_per_step,
                           audio_seconds_seen, t0)

        time_due = (tcfg.save_every_minutes > 0
                    and time.time() - last_save_t > tcfg.save_every_minutes * 60)
        if rank == 0 and ((step + 1) % tcfg.save_every == 0 or time_due):
            save_training_checkpoint(raw_model, optimizer, scheduler, step + 1,
                                     out_dir, logger)
            prune_checkpoints(out_dir, tcfg.keep_last_k, logger)
            last_save_t = time.time()

        if tcfg.debug_stop_after_steps and (step + 1) == tcfg.debug_stop_after_steps:
            logger.warning("debug_stop_after_steps=%d reached; crashing on purpose",
                           tcfg.debug_stop_after_steps)
            write_progress(out_dir, tcfg, step + 1, "failed", accum,
                           scheduler.get_last_lr()[0],
                           (sum(step_times) / len(step_times)) if step_times else 0.0,
                           audio_seconds_seen, t0)
            raise RuntimeError("debug_stop_after_steps crash simulation")

    if rank == 0:
        save_pretrained(raw_model, out_dir / "checkpoints" / "final")
        sec_per_step = (sum(step_times) / len(step_times)) if step_times else 0.0
        write_progress(out_dir, tcfg, tcfg.max_steps, "finished", last_losses,
                       scheduler.get_last_lr()[0], sec_per_step,
                       audio_seconds_seen, t0)
        logger.info("training finished in %.0fs (final checkpoint saved)",
                    time.time() - t0)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
