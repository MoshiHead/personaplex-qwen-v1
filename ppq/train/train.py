"""PPQ training entry point.

Single GPU:
  python -m ppq.train.train --config train_config.json
Multi-GPU (DDP, or FSDP2 with "fsdp": true):
  torchrun --nproc_per_node=8 -m ppq.train.train --config train_config.json

Expected run directory layout:
  <output_dir>/init/            initial checkpoint (from build_model_from_scratch)
  <output_dir>/checkpoints/step-NNNNNN/   periodic PPQ checkpoints
  <output_dir>/optimizer.pt     optimizer + step state for resume
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import time
from pathlib import Path

import torch

from ..checkpoint import load_pretrained, save_pretrained
from ..config import TrainConfig
from ..data.dataset import TokenShardDataset, collate
from .loss import compute_losses

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ppq.train")


def setup_distributed() -> tuple[int, int, int, bool]:
    """Returns (rank, world_size, local_rank, is_distributed)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        torch.distributed.init_process_group(backend="nccl")
        return rank, world, local, world > 1
    return 0, 1, 0, False


def freeze_for_stage(model, stage: str):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    tcfg = TrainConfig.from_json(args.config)
    rank, world, local_rank, distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tcfg.seed + rank)

    out_dir = Path(tcfg.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    manifests = sorted(glob.glob(str(Path(tcfg.data_dir) / "manifest*.jsonl")))
    assert manifests, f"no manifests found in {tcfg.data_dir}"
    dataset = TokenShardDataset(manifests, tcfg.model, segment_frames=tcfg.segment_frames,
                                seed=tcfg.seed + rank)
    sampler = None
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world, rank=rank,
                                                      shuffle=True, seed=tcfg.seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=tcfg.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=4, collate_fn=collate,
        pin_memory=True, drop_last=True)

    # --- model ---
    init_dir = Path(tcfg.resume) / "checkpoints" / "last" if tcfg.resume else out_dir / "init"
    logger.info("loading initial model from %s", init_dir)
    model = load_pretrained(init_dir, device="cpu", dtype=torch.float32)
    model.cfg = tcfg.model  # pick up any config overrides
    freeze_for_stage(model, tcfg.stage)
    model = maybe_apply_lora(model, tcfg)
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

    start_step = 0
    opt_path = Path(tcfg.resume) / "optimizer.pt" if tcfg.resume else None
    if opt_path is not None and opt_path.exists():
        state = torch.load(opt_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        logger.info("resumed optimizer state at step %d", start_step)

    # --- loop ---
    data_iter = iter(loader)
    t0 = time.time()
    for step in range(start_step, tcfg.max_steps):
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lm_out = model.forward_train(codes)
                losses = compute_losses(lm_out, codes, mask, tcfg, raw_model.cfg)
                loss = losses["total"] / tcfg.grad_accum
            loss.backward()
            for k, v in losses.items():
                accum[k] = accum.get(k, 0.0) + float(v.detach()) / tcfg.grad_accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in raw_model.parameters() if p.requires_grad], tcfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if rank == 0 and (step + 1) % tcfg.log_every == 0:
            elapsed = time.time() - t0
            msg = " ".join(f"{k}={v:.3f}" for k, v in sorted(accum.items()) if k != "total")
            logger.info("step %d/%d total=%.3f %s lr=%.2e elapsed=%.0fs",
                        step + 1, tcfg.max_steps, accum["total"], msg,
                        scheduler.get_last_lr()[0], elapsed)
        if rank == 0 and (step + 1) % tcfg.save_every == 0:
            ckpt_dir = out_dir / "checkpoints" / f"step-{step + 1:06d}"
            save_pretrained(raw_model, ckpt_dir)
            last_dir = out_dir / "checkpoints" / "last"
            save_pretrained(raw_model, last_dir)
            torch.save({"optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step + 1},
                       out_dir / "optimizer.pt")
            logger.info("saved checkpoint %s", ckpt_dir)

    if rank == 0:
        save_pretrained(raw_model, out_dir / "checkpoints" / "final")
        logger.info("training finished in %.0fs", time.time() - t0)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

