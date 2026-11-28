"""Training auto-resume: 4 steps + resumed 4 steps must equal a fresh 8-step
run bit-for-bit (deterministic tiny CPU setup), and a finished stage must
exit immediately without touching the final checkpoint.
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from ppq.checkpoint import is_complete, save_pretrained
from ppq.config import TrainConfig
from ppq.train import train

from test_ppq_model import tiny_model


def _write_tokens(data_dir: Path, cfg, n: int = 2, T: int = 60) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    g = np.random.RandomState(0)
    entries = []
    for i in range(n):
        codes = np.zeros((17, T), dtype=np.int32)
        codes[0] = g.randint(0, cfg.text_card - 3, T)      # avoid pad/epad ids
        codes[1:] = g.randint(0, 2048, (16, T))
        mask = np.ones((17, T), dtype=np.float16)
        path = data_dir / f"s{i}.npz"
        np.savez_compressed(path, codes=codes, loss_mask=mask)
        entries.append({"path": str(path), "frames": T})
    with open(data_dir / "manifest-0000.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _config(tmp_path: Path, cfg, out_name: str, max_steps: int,
            crash_after: int = 0, lora_rank: int = 0) -> Path:
    tcfg = TrainConfig(
        model=cfg,
        data_dir=str(tmp_path / "tokens"),
        output_dir=str(tmp_path / out_name),
        init_checkpoint=str(tmp_path / "init"),
        stage="b_full",
        batch_size=1, grad_accum=1, segment_frames=60,
        max_steps=max_steps, lr=1e-4, warmup_steps=2,
        save_every=4, log_every=1, seed=7,
        num_workers=0, gradient_checkpointing=False,
        save_every_minutes=0.0,
        shuffle=False,  # deterministic sample order for the bit-exact comparison
        debug_stop_after_steps=crash_after,
        lora_rank=lora_rank,
    )
    path = tmp_path / f"{out_name}.json"
    tcfg.to_json(path)
    return path


def _final_tensors(run_dir: Path) -> dict:
    return load_file(str(run_dir / "checkpoints" / "final" / "model.safetensors"))


def test_resume_matches_fresh_run(tmp_path):
    model = tiny_model()
    save_pretrained(model, tmp_path / "init")
    assert is_complete(tmp_path / "init")
    _write_tokens(tmp_path / "tokens", model.cfg)

    # "crash" after 4 steps (checkpoint saved first), then resume to 8.
    # Both runs use max_steps=8 so the LR schedule is identical.
    with pytest.raises(RuntimeError, match="debug_stop_after_steps"):
        train.main(str(_config(tmp_path, model.cfg, "run_a", max_steps=8,
                               crash_after=4)))
    last_ptr = tmp_path / "run_a" / "checkpoints" / "LAST"
    assert last_ptr.read_text(encoding="utf-8") == "step-000004"
    train.main(str(_config(tmp_path, model.cfg, "run_a", max_steps=8)))

    # fresh reference: 8 steps in one go
    train.main(str(_config(tmp_path, model.cfg, "run_b", max_steps=8)))

    ta, tb = _final_tensors(tmp_path / "run_a"), _final_tensors(tmp_path / "run_b")
    assert ta.keys() == tb.keys()
    for k in ta:
        assert torch.equal(ta[k], tb[k]), f"weight mismatch after resume: {k}"

    # progress file reflects a finished run
    prog = json.loads((tmp_path / "run_a" / "progress.json").read_text(encoding="utf-8"))
    assert prog["status"] == "finished" and prog["step"] == 8
    assert (tmp_path / "run_a" / "logs" / "train.log").exists()


def test_finished_stage_exits_immediately(tmp_path):
    model = tiny_model()
    save_pretrained(model, tmp_path / "init")
    _write_tokens(tmp_path / "tokens", model.cfg)

    cfg_path = _config(tmp_path, model.cfg, "run_c", max_steps=2)
    train.main(str(cfg_path))
    final = tmp_path / "run_c" / "checkpoints" / "final" / "model.safetensors"
    mtime = final.stat().st_mtime
    time.sleep(0.05)
    train.main(str(cfg_path))  # should exit without retraining
    assert final.stat().st_mtime == mtime


def test_lora_grouping_and_merged_final(tmp_path):
    """LoRA stage B: peft-prefixed param names must still land in the right
    optimizer LR groups, resume via the overlay path must work, and the final
    checkpoint must be merged (loadable into a plain PPQLMModel)."""
    pytest.importorskip("peft")
    from ppq.checkpoint import load_pretrained
    from ppq.train.train import make_optimizer, maybe_apply_lora

    model = tiny_model()
    save_pretrained(model, tmp_path / "init")
    _write_tokens(tmp_path / "tokens", model.cfg)

    # --- optimizer grouping with peft's "base_model.model." name prefix ---
    tcfg = TrainConfig(model=model.cfg, lora_rank=8, lr=1e-4,
                       backbone_lr_mult=0.5, depformer_lr_mult=1.5)
    lora_model = maybe_apply_lora(tiny_model(), tcfg)
    opt = make_optimizer(lora_model, tcfg)
    g_lora, g_backbone, g_dep, _ = opt.param_groups
    assert g_lora["lr"] == pytest.approx(tcfg.lora_lr)  # adapters need a real LR
    assert g_backbone["lr"] == pytest.approx(1e-4 * 0.5)
    assert g_dep["lr"] == pytest.approx(1e-4 * 1.5)
    lora_params = [p for n, p in lora_model.named_parameters() if "lora_" in n]
    assert lora_params, "peft did not add any LoRA params"
    assert all(any(p is q for q in g_lora["params"]) for p in lora_params)
    dep_trainable = [p for n, p in lora_model.named_parameters()
                     if p.requires_grad and ".depformer" in n and "lora_" not in n]
    assert dep_trainable
    assert all(any(p is q for q in g_dep["params"]) for p in dep_trainable)

    # --- crash + resume through the LoRA overlay path, then merged final ---
    with pytest.raises(RuntimeError, match="debug_stop_after_steps"):
        train.main(str(_config(tmp_path, model.cfg, "run_lora", max_steps=2,
                               crash_after=1, lora_rank=8)))
    train.main(str(_config(tmp_path, model.cfg, "run_lora", max_steps=2,
                           lora_rank=8)))

    final_dir = tmp_path / "run_lora" / "checkpoints" / "final"
    state = load_file(str(final_dir / "model.safetensors"))
    assert not any(k.startswith("base_model.model.") or "lora_" in k for k in state)
    merged = load_pretrained(final_dir, device="cpu")  # strict load, plain model
    assert type(merged) is type(model)


def test_eval_per_codebook_accepts_str_device(tmp_path):
    """eval_per_codebook must accept device as a plain string ("cpu"/"cuda")."""
    from ppq.data.dataset import TokenShardDataset, collate
    from ppq.train.evaluate import eval_per_codebook

    model = tiny_model()
    _write_tokens(tmp_path / "tokens", model.cfg)
    ds = TokenShardDataset([tmp_path / "tokens" / "manifest-0000.jsonl"],
                           model.cfg, segment_frames=60, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate,
                                         drop_last=True)
    losses = eval_per_codebook(model, loader, n_batches=2, device="cpu",
                               mcfg=model.cfg)
    assert losses and all(np.isfinite(v) for v in losses.values())
