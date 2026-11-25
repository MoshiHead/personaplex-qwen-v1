"""Toy training step: forward_train -> compute_losses -> backward."""
import math

import torch

from ppq.config import TrainConfig
from ppq.train.loss import compute_losses
from test_ppq_model import tiny_model, random_codes


def test_toy_training_step():
    model = tiny_model()
    model.train()
    codes = random_codes(60)
    mask = torch.ones(1, 17, 60)
    mask[:, :, :6] = 0  # pretend a 6-frame prompt prefix
    out = model.forward_train(codes)
    tcfg = TrainConfig()
    losses = compute_losses(out, codes, mask, tcfg, model.cfg)
    # Random-init CE should be near the vocabulary entropy.
    assert abs(losses["text"].item() - math.log(512)) < 1.0
    assert abs(losses["cb1"].item() - math.log(2048)) < 1.5
    # cb0 is weighted 100x by default; user codebooks are off by default.
    assert losses["cb0"].item() > 50 * losses["cb1"].item()
    assert losses["cb8"].item() == 0.0
    losses["total"].backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    assert n_with_grad == n_total
    assert torch.isfinite(losses["total"])


def test_user_codebook_weight():
    model = tiny_model()
    codes = random_codes(40)
    mask = torch.ones(1, 17, 40)
    out = model.forward_train(codes)
    tcfg = TrainConfig(user_codebook_weight=0.1)
    losses = compute_losses(out, codes, mask, tcfg, model.cfg)
    assert losses["cb8"].item() > 0.0
    assert torch.isfinite(losses["total"])
