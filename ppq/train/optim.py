"""bf16-safe AdamW: fp32 update math + stochastic-rounding write-back.

Pure-bf16 AdamW silently rounds away small updates: for a weight of ~0.02 the
bf16 ulp is ~1.6e-4, so any |update| below ~8e-5 (half ulp) rounds to zero.
At fine-tuning LRs (1e-6..1e-5) that is 96-99% of all updates — training
stalls after a few hundred steps (observed on the run2 Stage A/B logs, where
only ~3.5% of the intended backbone-group update was ever applied).

AdamWSR keeps params AND Adam states in the param dtype (bf16: same memory as
stock AdamW) but computes each update in fp32 and writes it back with
stochastic rounding: the exact fp32 candidate is rounded up or down to the
adjacent bf16 lattice values with probability proportional to proximity, so
E[rounded] == exact and tiny updates accumulate over steps instead of
vanishing. For non-bf16 params it degenerates to plain AdamW.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

_SR_CHUNK = 1 << 24  # bound stochastic-rounding temporaries to ~64 MB


def _stochastic_round_bf16(x: torch.Tensor,
                           generator: torch.Generator | None = None) -> torch.Tensor:
    """Round an fp32 tensor onto the bf16 lattice stochastically (unbiased).

    Each value becomes one of the two adjacent bf16-representable values, with
    probability proportional to proximity, so E[out] == x. Values already on
    the lattice are returned unchanged.
    """
    bits = x.view(torch.int32)
    trunc = bits & -65536        # 0xFFFF0000: drop low mantissa bits (toward zero)
    away = trunc + 0x10000       # one bf16 ulp away from zero
    lo = torch.where(x >= 0, trunc, away).view(torch.float32)
    hi = torch.where(x >= 0, away, trunc).view(torch.float32)
    span = hi - lo
    p_hi = torch.where(span > 0, (x - lo) / span.clamp(min=1e-45),
                       torch.zeros_like(x))
    r = torch.rand(x.shape, device=x.device, generator=generator)
    return torch.where(r < p_hi, hi, lo)


class AdamWSR(Optimizer):
    """AdamW (decoupled weight decay) with stochastic-rounding bf16 write-back.

    State (exp_avg, exp_avg_sq) is stored in the param dtype, so memory is
    identical to stock AdamW on bf16 params; only the update is computed in
    fp32 and stochastically rounded back. State dict is compatible with the
    standard Optimizer save/load used for training resume.
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                # fp32 moment updates (bf16 grad is promoted by the kernels).
                # NOTE: .float() aliases the state tensor when it is already
                # fp32, so the bias-corrected copies below must start with an
                # out-of-place op to avoid corrupting the stored moments.
                m = state["exp_avg"].float().mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v = state["exp_avg_sq"].float().mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2)
                state["exp_avg"].copy_(m)
                state["exp_avg_sq"].copy_(v)
                denom = v.div(1 - beta2 ** t).sqrt_().add_(eps)
                update = m.div(1 - beta1 ** t).div_(denom)
                candidate = p.float()
                if wd != 0:
                    candidate.mul_(1 - lr * wd)
                candidate.add_(update, alpha=-lr)
                if p.dtype == torch.bfloat16:
                    flat_p = p.view(-1)
                    flat_c = candidate.view(-1)
                    for i in range(0, flat_c.numel(), _SR_CHUNK):
                        flat_p[i:i + _SR_CHUNK].copy_(
                            _stochastic_round_bf16(flat_c[i:i + _SR_CHUNK]))
                else:
                    p.copy_(candidate.to(p.dtype))
        return loss
