"""AdamWSR: stochastic-rounding bf16 write-back must accumulate the tiny
updates that pure-bf16 AdamW rounds away (the run2 stall: only ~3.5% of the
intended update was ever applied), while matching stock AdamW on fp32 params.
"""
import torch

from ppq.train.optim import AdamWSR, _stochastic_round_bf16


def test_stochastic_round_unbiased_and_on_lattice():
    torch.manual_seed(0)
    x = torch.randn(1_000_000) * 0.02
    r = _stochastic_round_bf16(x)
    # every output is exactly representable in bf16
    assert torch.equal(r, r.bfloat16().float())
    # unbiased: the mean rounding error over 1M values is ~0
    assert abs((r - x).mean().item()) < 5e-6
    # values already on the lattice are returned unchanged
    y = torch.tensor([0.5, -2.0, 100.0, 0.0])
    assert torch.equal(_stochastic_round_bf16(y), y)


def _drift(opt_cls, dtype, steps=1000, lr=2.5e-6):
    """Mean |drift| of a bf16 param under a constant gradient (ideal = lr*steps)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter((torch.randn(256, 2048) * 0.02).to(dtype))
    p0 = p.detach().float().clone()
    opt = opt_cls([p], lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    g = torch.randn(256, 2048) * 1e-3
    for _ in range(steps):
        opt.zero_grad()
        p.grad = g.to(dtype).clone()
        opt.step()
    return (p.detach().float() - p0).abs().mean().item()


def test_tiny_updates_accumulate_bf16():
    # run2's backbone-group LR: at 2.5e-6 the per-step update is ~30x below the
    # bf16 half-ulp of a 0.02-scale weight, so stock AdamW rounds it away.
    ideal = 1000 * 2.5e-6
    sr = _drift(AdamWSR, torch.bfloat16)
    stock = _drift(torch.optim.AdamW, torch.bfloat16)
    assert sr > 0.6 * ideal, f"AdamWSR applied only {sr / ideal:.1%} of the ideal drift"
    assert stock < 0.25 * ideal, f"control unexpectedly moved: {stock / ideal:.1%}"


def test_fp32_params_match_stock_adamw():
    torch.manual_seed(1)
    p1 = torch.nn.Parameter(torch.randn(64, 64) * 0.02)
    p2 = torch.nn.Parameter(p1.detach().clone())
    o1 = AdamWSR([p1], lr=1e-3, weight_decay=0.01)
    o2 = torch.optim.AdamW([p2], lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95))
    for _ in range(20):
        g = torch.randn(64, 64)
        o1.zero_grad(); p1.grad = g.clone(); o1.step()
        o2.zero_grad(); p2.grad = g.clone(); o2.step()
    assert torch.allclose(p1, p2, atol=1e-5)


def test_state_dict_roundtrip():
    torch.manual_seed(2)
    p = torch.nn.Parameter((torch.randn(32, 32) * 0.02).bfloat16())
    opt = AdamWSR([p], lr=1e-4)
    for _ in range(5):
        opt.zero_grad()
        p.grad = torch.randn(32, 32).bfloat16()
        opt.step()
    sd = opt.state_dict()
    assert sd["state"][0]["step"] == 5
    p2 = torch.nn.Parameter((torch.randn(32, 32) * 0.02).bfloat16())
    opt2 = AdamWSR([p2], lr=1e-4)
    opt2.load_state_dict(sd)
    assert opt2.state[p2]["step"] == 5
    opt2.zero_grad()
    p2.grad = torch.randn(32, 32).bfloat16()
    opt2.step()  # resumes without error
    assert opt2.state[p2]["step"] == 6
