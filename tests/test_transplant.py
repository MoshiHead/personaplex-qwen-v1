"""Verify depformer transplant from a PersonaPlex-style checkpoint."""
import torch
import pytest
from safetensors.torch import save_file

from ppq.model import PPQLMModel
from ppq.transplant import transplant_depformer
from test_ppq_model import tiny_model


def make_source_ckpt(tmp_path):
    """Build a tiny moshi LMModel with an 8-step depformer (like PersonaPlex's
    checkpoint) and save its depformer-side weights."""
    from moshi.models.lm import LMModel
    torch.manual_seed(1234)
    delays = [0] * 17
    model = LMModel(
        delays=delays, n_q=16, dep_q=8, card=2048, text_card=512, dim=64, num_heads=2,
        hidden_scale=2.0, norm="rms_norm_f32", num_layers=2, causal=True, layer_scale=None,
        context=64, max_period=10000, gating="silu", positional_embedding="rope",
        depformer_dim=32, depformer_dim_feedforward=64, depformer_num_heads=4,
        depformer_num_layers=2, depformer_causal=True, depformer_layer_scale=None,
        depformer_multi_linear=True, depformer_context=8, depformer_max_period=10000,
        depformer_gating="silu", depformer_pos_emb="none", depformer_weights_per_step=True,
        existing_text_padding_id=3, device="cpu", dtype=torch.float32)
    path = tmp_path / "source.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    return model, path


def test_transplant_categories(tmp_path):
    src_model, path = make_source_ckpt(tmp_path)
    ppq = tiny_model()
    report = transplant_depformer(ppq, path)
    src_sd = src_model.state_dict()
    ppq_sd = ppq.state_dict()

    # depformer transformer layers: stacked per-step weights expanded 8 -> 16.
    key = "depformer.layers.0.self_attn.in_proj_weight"
    src_w = src_sd[key]
    ppq_w = ppq_sd[key]
    assert ppq_w.shape[0] == 2 * src_w.shape[0]
    assert torch.equal(ppq_w[: src_w.shape[0]], src_w)
    assert torch.equal(ppq_w[src_w.shape[0]:], src_w)
    assert key in report["expanded"]

    # norms copy 1:1 (moshi rms_norm_f32 parameter is named "alpha")
    norm_key = "depformer.layers.0.norm1.alpha"
    assert torch.allclose(ppq_sd[norm_key], src_sd[norm_key])

    # gating per-step entries 8..15 copied from 0..7
    for i in range(8):
        gkey = f"depformer.layers.0.gating.{i}.linear_in.weight"
        assert torch.equal(ppq_sd[gkey], src_sd[gkey])
    for i in range(8, 16):
        gkey = f"depformer.layers.0.gating.{i}.linear_in.weight"
        assert torch.equal(ppq_sd[gkey], src_sd[f"depformer.layers.0.gating.{i - 8}.linear_in.weight"])

    # depformer_emb 0..6 copied, 8..14 copied from 0..6, 7 stays fresh.
    for i in range(7):
        assert torch.equal(ppq_sd[f"depformer_emb.{i}.weight"], src_sd[f"depformer_emb.{i}.weight"])
    for i in range(8, 15):
        assert torch.equal(ppq_sd[f"depformer_emb.{i}.weight"], src_sd[f"depformer_emb.{i - 8}.weight"])
    assert any(n == "depformer_emb.7.weight" for n, _ in report["missing"])

    # linears 0..7 copied, 8..15 from 0..7.
    for i in range(8):
        assert torch.equal(ppq_sd[f"linears.{i}.weight"], src_sd[f"linears.{i}.weight"])
    for i in range(8, 16):
        assert torch.equal(ppq_sd[f"linears.{i}.weight"], src_sd[f"linears.{i - 8}.weight"])

    # depformer_in and depformer_text_emb always fresh.
    assert sum(1 for n, _ in report["missing"] if n.startswith("depformer_in.")) == 16
    assert any(n.startswith("depformer_text_emb.") for n, _ in report["missing"])


def test_transplant_functional_parity(tmp_path):
    """After transplant, PPQ's depformer step cb<8 must reproduce the source
    depformer's logits when fed the same inputs (given the same depformer_in
    projection and text embedding outputs)."""
    src_model, path = make_source_ckpt(tmp_path)
    ppq = tiny_model()
    transplant_depformer(ppq, path)
    # Force identical inputs: copy depformer_in[0] shape-compatible? Shapes
    # differ (64 -> 32 both here since dim=64 in the source model too).
    with torch.no_grad():
        ppq.depformer_in[0].weight.copy_(src_model.depformer_in[0].weight)
        ppq.depformer_text_emb.weight.copy_(src_model.depformer_text_emb.weight)
    torch.manual_seed(9)
    B = 2
    transformer_out = torch.randn(B, 1, 64)
    text_tokens = torch.randint(0, 500, (B, 1, 1))
    with torch.no_grad():
        ref = src_model.forward_depformer(0, text_tokens, transformer_out)
        out = ppq.forward_depformer(0, text_tokens, transformer_out)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max()
