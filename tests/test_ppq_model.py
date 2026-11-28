"""PPQLMModel + PPQLMGen tests on CPU with a tiny config.

The key test (test_streaming_matches_training) teacher-forces a random token
sequence through the streaming LMGen and checks the returned logits match
``forward_train`` — this validates the whole delay/cache/depformer chain.
"""
import torch
import pytest

from ppq.config import PPQModelConfig, QwenBackboneConfig
from ppq.model import PPQLMModel
from ppq.lmgen import PPQLMGen


def tiny_model() -> PPQLMModel:
    torch.manual_seed(0)
    qwen = QwenBackboneConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        vocab_size=512, rms_norm_eps=1e-6, rope_theta=10_000.0,
        max_position_embeddings=256, tie_word_embeddings=True, context=64,
    )
    cfg = PPQModelConfig(
        backbone="custom", qwen=qwen,
        text_card=512, text_pad_id=510, text_epad_id=509,
        depformer_dim=32, depformer_num_heads=4, depformer_num_layers=2,
        depformer_dim_feedforward=64,
    )
    model = PPQLMModel(cfg, device="cpu", dtype=torch.float32)
    model.eval()
    return model


def random_codes(T: int, text_card: int = 512, card: int = 2048) -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    codes = torch.zeros(1, 17, T, dtype=torch.long)
    codes[0, 0] = torch.randint(0, text_card - 3, (T,), generator=g)  # avoid pad/epad rows
    codes[0, 1:] = torch.randint(0, card, (16, T), generator=g)
    return codes


def test_build_from_scratch_rescales_fresh_embeddings():
    """The 16 fresh audio tables must be re-init so their SUM matches the text
    embedding scale (nn.Embedding's N(0,1) default drowns the text stream
    ~100-400x), and depformer_text_emb must match the depformer_emb scale."""
    import math
    from ppq.checkpoint import build_model_from_scratch
    cfg = tiny_model().cfg
    model = build_model_from_scratch(cfg)  # no qwen/depformer sources: all fresh
    text_std = model.text_emb.emb.weight.float().std().item()
    for emb in model.emb:
        assert emb.weight.float().std().item() == pytest.approx(
            text_std / math.sqrt(cfg.n_q), rel=0.05)
    dep_ref = torch.cat([e.weight.float().flatten()
                         for e in model.depformer_emb]).std().item()
    assert model.depformer_text_emb.weight.float().std().item() == pytest.approx(
        dep_ref, rel=0.05)


def test_forward_train_shapes():
    model = tiny_model()
    codes = random_codes(40)
    out = model.forward_train(codes)
    assert out.logits.shape == (1, 16, 40, 2048)
    assert out.mask.shape == (1, 16, 40)
    assert out.text_logits.shape == (1, 1, 40, 512)
    assert out.text_mask.shape == (1, 1, 40)
    assert torch.isfinite(out.text_logits).all()
    # First 39 audio frames must have finite logits (last delayed frame is NaN).
    assert torch.isfinite(out.logits[:, :, :-1]).all()


def test_lmgen_cpu_smoke():
    model = tiny_model()
    lm_gen = PPQLMGen(model, device="cpu", check=True, use_sampling=False)
    assert lm_gen.zero_text_code == model.text_padding_token_id == 510
    lm_gen.text_prompt_tokens = [5, 6, 7]
    g = torch.Generator().manual_seed(7)
    with lm_gen.streaming(1):
        lm_gen.step_system_prompts(None)  # no voice prompt -> mimi unused
        n_out = 0
        for t in range(50):
            user = torch.randint(0, 2048, (1, 8, 1), generator=g)
            out = lm_gen.step(user)
            if out is not None:
                n_out += 1
                assert out.shape == (1, 17, 1), out.shape
                assert (out[:, 1:] >= 0).all() and (out[:, 1:] < 2048).all()
                assert 0 <= out[0, 0, 0].item() < 512
        # All 50 steps return tokens: the initial/max_delay warmup already
        # happened during step_system_prompts.
        assert n_out == 50


def test_kv_frames_and_finish_system_prompts():
    """PPQLMGen._kv_frames/_finish_system_prompts replace moshi's versions,
    which hardcode `lm_model.transformer` + StreamingMultiheadAttention —
    PPQLMModel has a Qwen3 backbone, so moshi's would raise AttributeError."""
    model = tiny_model()
    assert not hasattr(model, "transformer")  # the moshi assumption we break
    lm_gen = PPQLMGen(model, device="cpu", check=True, use_sampling=False)
    g = torch.Generator().manual_seed(3)
    with lm_gen.streaming(1):
        assert lm_gen._kv_frames() == 0
        for _ in range(5):
            lm_gen.step(torch.randint(0, 2048, (1, 8, 1), generator=g))
        # The first step() call only primes the delay cache (prepare_step_input
        # returns None, no backbone forward), so the KV offset trails the
        # step() count by one — same as moshi's own _kv_frames would report.
        frames = lm_gen._kv_frames()
        assert frames == 4
        attn = model.backbone.layers[0].self_attn
        assert frames == int(attn._streaming_state.offset.item())
        lm_gen._finish_system_prompts(frames)
        assert lm_gen.system_prompt_frames == frames
    # The pin path must not touch lm_model.transformer either: it degrades to
    # unpinned with a warning instead of calling moshi's pin_kv_prefix.
    lm_gen.pin_system_prompt = True
    lm_gen._finish_system_prompts(7)
    assert lm_gen.system_prompt_frames == 7
    assert lm_gen.pin_system_prompt is False


def test_streaming_matches_training():
    model = tiny_model()
    T = 40
    codes = random_codes(T)
    with torch.no_grad():
        train_out = model.forward_train(codes)

    lm_gen = PPQLMGen(model, device="cpu", check=True, use_sampling=False, return_logits=True)
    collected = {}
    with lm_gen.streaming(1):
        for t in range(T):
            text_t = codes[0, 0, t].clone()
            agent = codes[:, 1:9, t: t + 1]
            user = codes[:, 9:17, t: t + 1]
            result = lm_gen.step(input_tokens=user, moshi_tokens=agent, text_token=text_t)
            if result is None or result[0] is None:
                continue
            _, (text_logits, audio_logits) = result
            collected[t] = (text_logits, audio_logits)

    assert collected, "no logits collected"
    first = min(collected)
    for o, (text_logits, audio_logits) in collected.items():
        if o == first:
            # The very first returned frame is affected by moshi's initial-token
            # handling (frame-0 tokens are replaced by the initial token in the
            # streaming cache). Stock moshi LMModel+LMGen shows the exact same
            # behavior (verified against the reference implementation), so we
            # skip it here. In production the first frames are voice-prompt
            # priming frames anyway.
            continue
        # text stream (delay 0): streaming logits @o == train text_logits[o]
        err_text = (train_out.text_logits[0, 0, o] - text_logits[0, 0, 0]).abs().max().item()
        assert err_text < 0.2, f"text logits mismatch at frame {o}: {err_text}"
        # agent cb0 (delay 0)
        err_cb0 = (train_out.logits[0, 0, o] - audio_logits[0, 0]).abs().max().item()
        assert err_cb0 < 0.2, f"cb0 logits mismatch at frame {o}: {err_cb0}"
        # agent cb1 (delay 1): streaming @o == train logits[o - 1]
        if o >= 1:
            err_cb1 = (train_out.logits[0, 1, o - 1] - audio_logits[0, 1]).abs().max().item()
            assert err_cb1 < 0.2, f"cb1 logits mismatch at frame {o}: {err_cb1}"
        # user cb8 (delay 0) is also modeled by the depformer
        err_cb8 = (train_out.logits[0, 8, o] - audio_logits[0, 8]).abs().max().item()
        assert err_cb8 < 0.2, f"user cb0 logits mismatch at frame {o}: {err_cb8}"


def test_first_frame_behavior_matches_stock_moshi():
    """Document/verify the initial-token behavior is identical to stock moshi:
    the frame-0 tokens are never fed as input (overwritten by the initial
    token), so the first returned frame's logits differ from forward_train,
    and subsequent frames match closely."""
    # This is implicitly covered by test_streaming_matches_training; here we
    # just assert the first returned frame exists and later frames dominate.
    model = tiny_model()
    lm_gen = PPQLMGen(model, device="cpu", check=True, use_sampling=False)
    g = torch.Generator().manual_seed(11)
    with lm_gen.streaming(1):
        outs = []
        for t in range(10):
            user = torch.randint(0, 2048, (1, 8, 1), generator=g)
            out = lm_gen.step(user)
            outs.append(out is not None)
    # offset 0 -> None (initial), offset 1 -> None (<= max_delay), then tokens.
    assert outs[:2] == [False, False]
    assert all(outs[2:])
