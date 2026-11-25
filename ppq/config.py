"""Configuration for PersonaPlex-Qwen (PPQ).

PPQ keeps the PersonaPlex/Moshi architecture (Mimi codec + temporal transformer
+ depformer, 17 token streams at 12.5 Hz) but replaces the Helium-7B backbone
with a Qwen3 backbone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# --- Constants shared with PersonaPlex (moshi/models/lm.py) ------------------
AUDIO_TOKENS_PER_STREAM = 8
FRAME_RATE_HZ = 12.5
SAMPLE_RATE = 24000
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE_HZ)  # 1920 samples
MIMI_CARD = 2048
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)

# 17 streams: text, 8 agent audio, 8 user audio. Delays in frames.
DEFAULT_DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]


@dataclass
class QwenBackboneConfig:
    """Subset of HF Qwen3 config.json that we need."""
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True
    # Streaming ring-buffer capacity, in frames (12.5 fps). 3000 = 240 s,
    # matching PersonaPlex's transformer context.
    context: int = 3000

    @classmethod
    def from_hf_config(cls, path: str | Path, context: int = 3000) -> "QwenBackboneConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg.get("model_type") == "qwen3", f"Expected a Qwen3 config, got {cfg.get('model_type')}"
        kw = {k: cfg[k] for k in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")}
        return cls(
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=cfg.get("rope_theta", 1_000_000.0),
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            context=context, **kw)



# Verified against the official config.json files on Hugging Face (2026-01).
QWEN3_PRESETS: dict[str, QwenBackboneConfig] = {
    "qwen3-0.6b": QwenBackboneConfig(  # debug/CPU preset
        hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, tie_word_embeddings=True,
    ),
    "qwen3-1.7b": QwenBackboneConfig(  # default
        hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, tie_word_embeddings=True,
    ),
    "qwen3-4b": QwenBackboneConfig(
        hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, tie_word_embeddings=True,
    ),
    "qwen3-8b": QwenBackboneConfig(
        hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, tie_word_embeddings=False,
    ),
}

DEFAULT_BACKBONE = "qwen3-1.7b"


@dataclass
class PPQModelConfig:
    """Full PPQ model configuration (serialized to config.json next to weights)."""
    backbone: str = DEFAULT_BACKBONE
    qwen: QwenBackboneConfig | None = None
    # Text stream. PAD/EPAD live in unused rows of the Qwen embedding table
    # (Qwen3 tokenizer uses < 151700 ids; the table has 151936 rows).
    text_card: int = 151936
    text_pad_id: int = 151934
    text_epad_id: int = 151933
    # Audio streams.
    card: int = MIMI_CARD
    n_q: int = 16          # 8 agent + 8 user codebooks
    dep_q: int = 16        # depformer models agent + user codebooks (PersonaPlex style)
    # Depformer (identical to PersonaPlex so weights can be transplanted).
    depformer_dim: int = 1024
    depformer_dim_feedforward: int = 4224  # int(4.125 * 1024)
    depformer_num_heads: int = 16
    depformer_num_layers: int = 6
    depformer_context: int = 8
    delays: list[int] = field(default_factory=lambda: list(DEFAULT_DELAYS))
    # Prompt handling (inference): frames of "silence" after voice/text prompts.
    audio_silence_frame_cnt: int = 6  # int(0.5 * 12.5)
    sample_rate: int = SAMPLE_RATE
    frame_rate: float = FRAME_RATE_HZ

    def __post_init__(self):
        if self.qwen is None:
            self.qwen = QWEN3_PRESETS[self.backbone]
        assert len(self.delays) == self.n_q + 1
        assert self.dep_q == self.n_q
        assert self.text_pad_id < self.text_card and self.text_epad_id < self.text_card
        assert self.qwen.vocab_size == self.text_card

    @property
    def dim(self) -> int:
        return self.qwen.hidden_size

    def to_json(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "PPQModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["qwen"] = QwenBackboneConfig(**data["qwen"])
        return cls(**data)


@dataclass
class TrainConfig:
    """Training hyper-parameters (serialized to train_config.json)."""
    model: PPQModelConfig = field(default_factory=PPQModelConfig)
    data_dir: str = "/workspace/ppq_data/tokens"
    output_dir: str = "/workspace/ppq_runs/run1"
    stage: str = "b_full"  # "a_audio_warmup" | "b_full"
    # Optimization
    lr: float = 2e-5
    backbone_lr_mult: float = 1.0
    depformer_lr_mult: float = 0.5
    batch_size: int = 8            # conversations per GPU
    grad_accum: int = 8
    max_steps: int = 2000
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    segment_frames: int = 1250     # 100 s at 12.5 fps
    # Losses
    text_padding_weight: float = 0.5
    first_codebook_weight_multiplier: float = 100.0
    user_codebook_weight: float = 0.0  # aux loss on user audio codebooks (depformer cb 8..15)
    # LoRA (0 = full fine-tune)
    lora_rank: int = 0
    lora_alpha: int = 256
    # Distributed / numerics
    fsdp: bool = False
    gradient_checkpointing: bool = True
    seed: int = 1234
    num_workers: int = 4
    shuffle: bool = True              # shuffle sample order (disable for exact-repeat tests)
    log_every: int = 10
    save_every: int = 250
    save_every_minutes: float = 30.0  # time-based saves: a crash loses at most this much work
    keep_last_k: int = 3              # step checkpoints to keep (disk usage)
    # Resume / initialization
    init_checkpoint: str = ""         # fresh start dir; empty -> <output_dir>/init
    auto_resume: bool = True          # resume from checkpoints/LAST when available
    # Debug/testing: crash after N steps (after saving, if a save was due)
    debug_stop_after_steps: int = 0

    def to_json(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        import dataclasses
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data["model"]
        m["qwen"] = QwenBackboneConfig(**m["qwen"])
        data["model"] = PPQModelConfig(**m)
        # tolerate configs written by other versions (unknown keys are ignored)
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)
