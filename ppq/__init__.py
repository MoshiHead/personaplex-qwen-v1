"""PersonaPlex-Qwen (PPQ): PersonaPlex full-duplex speech model with a Qwen3 backbone."""

__version__ = "0.1.0"

from .config import (
    QwenBackboneConfig,
    PPQModelConfig,
    TrainConfig,
    QWEN3_PRESETS,
    DEFAULT_BACKBONE,
    DEFAULT_DELAYS,
    SILENCE_TOKENS,
    SINE_TOKENS,
    SAMPLE_RATE,
    FRAME_RATE_HZ,
    FRAME_SIZE,
    MIMI_CARD,
)
