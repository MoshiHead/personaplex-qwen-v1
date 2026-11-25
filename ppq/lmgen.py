"""PPQ streaming inference engine: a thin subclass of PersonaPlex's LMGen.

The only PersonaPlex-specific constant baked into LMGen is `zero_text_code`
(the text PAD token, hardcoded to 3 for the SentencePiece tokenizer). We
override it with the Qwen-side PAD id. Everything else — delay cache, CUDA
graphs, voice/text system prompts — is reused unchanged.
"""
from __future__ import annotations

import torch

from moshi.models.lm import LMGen

from .model import PPQLMModel


class PPQLMGen(LMGen):
    def __init__(self, lm_model: PPQLMModel, device: str | torch.device, **kwargs):
        kwargs.setdefault("audio_silence_frame_cnt", lm_model.cfg.audio_silence_frame_cnt)
        kwargs.setdefault("sample_rate", lm_model.cfg.sample_rate)
        kwargs.setdefault("frame_rate", lm_model.cfg.frame_rate)
        super().__init__(lm_model, device=device, **kwargs)
        # LMGen hardcodes zero_text_code = 3 (SentencePiece PAD). Use Qwen's PAD.
        self.zero_text_code = lm_model.text_padding_token_id
