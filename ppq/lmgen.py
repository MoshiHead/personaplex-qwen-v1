"""PPQ streaming inference engine: a thin subclass of PersonaPlex's LMGen.

PersonaPlex-specific assumptions baked into LMGen that we override:
  * `zero_text_code` (the text PAD token, hardcoded to 3 for the SentencePiece
    tokenizer) -> the Qwen-side PAD id.
  * `_kv_frames` / `_finish_system_prompts` hardcode `lm_model.transformer`
    (moshi's StreamingTransformer) and scan for `StreamingMultiheadAttention`;
    the PPQ backbone is a Qwen3 stack whose attention is `Qwen3Attention`.
Everything else — delay cache, CUDA graphs, voice/text system prompts — is
reused unchanged.
"""
from __future__ import annotations

import torch

from moshi.models.lm import LMGen

from .model import PPQLMModel
from .qwen_backbone import Qwen3Attention


class PPQLMGen(LMGen):
    def __init__(self, lm_model: PPQLMModel, device: str | torch.device, **kwargs):
        kwargs.setdefault("audio_silence_frame_cnt", lm_model.cfg.audio_silence_frame_cnt)
        kwargs.setdefault("sample_rate", lm_model.cfg.sample_rate)
        kwargs.setdefault("frame_rate", lm_model.cfg.frame_rate)
        super().__init__(lm_model, device=device, **kwargs)
        # LMGen hardcodes zero_text_code = 3 (SentencePiece PAD). Use Qwen's PAD.
        self.zero_text_code = lm_model.text_padding_token_id

    def _kv_frames(self) -> int:
        """Number of frames written to the backbone KV cache so far.

        PersonaPlex's LMGen scans `lm_model.transformer` for
        `StreamingMultiheadAttention`; PPQLMModel has a Qwen3 backbone instead.
        """
        for module in self.lm_model.backbone.modules():
            if isinstance(module, Qwen3Attention) and module._streaming_state is not None:
                return int(module._streaming_state.offset.item())
        return 0

    def _finish_system_prompts(self, frames: int) -> None:
        self.system_prompt_frames = frames
        if getattr(self, "pin_system_prompt", False):
            # moshi's pin_kv_prefix() only knows StreamingMultiheadAttention,
            # so KV-prefix pinning is not wired up for the Qwen backbone. The
            # prompt still lives in the ring cache until the sliding window
            # evicts it: generation is correct, just not pinned.
            print(
                f"System prompt: {frames} frames "
                "(KV pinning not supported for the PPQ backbone; not pinned)."
            )
            self.pin_system_prompt = False
        else:
            print(f"System prompt: {frames} frames (not pinned).")
