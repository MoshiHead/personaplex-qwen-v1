"""Qwen text tokenizer wrapper for PPQ.

Replaces PersonaPlex's SentencePiece tokenizer. PAD / EPAD tokens live in
unused reserved rows of the Qwen embedding table (the Qwen3 tokenizer uses
< 151700 ids while the table has 151936 rows), so no resizing is needed.
"""
from __future__ import annotations

from .config import PPQModelConfig


def wrap_with_system_tags(text: str) -> str:
    """Same convention as the PersonaPlex server."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


class QwenTextTokenizer:
    def __init__(self, name_or_path: str, cfg: PPQModelConfig | None = None):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.cfg = cfg or PPQModelConfig()
        # PAD/EPAD must not collide with real tokens.
        n_tokens = len(self.tokenizer)
        for name, idx in (("text_pad_id", self.cfg.text_pad_id), ("text_epad_id", self.cfg.text_epad_id)):
            assert idx >= n_tokens, (
                f"{name}={idx} collides with a real tokenizer id (tokenizer has {n_tokens} tokens). "
                "Pick a higher reserved row of the embedding table.")
            assert idx < self.cfg.text_card

    @property
    def pad_id(self) -> int:
        return self.cfg.text_pad_id

    @property
    def epad_id(self) -> int:
        return self.cfg.text_epad_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_system_prompt(self, text: str) -> list[int]:
        return self.encode(wrap_with_system_tags(text))

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


class StreamingTextDecoder:
    """Incremental decode of the agent text stream for the transcript.

    Qwen's byte-level BPE can split a UTF-8 character across tokens, so we
    re-decode the accumulated ids and hold back incomplete characters.
    """

    def __init__(self, tokenizer: QwenTextTokenizer):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.offset = 0  # chars already emitted from the decoded string

    def reset(self):
        self.ids = []
        self.offset = 0

    def decode_step(self, token_id: int) -> str:
        """Return newly completed text for this token ('' for PAD/EPAD)."""
        if token_id in (self.tok.pad_id, self.tok.epad_id):
            return ""
        if token_id >= self.tok.cfg.text_card:
            return ""
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        emit = text[self.offset:]
        # Hold back a trailing run of U+FFFD (incomplete UTF-8 so far).
        stripped = emit.rstrip("�")
        held = len(emit) - len(stripped)
        self.offset = len(text) - held
        return stripped
