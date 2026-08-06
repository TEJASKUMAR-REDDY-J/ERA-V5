"""S2 tokenizer contract. The tokenizer is frozen before training and every
shard records the hash of the exact file that produced its token ids.

If the tokenizer file changes, every shard built with the old one is invalid —
the ids no longer mean what the manifest says they mean. The gate makes that
failure loud instead of silent.
"""
from __future__ import annotations
from pathlib import Path

from tokenizers import Tokenizer

from . import config
from .hashing import sha256_file


class TokenizerGate:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.TOKENIZER_PATH)
        if not self.path.exists():
            raise FileNotFoundError(f"frozen tokenizer missing: {self.path}")
        self.tok = Tokenizer.from_file(str(self.path))
        self.hash = sha256_file(self.path)
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(list(ids))

    def verify(self, expected_hash: str) -> bool:
        """True when the live tokenizer matches the hash a shard was built with."""
        return self.hash == expected_hash

    def fingerprint(self) -> dict:
        return {"tokenizer_id": config.TOKENIZER_ID, "tokenizer_hash": self.hash,
                "vocab_size": self.vocab_size, "path": self.path.name}
