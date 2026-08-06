"""Canonical hashing + append-only JSONL. Every id in this system is derived
from content, never from a counter, so the same input always yields the same id
across processes and runs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace. Two dicts with the
    same content always hash identically."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def hash_tokens(arr: np.ndarray) -> str:
    """Hash a token array by its exact bytes and dtype."""
    a = np.ascontiguousarray(arr, dtype=np.uint16)
    return sha256_bytes(a.tobytes() + str(a.dtype).encode())


def short(h: str, n: int = 12) -> str:
    return h[:n]


class JsonlLedger:
    """Append-only event log. Offsets are line numbers, so a checkpoint can bind
    itself to an exact position in the consumed stream."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> int:
        """Append one record; return its offset (0-based line number)."""
        offset = self.count()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(record) + "\n")
        return offset

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def iter(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def truncate_to(self, offset: int) -> None:
        """Roll the ledger back to `offset` records. Used when a crash left
        events written past the last checkpoint — the resumed run must not
        double-count them."""
        rows = self.read()[:offset]
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(canonical_json(r) + "\n")
