"""Packing policies, loss masks, attention masks and position ids.

Three masks travel with every batch, and each one prevents a specific failure:

  loss mask       which tokens produce gradient.
                  Training on tool observations teaches the model to invent
                  tool results instead of calling the tool.
  attention mask  what each token may look at. Packed documents must not see
                  each other, or the model learns that unrelated text is a
                  natural continuation.
  position ids    where a token sits *inside its own document*. Positions reset
                  at every document boundary, otherwise they lie about distance.

LOSS_IGNORE marks a position as non-loss-bearing (the torch cross-entropy
convention).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .hashing import sha256_bytes

LOSS_IGNORE = -100
PAD_ID = 0


@dataclass
class PackedSequence:
    input_ids: np.ndarray          # (T,)
    labels: np.ndarray             # (T,) LOSS_IGNORE where no loss
    segment_ids: np.ndarray        # (T,) document index inside the pack
    position_ids: np.ndarray       # (T,) resets per document
    provenance: list[dict] = field(default_factory=list)

    @property
    def n_real(self) -> int:
        return int((self.input_ids != PAD_ID).sum())

    @property
    def n_loss(self) -> int:
        return int((self.labels != LOSS_IGNORE).sum())


@dataclass
class PackedBatch:
    seqs: list[PackedSequence]
    step: int
    micro: int

    def tensors(self):
        return (np.stack([s.input_ids for s in self.seqs]),
                np.stack([s.labels for s in self.seqs]),
                np.stack([s.segment_ids for s in self.seqs]),
                np.stack([s.position_ids for s in self.seqs]))

    @property
    def utilization(self) -> float:
        tot = sum(s.input_ids.size for s in self.seqs)
        return sum(s.n_real for s in self.seqs) / max(1, tot)

    @property
    def loss_token_fraction(self) -> float:
        tot = sum(s.input_ids.size for s in self.seqs)
        return sum(s.n_loss for s in self.seqs) / max(1, tot)

    def batch_hash(self) -> str:
        """Identity of this batch. Covers the tokens, where loss lands, document
        isolation, positions and the exact source spans — so a replay that
        differs in any of those produces a different hash."""
        ids, lab, seg, pos = self.tensors()
        payload = (ids.astype(np.uint16).tobytes() +
                   (lab != LOSS_IGNORE).astype(np.uint8).tobytes() +
                   seg.astype(np.uint16).tobytes() +
                   pos.astype(np.uint16).tobytes())
        prov = "|".join(f"{p['shard_id']}:{p['start']}:{p['end']}"
                        for s in self.seqs for p in s.provenance)
        return sha256_bytes(payload + prov.encode())


def attention_allow(segment_ids: np.ndarray) -> np.ndarray:
    """(T,T) bool: causal AND same-document. This is the matrix that stops
    cross-document attention inside a pack."""
    T = segment_ids.shape[0]
    causal = np.tril(np.ones((T, T), dtype=bool))
    same = segment_ids[:, None] == segment_ids[None, :]
    return causal & same


# --------------------------------------------------------------------------
# policy: concat-and-chop (plain text lanes)
# --------------------------------------------------------------------------
def pack_stream_window(tokens: np.ndarray, eos_id: int, shard_id: str,
                       start: int, seq_len: int) -> PackedSequence:
    """One fixed window cut from a concatenated token stream. Documents inside
    the window are separated by EOS; each becomes its own attention segment with
    its own position numbering."""
    win = tokens[start:start + seq_len + 1].astype(np.int64)
    if win.size < seq_len + 1:                       # tail of shard: pad
        win = np.concatenate([win, np.full(seq_len + 1 - win.size, PAD_ID, dtype=np.int64)])
    inp, tgt = win[:-1], win[1:]

    seg = np.zeros(seq_len, dtype=np.int64)
    pos = np.zeros(seq_len, dtype=np.int64)
    s, p = 0, 0
    for i in range(seq_len):
        seg[i] = s
        pos[i] = p
        if inp[i] == eos_id:                          # document ended here
            s += 1
            p = 0
        else:
            p += 1

    labels = tgt.copy()
    labels[inp == PAD_ID] = LOSS_IGNORE               # never learn from padding
    labels[tgt == PAD_ID] = LOSS_IGNORE
    # do not predict across a document boundary
    labels[seg != np.concatenate([seg[1:], seg[-1:]])] = LOSS_IGNORE
    return PackedSequence(inp, labels, seg, pos,
                          [{"shard_id": shard_id, "start": int(start),
                            "end": int(start + seq_len), "policy": "concat_chop"}])


# --------------------------------------------------------------------------
# policy: structure-preserving (reasoning / agentic)
# --------------------------------------------------------------------------
def pack_record(record: dict, shard_id: str, seq_len: int, loss_policy: str) -> PackedSequence:
    """One sample per sequence. Segments tagged `context` (user request, tool
    observations) never bear loss; `model` segments do."""
    toks = np.array(record["tokens"][:seq_len + 1], dtype=np.int64)
    if toks.size < seq_len + 1:
        toks = np.concatenate([toks, np.full(seq_len + 1 - toks.size, PAD_ID, dtype=np.int64)])
    inp, tgt = toks[:-1], toks[1:]

    loss_ok = np.zeros(seq_len, dtype=bool)
    for s in record["segments"]:
        lo, hi = s["start"], min(s["end"], seq_len)
        if hi <= lo:
            continue
        if loss_policy == "all" or (loss_policy in ("response", "model") and s["role"] == "model"):
            loss_ok[lo:hi] = True

    labels = tgt.copy()
    labels[~loss_ok] = LOSS_IGNORE
    labels[inp == PAD_ID] = LOSS_IGNORE
    labels[tgt == PAD_ID] = LOSS_IGNORE
    seg = np.zeros(seq_len, dtype=np.int64)           # single sample = one segment
    pos = np.arange(seq_len, dtype=np.int64)
    return PackedSequence(inp, labels, seg, pos,
                          [{"shard_id": shard_id, "start": 0,
                            "end": int(min(len(record["tokens"]), seq_len)),
                            "doc_id": record["doc_id"], "policy": "structure_preserving"}])
