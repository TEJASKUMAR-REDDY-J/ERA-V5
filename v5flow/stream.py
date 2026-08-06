"""The deterministic batch stream.

This is the architectural centre of the system. The composition of any batch is
a PURE FUNCTION of

    (seed, branch_id, global_step, microbatch_index)

and nothing else. There is no iterator position, no shuffle buffer, no worker
state, no consumed-set that mutates as the run proceeds.

Everything the assignment asks to prove falls out of that one property:

  resume  recompute batch(step) after restarting -> identical by construction
  replay  recompute batch(step) for an old interval -> identical by construction
  fork    change branch_id -> a provably different stream from the same weights

A stateful dataloader can only *assert* these; a pure function makes them true.
"""
from __future__ import annotations

import numpy as np

from . import config, mixture, shards
from .hashing import hash_obj
from .packing import PackedBatch, pack_record, pack_stream_window

CANDIDATES_PER_SLOT = 3        # OPUS scores this many candidates per needed sequence


def branch_int(branch_id: str) -> int:
    return int(hash_obj({"branch": branch_id})[:8], 16)


def _rng(seed: int, branch_id: str, step: int, micro: int, salt: int = 0):
    """Counter-based: derived only from its coordinates, never advanced in place."""
    return np.random.default_rng([seed, branch_int(branch_id), step, micro, salt])


class StreamBuilder:
    def __init__(self, registry, gate, opus, seed: int = config.SEED,
                 branch_id: str = "main"):
        self.reg = registry
        self.gate = gate
        self.opus = opus
        self.seed = seed
        self.branch_id = branch_id
        self.eos_id = gate.tok.token_to_id("[UNK]") or 0
        self._stream_cache: dict[str, np.ndarray] = {}
        self._record_cache: dict[str, list[dict]] = {}

    # ---- shard access -------------------------------------------------------
    def _stream(self, sid: str) -> np.ndarray:
        if sid not in self._stream_cache:
            self._stream_cache[sid] = shards.load_stream(sid)
        return self._stream_cache[sid]

    def _records(self, sid: str) -> list[dict]:
        if sid not in self._record_cache:
            self._record_cache[sid] = shards.load_records(sid)
        return self._record_cache[sid]

    def _lane_shards(self, lane: str, plan) -> list[str]:
        """Anneal-reserve shards stay invisible until the anneal stage. If the
        selector spends the best data early there is nothing left to concentrate
        at the end, so the reserve is enforced here rather than hoped for."""
        ids = self.reg.trainable(lane, include_reserve=False)
        if plan.is_anneal:
            ids = sorted(ids + self.reg.reserve(lane))
        return ids

    # ---- candidate generation ----------------------------------------------
    def candidates(self, plan, micro: int) -> list[dict]:
        """Deterministic candidate pool for one microbatch."""
        out: list[dict] = []
        for lane in config.LANES:
            need = plan.quota.get(lane, 0)
            if need <= 0:
                continue
            sids = self._lane_shards(lane, plan)
            if not sids:
                continue
            # Stable lane salt. Python's builtin hash() on str is randomised per
            # process (PYTHONHASHSEED), so using it here would make the stream
            # differ between the training process and any replay process --
            # silently destroying the purity the whole design depends on.
            r = _rng(self.seed, self.branch_id, plan.step, micro,
                     salt=config.LANES.index(lane) + 1)
            n_cand = need * CANDIDATES_PER_SLOT
            for k in range(n_cand):
                sid = sids[int(r.integers(0, len(sids)))]
                man = self.reg.get(sid)
                if man["kind"] == "stream":
                    arr = self._stream(sid)
                    max_start = max(1, arr.size - plan.sequence_length - 1)
                    start = int(r.integers(0, max_start))
                    preview = arr[start:start + plan.sequence_length]
                    out.append({"candidate_id": f"c-{plan.step}-{micro}-{lane}-{k}",
                                "lane": lane, "shard_id": sid, "kind": "stream",
                                "start": start, "tokens_preview": preview,
                                "n_tokens": int(preview.size)})
                else:
                    recs = self._records(sid)
                    ridx = int(r.integers(0, len(recs)))
                    rec = recs[ridx]
                    out.append({"candidate_id": f"c-{plan.step}-{micro}-{lane}-{k}",
                                "lane": lane, "shard_id": sid, "kind": "records",
                                "record_index": ridx,
                                "tokens_preview": np.array(rec["tokens"][:plan.sequence_length]),
                                "n_tokens": len(rec["tokens"])})
        return out

    # ---- batch assembly -----------------------------------------------------
    def build(self, step: int, micro: int, log) -> tuple[PackedBatch, list[dict]]:
        plan = mixture.compile_step(step, config.MICRO_BATCH)
        cands = self.candidates(plan, micro)
        chosen = self.opus.judge(cands, plan, log)

        seqs, picks = [], []
        for c in chosen:
            # firewall is checked again at the point of use, not only at build
            if not self.reg.admit(c["shard_id"])[0]:
                self.reg.admit_or_block(c["shard_id"], log, context=f"step{step}")
                continue
            if c["kind"] == "stream":
                seq = pack_stream_window(self._stream(c["shard_id"]), self.eos_id,
                                         c["shard_id"], c["start"], plan.sequence_length)
            else:
                rec = self._records(c["shard_id"])[c["record_index"]]
                seq = pack_record(rec, c["shard_id"], plan.sequence_length,
                                  config.LOSS_POLICY[c["lane"]])
            seqs.append(seq)
            picks.append({"lane": c["lane"], "shard_id": c["shard_id"],
                          "candidate_id": c["candidate_id"],
                          "opus_status": c.get("opus_status"),
                          "span": seq.provenance[0]})
        batch = PackedBatch(seqs=seqs, step=step, micro=micro)
        return batch, picks

    def expected_batch_hash(self, step: int, micro: int, log) -> str:
        """Recompute a batch without training on it. Used to state the expected
        next batch *before* a crash and to verify it *after* a resume."""
        b, _ = self.build(step, micro, log)
        return b.batch_hash()
