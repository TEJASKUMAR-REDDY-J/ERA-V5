"""Replay, fork and audit.

replay  recompute the batches for a historical step range and compare them, hash
        by hash, against what the original run recorded. Nothing is re-scored
        from the live model, so a matching hash means the same tokens, the same
        loss placement, the same document isolation and the same source spans.

fork    same checkpoint, new branch id. The stream must *differ* -- a fork that
        silently reproduces the parent stream is not a fork, and proving the
        divergence matters as much as proving replay equality.

audit   answer "which shards trained the model between step A and B" from the
        consumption ledger alone.
"""
from __future__ import annotations
from collections import defaultdict

from . import config
from .opus import Opus
from .stream import StreamBuilder


def replay_interval(registry, gate, ledgers, step_lo: int, step_hi: int, log,
                    branch_id: str = "main") -> dict:
    """Re-derive an interval and compare with the recorded stream."""
    original = {}
    for rec in ledgers.consumption.iter():
        if rec["branch_id"] == branch_id and step_lo <= rec["global_step"] < step_hi:
            original[(rec["global_step"], rec["microbatch"])] = rec

    sb = StreamBuilder(registry, gate, Opus(), seed=config.SEED, branch_id=branch_id)
    compared, matched, mismatches = 0, 0, []
    span_matched = 0
    for (step, micro), rec in sorted(original.items()):
        batch, picks = sb.build(step, micro, log)
        h = batch.batch_hash()
        compared += 1
        if h == rec["batch_hash"]:
            matched += 1
        else:
            mismatches.append({"step": step, "micro": micro,
                               "original": rec["batch_hash"][:16], "replay": h[:16]})
        spans = [{"shard_id": p["shard_id"], "start": p["span"]["start"],
                  "end": p["span"]["end"], "lane": p["lane"]} for p in picks]
        if spans == rec["token_spans"]:
            span_matched += 1
    return {"range": [step_lo, step_hi], "compared": compared, "hash_matched": matched,
            "span_matched": span_matched, "mismatches": mismatches[:5],
            "all_matched": compared > 0 and matched == compared and span_matched == compared}


def fork_divergence(registry, gate, parent_branch: str, child_branch: str,
                    steps: list[int], log) -> dict:
    """A fork must produce a different stream from the same checkpoint."""
    a = StreamBuilder(registry, gate, Opus(), seed=config.SEED, branch_id=parent_branch)
    b = StreamBuilder(registry, gate, Opus(), seed=config.SEED, branch_id=child_branch)
    rows, diff = [], 0
    for s in steps:
        ha = a.build(s, 0, log)[0].batch_hash()
        hb = b.build(s, 0, log)[0].batch_hash()
        rows.append({"step": s, "parent": ha[:16], "child": hb[:16], "differs": ha != hb})
        diff += int(ha != hb)
    return {"parent_branch": parent_branch, "child_branch": child_branch,
            "steps_compared": len(steps), "steps_diverged": diff,
            "diverged_everywhere": diff == len(steps), "rows": rows}


def audit_interval(ledgers, step_lo: int, step_hi: int) -> dict:
    """Which data shaped the model over a step range? Answered from the ledger."""
    shard_tokens: dict[str, int] = defaultdict(int)
    lane_tokens: dict[str, int] = defaultdict(int)
    docs: set[str] = set()
    steps = set()
    for rec in ledgers.consumption.iter():
        if not (step_lo <= rec["global_step"] < step_hi):
            continue
        steps.add(rec["global_step"])
        for sp in rec["token_spans"]:
            shard_tokens[sp["shard_id"]] += max(0, sp["end"] - sp["start"])
            lane_tokens[sp["lane"]] += max(0, sp["end"] - sp["start"])
        docs.update(str(d) for d in rec["packed_sample_ids"])
    return {"range": [step_lo, step_hi], "steps_covered": len(steps),
            "distinct_shards": len(shard_tokens), "distinct_samples": len(docs),
            "tokens_by_lane": dict(sorted(lane_tokens.items())),
            "top_shards": sorted(({"shard_id": k, "tokens": v} for k, v in shard_tokens.items()),
                                 key=lambda x: -x["tokens"])[:10]}
