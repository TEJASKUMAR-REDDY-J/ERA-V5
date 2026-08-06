"""The two ledgers.

consumption  what the run actually consumed. Append-only, one record per
             microbatch, carrying enough identity to reconstruct the batch:
             step, branch, shard ids, token spans, mask hashes, lane, stage,
             OPUS decision ids, checkpoint id.

learning     what came of it. Loss attributed back to the shard and lane that
             produced it, so the next corpus version can be told what helped.

A checkpoint stores its consumption-ledger offset. That binding is what makes a
checkpoint a data position as well as a model state -- a checkpoint without one
is incomplete, because you can restore the weights but not the stream.
"""
from __future__ import annotations
from collections import defaultdict

from . import config
from .hashing import JsonlLedger, sha256_bytes


class Ledgers:
    def __init__(self, run_id: str, branch_id: str = "main", suffix: str = ""):
        config.LEDGERS.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.branch_id = branch_id
        self.consumption = JsonlLedger(config.LEDGERS / f"consumption{suffix}.jsonl")
        self.learning = JsonlLedger(config.LEDGERS / f"learning{suffix}.jsonl")
        self.opus = JsonlLedger(config.LEDGERS / f"opus_decisions{suffix}.jsonl")

    # ---- consumption --------------------------------------------------------
    def record_batch(self, *, step: int, micro: int, batch, picks: list[dict],
                     plan, checkpoint_id: str | None, tokenizer_hash: str,
                     dataloader_version: str = "v5flow-1.0.0") -> int:
        ids, labels, seg, pos = batch.tensors()
        rec = {
            "run_id": self.run_id, "branch_id": self.branch_id,
            "global_step": step, "microbatch": micro,
            "checkpoint_id": checkpoint_id, "rank": 0,
            "stage": plan.stage, "sequence_length": plan.sequence_length,
            "batch_hash": batch.batch_hash(),
            "loss_mask_hash": sha256_bytes((labels != -100).astype("uint8").tobytes()),
            "attention_policy": "causal+block_diagonal_by_document",
            "position_policy": "reset_per_document",
            "packed_sample_ids": [p["span"].get("doc_id", p["shard_id"]) for p in picks],
            "shard_ids": sorted({p["shard_id"] for p in picks}),
            "token_spans": [{"shard_id": p["shard_id"], "start": p["span"]["start"],
                             "end": p["span"]["end"], "lane": p["lane"]} for p in picks],
            "lanes": [p["lane"] for p in picks],
            "opus_decision_ids": [p["candidate_id"] for p in picks],
            "opus_statuses": [p.get("opus_status") for p in picks],
            "tokens_total": int(ids.size),
            "tokens_loss_bearing": int((labels != -100).sum()),
            "packing_utilization": round(batch.utilization, 6),
            "tokenizer_hash": tokenizer_hash,
            "dataloader_version": dataloader_version,
        }
        return self.consumption.append(rec)

    # ---- learning -----------------------------------------------------------
    def record_learning(self, *, step: int, micro: int, picks: list[dict],
                        per_token_loss, loss_mask, batch, model_phase: str) -> None:
        """Attribute loss back to the shard/lane that supplied each sequence."""
        for i, p in enumerate(picks):
            if i >= per_token_loss.shape[0]:
                break
            m = loss_mask[i]
            n = int(m.sum())
            if n == 0:
                continue
            vals = per_token_loss[i][m]
            mean = float(vals.mean())
            hi = float((vals > (mean + 2.0)).float().mean())
            self.learning.append({
                "run_id": self.run_id, "branch_id": self.branch_id,
                "global_step": step, "microbatch": micro,
                "shard_id": p["shard_id"], "lane": p["lane"],
                "doc_id": p["span"].get("doc_id"),
                "loss_bearing_tokens": n,
                "mean_token_loss": round(mean, 6),
                "max_token_loss": round(float(vals.max()), 6),
                "token_perplexity": round(float(min(1e6, pow(2.718281828, mean))), 4),
                "high_perplexity_fraction": round(hi, 6),
                "opus_status": p.get("opus_status"),
                "model_phase": model_phase,
            })

    def record_opus(self, rows: list[dict]) -> None:
        for r in rows:
            self.opus.append(r)

    # ---- views used by the evidence pass ------------------------------------
    def realised_mixture(self) -> dict[str, float]:
        c: dict[str, int] = defaultdict(int)
        for rec in self.consumption.iter():
            for lane in rec["lanes"]:
                c[lane] += 1
        tot = sum(c.values()) or 1
        return {l: c[l] / tot for l in sorted(c)}

    def shard_report_card(self) -> list[dict]:
        agg: dict[tuple, dict] = {}
        for r in self.learning.iter():
            k = (r["shard_id"], r["lane"])
            a = agg.setdefault(k, {"shard_id": r["shard_id"], "lane": r["lane"],
                                   "exposures": 0, "tokens": 0, "loss_sum": 0.0,
                                   "first_step": r["global_step"], "last_step": r["global_step"]})
            a["exposures"] += 1
            a["tokens"] += r["loss_bearing_tokens"]
            a["loss_sum"] += r["mean_token_loss"] * r["loss_bearing_tokens"]
            a["first_step"] = min(a["first_step"], r["global_step"])
            a["last_step"] = max(a["last_step"], r["global_step"])
        out = []
        for a in agg.values():
            a["mean_token_loss"] = round(a["loss_sum"] / max(1, a["tokens"]), 6)
            a.pop("loss_sum")
            out.append(a)
        return sorted(out, key=lambda x: (x["lane"], x["shard_id"]))

    def last_offset(self) -> int:
        return self.consumption.count()
