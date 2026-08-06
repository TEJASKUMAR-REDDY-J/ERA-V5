"""Shard registry, admission gate and evaluation firewall.

The registry knows about every shard, including the ones that must never train.
That is the point: the system can only block evaluation data if it knows the
data exists. Test shards are registered with never_train=True and their content
hashes are kept as contamination fingerprints.
"""
from __future__ import annotations
import json
from dataclasses import asdict

from . import config
from .shards import Manifest


class AdmissionError(Exception):
    pass


class Registry:
    def __init__(self, gate_hash: str):
        self.gate_hash = gate_hash
        self.manifests: dict[str, dict] = {}
        self.blocked: list[dict] = []
        self.eval_fingerprints: set[str] = set()

    # ---- registration -------------------------------------------------------
    def register(self, m: Manifest, log) -> None:
        d = asdict(m)
        self.manifests[m.shard_id] = d
        if d["never_train"]:
            self.eval_fingerprints.add(d["content_hash"])
            log.event("eval_shard_registered", shard_id=m.shard_id,
                      content_hash=d["content_hash"][:16])

    def load_from_disk(self) -> None:
        for p in sorted(config.MANIFESTS.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            self.manifests[d["shard_id"]] = d
            if d.get("never_train"):
                self.eval_fingerprints.add(d["content_hash"])

    # ---- admission ----------------------------------------------------------
    def admit(self, shard_id: str) -> tuple[bool, str]:
        """Decide whether a shard may enter a loss-bearing batch.
        Returns (allowed, reason). Every rejection reason is distinct so the
        evidence bundle can prove which rule fired."""
        d = self.manifests.get(shard_id)
        if d is None:
            return False, "unknown_shard"
        if d.get("never_train"):
            return False, "eval_firewall_never_train"
        if d["content_hash"] in self.eval_fingerprints:
            return False, "eval_firewall_fingerprint_match"
        if d.get("eval_overlap"):
            return False, "eval_overlap"
        if d["tokenizer_hash"] != self.gate_hash:
            return False, "tokenizer_hash_mismatch"
        if not d.get("cleaning_pipeline_hash"):
            return False, "missing_cleaning_lineage"
        if d.get("contamination_status") not in ("clean",):
            return False, "contaminated"
        if d.get("license") in (None, "", "unknown"):
            return False, "unsafe_license"
        return True, "admitted"

    def admit_or_block(self, shard_id: str, log, context: str = "") -> bool:
        ok, reason = self.admit(shard_id)
        if not ok:
            rec = {"shard_id": shard_id, "reason": reason, "context": context}
            self.blocked.append(rec)
            log.event("shard_blocked", **rec)
        return ok

    # ---- views --------------------------------------------------------------
    def trainable(self, lane: str | None = None, include_reserve: bool = False) -> list[str]:
        out = []
        for sid, d in self.manifests.items():
            if not self.admit(sid)[0]:
                continue
            if lane and d["lane"] != lane:
                continue
            if d.get("anneal_reserve") and not include_reserve:
                continue
            out.append(sid)
        return sorted(out)

    def reserve(self, lane: str | None = None) -> list[str]:
        return sorted(sid for sid, d in self.manifests.items()
                      if d.get("anneal_reserve") and self.admit(sid)[0]
                      and (lane is None or d["lane"] == lane))

    def get(self, shard_id: str) -> dict:
        return self.manifests[shard_id]
