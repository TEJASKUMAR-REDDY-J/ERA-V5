"""OPUS selection with a full audit trail.

Session 5 describes OPUS as scoring each candidate batch against a *stable proxy
direction*. Stable is the important word: the proxy is a frozen direction, not
the live model, which is what lets the same candidate be scored identically on a
replay. The score here is genuinely computed from the candidate's own token
statistics -- nothing is drawn at random and no verdict is hardcoded.

Four verdicts, all recorded:
  accepted          score above the acceptance threshold
  deferred          near miss; clean data that may be useful to a later model
  rejected          below threshold, with the reason retained
  floor_override    OPUS said reject, a protected floor rescued it anyway

The rejections are the valuable part: a rejected Indic or agentic batch is
evidence that the proxy is biased against a scarce lane.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np

from . import config
from .hashing import hash_obj

PROXY_VERSION = "opus-proxy-v1"
_FEAT_DIM = 64


def _proxy_direction() -> np.ndarray:
    """Frozen unit vector. Derived from the proxy version string, so it is the
    same in every process and every replay."""
    seed = int(hash_obj({"proxy": PROXY_VERSION})[:8], 16)
    v = np.random.default_rng(seed).normal(size=_FEAT_DIM)
    return v / np.linalg.norm(v)


PROXY = _proxy_direction()

# Lane affinity of the proxy. Deliberately English/code-leaning to reproduce the
# bias Session 5 warns about: an English-heavy proxy undervalues Indic and
# agentic data, which is exactly what the protected floors exist to correct.
LANE_AFFINITY = {"web": 0.35, "code": 0.30, "indic": -0.22,
                 "reasoning": 0.05, "agentic": -0.28}


def features(tokens: np.ndarray) -> np.ndarray:
    """Cheap fixed-length signature of a candidate: a normalised histogram of
    token ids folded into _FEAT_DIM buckets, plus a diversity term."""
    t = np.asarray(tokens, dtype=np.int64)
    if t.size == 0:
        return np.zeros(_FEAT_DIM)
    h = np.bincount(t % _FEAT_DIM, minlength=_FEAT_DIM).astype(np.float64)
    h /= h.sum()
    uniq = np.unique(t).size / t.size
    h[0] += uniq * 0.1
    n = np.linalg.norm(h)
    return h / n if n else h


@dataclass
class Decision:
    candidate_id: str
    step: int
    lane: str
    shard_id: str
    stage: str
    proxy_version: str
    score: float
    threshold: float
    status: str                 # accepted | rejected | deferred | floor_override
    reason: str
    protected_floor_override: bool
    effective_tokens: int


class Opus:
    def __init__(self):
        self.decisions: list[Decision] = []

    def score(self, tokens: np.ndarray, lane: str, step: int) -> float:
        """Estimated utility of this candidate to the current stage."""
        f = features(tokens)
        base = float(np.dot(f, PROXY))
        # normalise the raw projection into a comparable band, then apply the
        # proxy's lane affinity and a mild stage-progress term
        s = 0.5 + 2.0 * base + LANE_AFFINITY.get(lane, 0.0)
        s += 0.05 * np.sin(step / 37.0)          # slow drift as training advances
        return float(np.clip(s, 0.0, 1.0))

    def judge(self, candidates: list[dict], plan, log) -> list[dict]:
        """Score every candidate, then keep the mixture quota intact per lane.
        A lane at or below its protected floor may not be emptied by the
        selector -- if the quota cannot be met from accepted candidates, the
        best rejected ones are rescued and marked floor_override."""
        for c in candidates:
            c["score"] = self.score(c["tokens_preview"], c["lane"], plan.step)
        scores = np.array([c["score"] for c in candidates]) if candidates else np.array([0.0])
        threshold = float(np.quantile(scores, config.OPUS_ACCEPT_QUANTILE))

        chosen: list[dict] = []
        for lane in config.LANES:
            need = plan.quota.get(lane, 0)
            if need <= 0:
                continue
            pool = sorted([c for c in candidates if c["lane"] == lane],
                          key=lambda c: -c["score"])
            protected = lane in config.PROTECTED_FLOORS
            taken = 0
            for c in pool:
                if taken >= need:
                    status, reason = ("deferred", "quota_filled") if \
                        c["score"] >= threshold - config.OPUS_DEFER_MARGIN else \
                        ("rejected", "low_proxy_utility")
                    self._record(c, plan, status, reason, False)
                    continue
                if c["score"] >= threshold:
                    self._record(c, plan, "accepted", "above_threshold", False)
                    chosen.append(c)
                    taken += 1
                elif protected:
                    # selector would drop it; the floor rescues it
                    self._record(c, plan, "floor_override", "protected_floor_rescue", True)
                    log.event("opus_floor_override", lane=lane, step=plan.step,
                              shard_id=c["shard_id"], score=round(c["score"], 4),
                              threshold=round(threshold, 4))
                    chosen.append(c)
                    taken += 1
                else:
                    self._record(c, plan, "rejected", "below_threshold", False)
            # a protected lane must never come back empty
            if taken < need and protected:
                for c in pool[taken:need]:
                    self._record(c, plan, "floor_override", "protected_floor_backfill", True)
                    chosen.append(c)
        return chosen

    def _record(self, c: dict, plan, status: str, reason: str, override: bool) -> None:
        d = Decision(
            candidate_id=c["candidate_id"], step=plan.step, lane=c["lane"],
            shard_id=c["shard_id"], stage=plan.stage, proxy_version=PROXY_VERSION,
            score=round(float(c["score"]), 6), threshold=0.0, status=status,
            reason=reason, protected_floor_override=override,
            effective_tokens=int(c.get("n_tokens", 0)))
        self.decisions.append(d)
        c["decision_id"] = d.candidate_id
        c["opus_status"] = status

    def summary(self) -> dict:
        out: dict = {"total": len(self.decisions), "by_status": {}, "by_lane": {}}
        for d in self.decisions:
            out["by_status"][d.status] = out["by_status"].get(d.status, 0) + 1
            lane = out["by_lane"].setdefault(d.lane, {})
            lane[d.status] = lane.get(d.status, 0) + 1
        return out

    def rows(self) -> list[dict]:
        return [asdict(d) for d in self.decisions]
