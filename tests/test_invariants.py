"""Invariant tests for the training-data execution system.

These test the properties the system claims, not the happy path:
  - the batch stream is a pure function of its coordinates
  - a different branch produces a different stream
  - packed documents cannot attend to each other
  - padding and tool observations never bear loss
  - the evaluation firewall refuses held-out data
  - realised lane shares match the compiled plan
  - ledger offsets round-trip

Run:  python -m pytest tests -q
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v5flow import config, mixture, shards                      # noqa: E402
from v5flow.hashing import JsonlLedger, hash_tokens             # noqa: E402
from v5flow.opus import Opus                                    # noqa: E402
from v5flow.packing import (LOSS_IGNORE, attention_allow,       # noqa: E402
                            pack_record)
from v5flow.registry import Registry                            # noqa: E402
from v5flow.runlog import RunLog                                # noqa: E402
from v5flow.stream import StreamBuilder                         # noqa: E402
from v5flow.tokenizer_gate import TokenizerGate                 # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.MANIFESTS.exists() or not any(config.MANIFESTS.glob("*.json")),
    reason="run `python run_demo.py` first to generate artifacts")


@pytest.fixture(scope="module")
def world():
    gate = TokenizerGate()
    reg = Registry(gate.hash)
    reg.load_from_disk()
    log = RunLog(path=config.ARTIFACTS / "test.log", echo=False)
    return gate, reg, log


# ---------------------------------------------------------------- determinism
def test_stream_is_pure_function(world):
    gate, reg, log = world
    a = StreamBuilder(reg, gate, Opus()).build(37, 1, log)[0].batch_hash()
    b = StreamBuilder(reg, gate, Opus()).build(37, 1, log)[0].batch_hash()
    assert a == b, "same coordinates must yield the same batch"


def test_different_step_differs(world):
    gate, reg, log = world
    sb = StreamBuilder(reg, gate, Opus())
    assert sb.build(37, 0, log)[0].batch_hash() != sb.build(38, 0, log)[0].batch_hash()


def test_fork_diverges(world):
    gate, reg, log = world
    main = StreamBuilder(reg, gate, Opus(), branch_id="main").build(37, 0, log)[0]
    fork = StreamBuilder(reg, gate, Opus(), branch_id="branch-b").build(37, 0, log)[0]
    assert main.batch_hash() != fork.batch_hash(), "a fork must change the stream"


def test_lane_salt_is_process_stable():
    """Guards the bug that broke replay: builtin hash() on str is randomised
    per process, so it must never appear in stream coordinates."""
    src = (Path(__file__).resolve().parent.parent / "v5flow" / "stream.py").read_text(encoding="utf-8")
    assert "hash(lane)" not in src


# ---------------------------------------------------------------------- masks
def test_no_cross_document_attention(world):
    gate, reg, log = world
    batch, _ = StreamBuilder(reg, gate, Opus()).build(12, 0, log)
    for s in batch.seqs:
        allow = attention_allow(s.segment_ids)
        different = s.segment_ids[:, None] != s.segment_ids[None, :]
        assert not (allow & different).any(), "packed documents must not see each other"


def test_attention_is_causal(world):
    gate, reg, log = world
    batch, _ = StreamBuilder(reg, gate, Opus()).build(12, 0, log)
    for s in batch.seqs:
        allow = attention_allow(s.segment_ids)
        assert not np.triu(allow, k=1).any(), "no token may attend to the future"


def test_padding_never_bears_loss(world):
    gate, reg, log = world
    for step in (3, 45, 100, 150):
        batch, _ = StreamBuilder(reg, gate, Opus()).build(step, 0, log)
        for s in batch.seqs:
            assert not ((s.input_ids == 0) & (s.labels != LOSS_IGNORE)).any()


def test_positions_reset_per_document(world):
    gate, reg, log = world
    batch, _ = StreamBuilder(reg, gate, Opus()).build(20, 0, log)
    for s in batch.seqs:
        for seg in np.unique(s.segment_ids):
            p = s.position_ids[s.segment_ids == seg]
            assert p[0] == 0, "each document starts at position 0"
            assert np.all(np.diff(p) == 1), "positions increase by one inside a document"


def test_tool_observations_never_bear_loss(world):
    """The agentic rule: training on a tool response teaches the model to
    invent tool results instead of calling the tool."""
    gate, reg, log = world
    checked = 0
    for sid in reg.trainable("agentic", include_reserve=True):
        for rec in shards.load_records(sid)[:6]:
            s = pack_record(rec, sid, config.SEQ_LEN, config.LOSS_POLICY["agentic"])
            for seg in rec["segments"]:
                if seg["role"] != "context":
                    continue
                lo, hi = seg["start"], min(seg["end"], config.SEQ_LEN)
                if hi > lo:
                    assert (s.labels[lo:hi] == LOSS_IGNORE).all()
            checked += 1
    assert checked > 0


# ------------------------------------------------------------------- firewall
def test_eval_shards_are_blocked(world):
    gate, reg, log = world
    evals = [sid for sid, m in reg.manifests.items() if m.get("never_train")]
    assert evals, "the registry must know evaluation data exists"
    for sid in evals:
        ok, reason = reg.admit(sid)
        assert not ok and "eval" in reason


def test_eval_data_never_reached_a_batch(world):
    gate, reg, log = world
    evals = {sid for sid, m in reg.manifests.items() if m.get("never_train")}
    cons = JsonlLedger(config.LEDGERS / "consumption.jsonl").read()
    assert cons, "run the demo first"
    for rec in cons:
        assert not (evals & set(rec["shard_ids"]))


def test_tampered_tokenizer_hash_is_rejected(world):
    gate, reg, log = world
    sid = reg.trainable("web")[0]
    original = reg.manifests[sid]["tokenizer_hash"]
    reg.manifests[sid]["tokenizer_hash"] = "deadbeef" * 8
    try:
        ok, reason = reg.admit(sid)
        assert not ok and reason == "tokenizer_hash_mismatch"
    finally:
        reg.manifests[sid]["tokenizer_hash"] = original


# -------------------------------------------------------------------- mixture
def test_quota_timeline_sums_to_batch():
    for q in mixture.quota_timeline(config.MICRO_BATCH):
        assert sum(q.values()) == config.MICRO_BATCH


def test_protected_lanes_are_served():
    """A 4% lane rounds to zero in an 8-sequence microbatch; the deficit
    carry-forward is what stops a protected lane from vanishing."""
    table = mixture.quota_timeline(config.MICRO_BATCH)
    for lane in config.PROTECTED_FLOORS:
        assert sum(q[lane] for q in table) > 0


def test_realised_mixture_tracks_plan():
    cons = JsonlLedger(config.LEDGERS / "consumption.jsonl").read()
    assert cons
    realised = {l: 0 for l in config.LANES}
    for r in cons:
        for lane in r["lanes"]:
            realised[lane] += 1
    tot = sum(realised.values())
    planned = {l: 0 for l in config.LANES}
    for q in mixture.quota_timeline(config.MICRO_BATCH):
        for l, v in q.items():
            planned[l] += v
    ptot = sum(planned.values())
    for l in config.LANES:
        assert abs(realised[l] / tot - planned[l] / ptot) < 0.08, f"lane {l} drifted"


# --------------------------------------------------------------------- shards
def test_shard_content_hashes_match_payloads(world):
    gate, reg, log = world
    checked = 0
    for sid, m in reg.manifests.items():
        if m["kind"] == "stream":
            p = config.SHARD_DIR / f"{sid}.npy"
            if p.exists():
                assert hash_tokens(np.load(p)) == m["content_hash"]
                checked += 1
    assert checked > 0


def test_every_manifest_has_full_lineage(world):
    gate, reg, log = world
    required = ("tokenizer_hash", "cleaning_pipeline_hash", "content_hash",
                "license", "provenance_tier", "dedup_status", "contamination_status")
    for sid, m in reg.manifests.items():
        for f in required:
            assert m.get(f) not in (None, ""), f"{sid} missing {f}"


# --------------------------------------------------------------------- ledger
def test_no_duplicate_microbatches():
    cons = JsonlLedger(config.LEDGERS / "consumption.jsonl").read()
    keys = [(r["global_step"], r["microbatch"]) for r in cons]
    assert len(keys) == len(set(keys)), "resume must not repeat a batch"


def test_no_missing_steps():
    cons = JsonlLedger(config.LEDGERS / "consumption.jsonl").read()
    steps = sorted({r["global_step"] for r in cons})
    assert steps == list(range(steps[0], steps[-1] + 1)), "resume must not skip a step"


def test_learning_records_resolve_to_real_shards(world):
    gate, reg, log = world
    learn = JsonlLedger(config.LEDGERS / "learning.jsonl").read()
    assert learn
    for r in learn[:400]:
        assert r["shard_id"] in reg.manifests


def test_evidence_bundle_has_no_failures():
    p = config.ARTIFACTS / "evidence.json"
    bundle = json.loads(p.read_text(encoding="utf-8"))
    failed = [r["requirement"] for r in bundle["requirements"] if r["result"] != "PASS"]
    assert not failed, f"failing requirements: {failed}"
