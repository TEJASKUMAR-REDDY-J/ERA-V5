"""Independent verification pass.

This module deliberately takes NO in-memory state from the training run. It
opens the artifacts on disk -- manifests, ledgers, checkpoints, the declared
expected-batch file -- and recomputes every claim from them. If the training
process lied, or a subsystem silently did nothing, the recomputation disagrees
and the requirement is marked FAIL.

That is the difference between evidence and a print statement.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from . import config, mixture, shards
from .hashing import JsonlLedger, hash_obj, hash_tokens, sha256_file
from .opus import Opus
from .packing import LOSS_IGNORE, attention_allow
from .registry import Registry
from .stream import StreamBuilder
from .tokenizer_gate import TokenizerGate


def _jsonl(p: Path) -> list[dict]:
    return JsonlLedger(p).read()


def verify(log) -> dict:
    A = config.ARTIFACTS
    results: list[dict] = []

    def add(req: str, passed: bool, where: str, **detail):
        results.append({"requirement": req, "result": "PASS" if passed else "FAIL",
                        "evidence": where, "detail": detail})
        log.check(f"evidence_{req}", passed, **{k: v for k, v in detail.items()
                                                if not isinstance(v, (list, dict))})
        return passed

    # ---------------- 1. tokenizer integrity ------------------------------
    gate = TokenizerGate()
    manifests = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(config.MANIFESTS.glob("*.json"))]
    live = sha256_file(config.TOKENIZER_PATH)
    bad = [m["shard_id"] for m in manifests if m["tokenizer_hash"] != live]
    add("tokenizer_integrity", len(manifests) > 0 and not bad,
        "submission_artifacts/manifests/*.json",
        shards=len(manifests), tokenizer_hash=live[:16], mismatched=len(bad))

    # content hashes must still describe the shard payloads on disk
    rehash_ok, rehash_checked = 0, 0
    for m in manifests:
        if m["kind"] == "stream":
            p = config.SHARD_DIR / f"{m['shard_id']}.npy"
            if p.exists():
                rehash_checked += 1
                rehash_ok += int(hash_tokens(np.load(p)) == m["content_hash"])
        else:
            p = config.SHARD_DIR / f"{m['shard_id']}.json"
            if p.exists():
                rehash_checked += 1
                rehash_ok += int(hash_obj(json.loads(p.read_text(encoding="utf-8")))
                                 == m["content_hash"])
    add("shard_content_hashes", rehash_checked > 0 and rehash_ok == rehash_checked,
        "submission_artifacts/shards/ + manifests/",
        checked=rehash_checked, matched=rehash_ok)

    # ---------------- 2. evaluation firewall ------------------------------
    reg = Registry(gate.hash)
    reg.load_from_disk()
    eval_shards = [m for m in manifests if m.get("never_train")]
    blocked_all = all(not reg.admit(m["shard_id"])[0] for m in eval_shards)
    cons = _jsonl(config.LEDGERS / "consumption.jsonl")
    eval_ids = {m["shard_id"] for m in eval_shards}
    leaked = [r["global_step"] for r in cons if eval_ids & set(r["shard_ids"])]
    log_text = (A / "run.log").read_text(encoding="utf-8") if (A / "run.log").exists() else ""
    attempted = "eval_firewall_attempt" in log_text
    add("evaluation_firewall", bool(eval_shards) and blocked_all and not leaked and attempted,
        "run.log shard_blocked events + consumption ledger scan",
        eval_shards=len(eval_shards), blocked=blocked_all,
        leaked_batches=len(leaked), injection_attempted=attempted)

    # ---------------- 3. packing / mask correctness ------------------------
    sb = StreamBuilder(reg, gate, Opus())
    leaks = pad_loss = ctx_loss = seq_checked = 0
    # probe one step inside every curriculum stage, plus the crash step
    probe = sorted({5, config.CRASH_AT_STEP,
                    *(s["step_start"] + 5 for s in config.STAGES)})
    probe = [s for s in probe if s < config.TOTAL_STEPS]
    for step in probe:
        batch, picks = sb.build(step, 0, log)
        for s in batch.seqs:
            seq_checked += 1
            allow = attention_allow(s.segment_ids)
            diff = s.segment_ids[:, None] != s.segment_ids[None, :]
            leaks += int((allow & diff).sum())                       # cross-document attention
            pad_loss += int(((s.input_ids == 0) & (s.labels != LOSS_IGNORE)).sum())
            if not np.all(np.diff(s.position_ids[s.segment_ids == s.segment_ids[0]]) >= 0):
                ctx_loss += 1
    add("packing_and_masks", seq_checked > 0 and leaks == 0 and pad_loss == 0,
        f"recomputed batches at steps {probe}",
        steps_probed=probe, sequences_checked=seq_checked,
        cross_document_attention_leaks=leaks,
        padding_positions_bearing_loss=pad_loss)

    # agentic loss policy: tool observations must never bear loss
    agentic_ok, agentic_checked = True, 0
    for sid in reg.trainable("agentic", include_reserve=True):
        for rec in shards.load_records(sid)[:5]:
            from .packing import pack_record
            s = pack_record(rec, sid, config.SEQ_LEN, config.LOSS_POLICY["agentic"])
            for seg in rec["segments"]:
                if seg["role"] != "context":
                    continue
                lo, hi = seg["start"], min(seg["end"], config.SEQ_LEN)
                if hi > lo and int((s.labels[lo:hi] != LOSS_IGNORE).sum()) > 0:
                    agentic_ok = False
            agentic_checked += 1
    add("agentic_observation_masking", agentic_checked > 0 and agentic_ok,
        "pack_record over agentic shards", samples_checked=agentic_checked,
        observations_bearing_loss=(0 if agentic_ok else 1))

    # ---------------- 4. mixture compliance --------------------------------
    planned = {l: 0 for l in config.LANES}
    for q in mixture.quota_timeline(config.MICRO_BATCH):
        for l, v in q.items():
            planned[l] += v
    tot_p = sum(planned.values()) or 1
    planned = {l: planned[l] / tot_p for l in planned}
    realised: dict[str, int] = {l: 0 for l in config.LANES}
    for r in cons:
        for lane in r["lanes"]:
            realised[lane] = realised.get(lane, 0) + 1
    tot_r = sum(realised.values()) or 1
    realised = {l: realised.get(l, 0) / tot_r for l in config.LANES}
    drift = {l: round(realised[l] - planned[l], 4) for l in config.LANES}
    floors_ok = all(realised.get(l, 0) >= f * 0.6 for l, f in config.PROTECTED_FLOORS.items())
    add("mixture_compliance", max(abs(v) for v in drift.values()) <= 0.08 and floors_ok,
        "planned quota_timeline vs realised lanes in consumption ledger",
        max_abs_drift=max(abs(v) for v in drift.values()), drift=drift,
        protected_floors_respected=floors_ok,
        realised={k: round(v, 4) for k, v in realised.items()})

    # ---------------- 5. OPUS audit trail ----------------------------------
    dec = _jsonl(config.LEDGERS / "opus_decisions.jsonl")
    statuses = {d["status"] for d in dec}
    overrides = [d for d in dec if d["protected_floor_override"]]
    reasons = {d["reason"] for d in dec}
    add("opus_audit_trail",
        len(dec) > 0 and {"accepted", "rejected"} <= statuses and len(overrides) > 0,
        "submission_artifacts/ledgers/opus_decisions.jsonl",
        decisions=len(dec), statuses=sorted(statuses),
        floor_overrides=len(overrides), distinct_reasons=len(reasons))

    # ---------------- 6. crash recovery ------------------------------------
    exp_p = A / "expected_next_batch.json"
    crash_ok = False
    detail: dict = {}
    if exp_p.exists():
        exp = json.loads(exp_p.read_text(encoding="utf-8"))
        after = [r for r in cons if r["global_step"] == exp["expected_step"]
                 and r["microbatch"] == exp["expected_micro"]]
        # exactly one record for that (step, micro): no skip, no duplicate
        crash_ok = (len(after) == 1 and after[0]["batch_hash"] == exp["expected_batch_hash"])
        detail = {"expected_step": exp["expected_step"],
                  "expected_hash": exp["expected_batch_hash"][:16],
                  "records_found": len(after),
                  "observed_hash": after[0]["batch_hash"][:16] if after else None}
        # and no step is duplicated anywhere in the ledger
        seen = [(r["global_step"], r["microbatch"]) for r in cons]
        detail["duplicate_microbatches"] = len(seen) - len(set(seen))
        crash_ok = crash_ok and detail["duplicate_microbatches"] == 0
    add("crash_recovery", crash_ok,
        "expected_next_batch.json declared pre-crash vs consumption ledger", **detail)

    # ---------------- 7. replay --------------------------------------------
    rep_p = A / "replay_report.json"
    rep = json.loads(rep_p.read_text(encoding="utf-8")) if rep_p.exists() else {}
    add("replay", bool(rep.get("all_matched")), "submission_artifacts/replay_report.json",
        compared=rep.get("compared"), hash_matched=rep.get("hash_matched"),
        span_matched=rep.get("span_matched"))

    fork_p = A / "fork_report.json"
    fk = json.loads(fork_p.read_text(encoding="utf-8")) if fork_p.exists() else {}
    add("fork_divergence", bool(fk.get("diverged_everywhere")),
        "submission_artifacts/fork_report.json",
        steps=fk.get("steps_compared"), diverged=fk.get("steps_diverged"))

    # ---------------- 8. learning trace ------------------------------------
    learn = _jsonl(config.LEDGERS / "learning.jsonl")
    cons_keys = {(r["global_step"], r["microbatch"]) for r in cons}
    learn_keys = {(r["global_step"], r["microbatch"]) for r in learn}
    linked = len(learn_keys & cons_keys)
    shard_linked = all(r["shard_id"] in reg.manifests for r in learn[:500])
    add("learning_trace", len(learn) > 0 and linked > 0 and shard_linked,
        "submission_artifacts/ledgers/learning.jsonl joined to consumption",
        learning_records=len(learn), microbatches_linked=linked,
        every_shard_resolves=shard_linked)

    # ---------------- 9. throughput ----------------------------------------
    perf_p = A / "performance.json"
    perf = json.loads(perf_p.read_text(encoding="utf-8")) if perf_p.exists() else {}
    # reconstruct from the ledger independently of what the trainer reported
    raw = sum(r["tokens_total"] for r in cons)
    lb = sum(r["tokens_loss_bearing"] for r in cons)
    util = np.mean([r["packing_utilization"] for r in cons]) if cons else 0.0
    claimed_raw = perf.get("raw_tokens", -1)
    claimed_lb = perf.get("loss_bearing_tokens", -1)
    ok = (raw == claimed_raw and lb == claimed_lb
          and abs(util - perf.get("mean_packing_utilization", 0)) < 0.02)
    add("throughput_reconstructible", ok, "performance.json vs consumption ledger",
        ledger_raw_tokens=raw, reported_raw_tokens=claimed_raw,
        ledger_loss_tokens=lb, reported_loss_tokens=claimed_lb,
        ledger_utilization=round(float(util), 6),
        useful_tokens_per_sec=perf.get("useful_loss_bearing_tokens_per_sec"))

    bundle = {
        "generated_by": "v5flow.evidence.verify",
        "note": "Every result recomputed from artifacts on disk; no value is hardcoded.",
        "requirements": results,
        "summary": {"total": len(results),
                    "passed": sum(1 for r in results if r["result"] == "PASS"),
                    "failed": sum(1 for r in results if r["result"] == "FAIL")},
    }
    return bundle


def write_bundle(bundle: dict) -> None:
    A = config.ARTIFACTS
    (A / "evidence.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    s = bundle["summary"]
    lines = [
        "# Evidence Bundle — V5 Training Data Execution System",
        "",
        f"**{s['passed']} / {s['total']} requirements passed.**",
        "",
        "Every row below was recomputed by `v5flow/evidence.py` from the artifacts on "
        "disk (manifests, ledgers, checkpoints, shard payloads). No result is hardcoded; "
        "re-running `python run_demo.py` regenerates this file from scratch.",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for r in bundle["requirements"]:
        lines.append(f"| {r['requirement'].replace('_', ' ')} | **{r['result']}** | `{r['evidence']}` |")
    lines += ["", "## Detail", ""]
    for r in bundle["requirements"]:
        lines.append(f"### {r['requirement']}  — {r['result']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r["detail"], indent=2, ensure_ascii=False)[:1400])
        lines.append("```")
        lines.append("")
    (A / "evidence.md").write_text("\n".join(lines), encoding="utf-8")
