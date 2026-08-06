#!/usr/bin/env python3
"""One command that runs the entire demonstration.

    python run_demo.py

Phases:
  1  build immutable shards + manifests from the vendored corpus
  2  register shards, prove the evaluation firewall blocks held-out data
  3  compile the S5 mixture into per-step quotas and check lane feasibility
  4  train, ledger every microbatch, checkpoint bound to a ledger offset
  5  CRASH: the trainer is a child process that is hard-killed mid-run
  6  resume from the checkpoint and prove the next batch is the declared one
  7  replay a historical interval and compare hashes and token spans
  8  fork a new branch from the same checkpoint and prove divergence
  9  audit an interval from the ledger alone
 10  measure throughput, then regenerate the evidence bundle from disk

The crash is a real process exit (os._exit inside the child), not a caught
exception, so recovery is demonstrated rather than simulated.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time


from v5flow import config, evidence, mixture, recovery, shards
from v5flow import trainer as trainer_mod
from v5flow.hashing import canonical_json
from v5flow.ledger import Ledgers
from v5flow.opus import Opus
from v5flow.registry import Registry
from v5flow.runlog import RunLog
from v5flow.tokenizer_gate import TokenizerGate
from v5flow.trainer import Trainer, latest_checkpoint

RUN_ID = "v5-s6-demo"


def fresh_artifacts() -> None:
    if config.ARTIFACTS.exists():
        shutil.rmtree(config.ARTIFACTS)
    for d in (config.ARTIFACTS, config.MANIFESTS, config.LEDGERS,
              config.CHECKPOINTS, config.SHARD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def build_world(log) -> tuple[TokenizerGate, Registry]:
    gate = TokenizerGate()
    log.check("tokenizer_hash_verified", len(gate.hash) == 64, **gate.fingerprint())
    ms = shards.build_all(gate, log)
    reg = Registry(gate.hash)
    for m in ms:
        reg.register(m, log)
    log.event("manifests_validated", shards=len(ms),
              tokens=sum(m.token_count for m in ms))
    return gate, reg


def firewall_drill(reg: Registry, log) -> None:
    """Actively try to admit held-out evaluation data. It must be refused."""
    log.section("EVALUATION FIREWALL")
    evals = [sid for sid, m in reg.manifests.items() if m.get("never_train")]
    log.event("eval_firewall_attempt", shard_ids=evals,
              note="deliberately submitting held-out shards for admission")
    blocked = []
    for sid in evals:
        ok, reason = reg.admit(sid)
        reg.admit_or_block(sid, log, context="firewall_drill")
        blocked.append(not ok)
    log.check("eval_shard_blocked", bool(evals) and all(blocked),
              attempted=len(evals), blocked=sum(blocked))

    # a tampered shard must also be refused: proves the gate reads the manifest
    victim = reg.trainable("web")[0]
    saved = reg.manifests[victim]["tokenizer_hash"]
    reg.manifests[victim]["tokenizer_hash"] = "0" * 64
    ok, reason = reg.admit(victim)
    log.check("tampered_tokenizer_hash_rejected", (not ok) and reason == "tokenizer_hash_mismatch",
              shard_id=victim, reason=reason)
    reg.manifests[victim]["tokenizer_hash"] = saved


def train_child(crash: bool, start: int, end: int) -> int:
    """Run the trainer in a child process so the crash can be a real kill."""
    cmd = [sys.executable, __file__, "--child", "--start", str(start), "--end", str(end)]
    if crash:
        cmd += ["--crash-at", str(config.CRASH_AT_STEP)]
    return subprocess.run(cmd, cwd=str(config.ROOT)).returncode


def child_main(args) -> int:
    log = RunLog(echo=True)
    gate = TokenizerGate()
    reg = Registry(gate.hash)
    reg.load_from_disk()

    # ---- fork: restore an EARLIER checkpoint onto a new branch --------------
    if args.fork_from is not None:
        tr = Trainer(gate, reg, Opus(), log, RUN_ID, branch_id=args.branch,
                     ledger_suffix=f"_{args.branch}")
        tr.restore(args.fork_from, branch="main", roll_back_ledger=False)
        log.event("fork_created", parent_branch="main", child_branch=args.branch,
                  forked_at_step=args.fork_from,
                  note="same weights, new branch id, separate ledger")
        tr.run(args.fork_from + 1, args.fork_from + 1 + args.fork_steps)
        tr.ledgers.record_opus(tr.opus.rows())
        return 0

    tr = Trainer(gate, reg, Opus(), log, RUN_ID)
    if args.start > 0:
        ck = latest_checkpoint("main")
        if ck:
            tr.restore(ck["global_step"])
            # The checkpoint is written *after* its step was consumed and
            # ledgered, so the resumed run must begin at the following step.
            # Restarting at ck.global_step would re-consume it and put a
            # duplicate microbatch in the ledger.
            args.start = ck["global_step"] + 1
            log.event("resume_from_checkpoint", checkpoint_id=ck["checkpoint_id"],
                      checkpoint_step=ck["global_step"], resuming_at_step=args.start)
    tr.run(args.start, args.end, crash_at=args.crash_at)
    tr.ledgers.record_opus(tr.opus.rows())
    # Counters were restored from the checkpoint, so these cover the entire
    # consumed stream (pre-crash portion included), not just this process.
    (config.ARTIFACTS / "performance.json").write_text(
        canonical_json(trainer_mod.perf_report(tr.perf_counters())), encoding="utf-8")
    (config.ARTIFACTS / "opus_summary.json").write_text(
        canonical_json(tr.opus.summary()), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=config.TOTAL_STEPS)
    ap.add_argument("--crash-at", type=int, default=None)
    ap.add_argument("--fork-from", type=int, default=None)
    ap.add_argument("--branch", type=str, default="main")
    ap.add_argument("--fork-steps", type=int, default=6)
    args = ap.parse_args()
    if args.child:
        return child_main(args)

    t0 = time.time()
    fresh_artifacts()
    log = RunLog(echo=True)
    log.section("V5 TRAINING DATA EXECUTION SYSTEM — FULL DEMONSTRATION")

    # --- 1/2 shards, manifests, firewall ---------------------------------
    log.section("SHARDS AND MANIFESTS")
    gate, reg = build_world(log)
    firewall_drill(reg, log)

    # --- 3 mixture -------------------------------------------------------
    log.section("MIXTURE COMPILATION")
    feas = mixture.feasibility(reg, log)
    (config.ARTIFACTS / "feasibility.json").write_text(canonical_json(feas), encoding="utf-8")
    tl = mixture.quota_timeline(config.MICRO_BATCH)
    log.check("mixture_compiled", len(tl) == config.TOTAL_STEPS,
              steps=len(tl), lanes=len(config.LANES),
              protected_floors=config.PROTECTED_FLOORS)

    # --- 4/5 train until the crash ---------------------------------------
    log.section(f"TRAINING (crash scheduled at step {config.CRASH_AT_STEP})")
    rc = train_child(crash=True, start=0, end=config.TOTAL_STEPS)
    log.event("child_exited", returncode=rc)
    log.check("crash_simulated", rc != 0, returncode=rc,
              mode="hard process exit, not an exception")

    # --- 6 resume --------------------------------------------------------
    log.section("RESUME")
    exp = json.loads((config.ARTIFACTS / "expected_next_batch.json").read_text(encoding="utf-8"))
    log.event("expected_next_batch_declared", step=exp["expected_step"],
              batch_hash=exp["expected_batch_hash"][:16])
    rc = train_child(crash=False, start=1, end=config.TOTAL_STEPS)
    log.event("child_exited", returncode=rc)

    led = Ledgers(RUN_ID, "main")
    rows = [r for r in led.consumption.iter()
            if r["global_step"] == exp["expected_step"] and r["microbatch"] == exp["expected_micro"]]
    seen = [(r["global_step"], r["microbatch"]) for r in led.consumption.iter()]
    dupes = len(seen) - len(set(seen))
    log.check("resume_next_batch_matched",
              len(rows) == 1 and rows[0]["batch_hash"] == exp["expected_batch_hash"] and dupes == 0,
              expected=exp["expected_batch_hash"][:16],
              observed=rows[0]["batch_hash"][:16] if rows else None,
              records_for_that_microbatch=len(rows), duplicate_microbatches=dupes)

    # --- 7 replay --------------------------------------------------------
    log.section("REPLAY")
    lo, hi = config.REPLAY_RANGE
    rep = recovery.replay_interval(reg, gate, led, lo, hi, log)
    (config.ARTIFACTS / "replay_report.json").write_text(canonical_json(rep), encoding="utf-8")
    log.check("replay_hash_matched", rep["all_matched"], **{
        k: rep[k] for k in ("compared", "hash_matched", "span_matched")})

    # --- 8 fork ----------------------------------------------------------
    log.section("FORK")
    # fork from an earlier checkpoint that still has weights on disk
    pts = sorted(config.CHECKPOINTS.glob("ckpt_main_step*.pt"))
    fork_step = int(pts[0].stem.split("step")[-1])
    log.event("fork_requested", parent_branch="main", child_branch="branch-b",
              from_checkpoint_step=fork_step)
    rc = subprocess.run([sys.executable, __file__, "--child", "--fork-from", str(fork_step),
                         "--branch", "branch-b", "--fork-steps", "6"],
                        cwd=str(config.ROOT)).returncode
    log.event("fork_child_exited", returncode=rc)

    fk = recovery.fork_divergence(reg, gate, "main", "branch-b",
                                  [fork_step + 1, fork_step + 3, fork_step + 5], log)
    fork_led = Ledgers(RUN_ID, "branch-b", "_branch-b")
    fork_rows = fork_led.consumption.read()
    main_rows = {(r["global_step"], r["microbatch"]): r["batch_hash"]
                 for r in led.consumption.iter()}
    differing = sum(1 for r in fork_rows
                    if main_rows.get((r["global_step"], r["microbatch"])) != r["batch_hash"])
    fk.update({"forked_from_checkpoint_step": fork_step,
               "fork_ledger_records": len(fork_rows),
               "records_differing_from_main": differing,
               "separate_ledger": "ledgers/consumption_branch-b.jsonl"})
    (config.ARTIFACTS / "fork_report.json").write_text(canonical_json(fk), encoding="utf-8")
    log.check("fork_branch_diverged",
              fk["diverged_everywhere"] and len(fork_rows) > 0 and differing == len(fork_rows),
              forked_at=fork_step, fork_records=len(fork_rows),
              differing_from_main=differing, steps_compared=fk["steps_compared"])

    # --- 9 audit ---------------------------------------------------------
    log.section("AUDIT")
    aud = recovery.audit_interval(led, lo, hi)
    (config.ARTIFACTS / "audit_report.json").write_text(canonical_json(aud), encoding="utf-8")
    log.check("audit_completed", aud["distinct_shards"] > 0,
              shards=aud["distinct_shards"], samples=aud["distinct_samples"],
              tokens_by_lane=aud["tokens_by_lane"])
    (config.ARTIFACTS / "shard_report_cards.json").write_text(
        canonical_json(led.shard_report_card()), encoding="utf-8")

    # --- 10 performance + evidence ---------------------------------------
    log.section("PERFORMANCE")
    perf = json.loads((config.ARTIFACTS / "performance.json").read_text(encoding="utf-8"))
    log.check("performance_measured", perf["useful_loss_bearing_tokens_per_sec"] > 0,
              raw_tokens_per_sec=perf["raw_tokens_per_sec"],
              useful_tokens_per_sec=perf["useful_loss_bearing_tokens_per_sec"],
              packing_utilization=perf["mean_packing_utilization"])

    log.section("EVIDENCE BUNDLE")
    bundle = evidence.verify(log)
    evidence.write_bundle(bundle)
    s = bundle["summary"]
    log.check("evidence_bundle_generated", s["failed"] == 0,
              n_passed=s["passed"], n_failed=s["failed"], n_total=s["total"])
    log.event("demo_complete", seconds=round(time.time() - t0, 1))
    print(f"\n{s['passed']}/{s['total']} requirements PASSED "
          f"in {time.time() - t0:.1f}s -> {config.ARTIFACTS}")
    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
