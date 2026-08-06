# Evidence Bundle — V5 Training Data Execution System

**12 / 12 requirements passed.**

Every row below was recomputed by `v5flow/evidence.py` from the artifacts on disk (manifests, ledgers, checkpoints, shard payloads). No result is hardcoded; re-running `python run_demo.py` regenerates this file from scratch.

| Requirement | Result | Evidence |
|---|---|---|
| tokenizer integrity | **PASS** | `submission_artifacts/manifests/*.json` |
| shard content hashes | **PASS** | `submission_artifacts/shards/ + manifests/` |
| evaluation firewall | **PASS** | `run.log shard_blocked events + consumption ledger scan` |
| packing and masks | **PASS** | `recomputed batches at steps [5, 85, 90, 140]` |
| agentic observation masking | **PASS** | `pack_record over agentic shards` |
| mixture compliance | **PASS** | `planned quota_timeline vs realised lanes in consumption ledger` |
| opus audit trail | **PASS** | `submission_artifacts/ledgers/opus_decisions.jsonl` |
| crash recovery | **PASS** | `expected_next_batch.json declared pre-crash vs consumption ledger` |
| replay | **PASS** | `submission_artifacts/replay_report.json` |
| fork divergence | **PASS** | `submission_artifacts/fork_report.json` |
| learning trace | **PASS** | `submission_artifacts/ledgers/learning.jsonl joined to consumption` |
| throughput reconstructible | **PASS** | `performance.json vs consumption ledger` |

## Detail

### tokenizer_integrity  — PASS

```json
{
  "shards": 22,
  "tokenizer_hash": "358592476e854961",
  "mismatched": 0
}
```

### shard_content_hashes  — PASS

```json
{
  "checked": 22,
  "matched": 22
}
```

### evaluation_firewall  — PASS

```json
{
  "eval_shards": 1,
  "blocked": true,
  "leaked_batches": 0,
  "injection_attempted": true
}
```

### packing_and_masks  — PASS

```json
{
  "steps_probed": [
    5,
    85,
    90,
    140
  ],
  "sequences_checked": 32,
  "cross_document_attention_leaks": 0,
  "padding_positions_bearing_loss": 0
}
```

### agentic_observation_masking  — PASS

```json
{
  "samples_checked": 25,
  "observations_bearing_loss": 0
}
```

### mixture_compliance  — PASS

```json
{
  "max_abs_drift": 0.0,
  "drift": {
    "web": 0.0,
    "code": 0.0,
    "indic": 0.0,
    "reasoning": 0.0,
    "agentic": 0.0
  },
  "protected_floors_respected": true,
  "realised": {
    "web": 0.3586,
    "code": 0.2477,
    "indic": 0.1828,
    "reasoning": 0.1469,
    "agentic": 0.0641
  }
}
```

### opus_audit_trail  — PASS

```json
{
  "decisions": 4752,
  "statuses": [
    "accepted",
    "deferred",
    "floor_override",
    "rejected"
  ],
  "floor_overrides": 483,
  "distinct_reasons": 4
}
```

### crash_recovery  — PASS

```json
{
  "expected_step": 90,
  "expected_hash": "2bf8bad2184e8f2e",
  "records_found": 1,
  "observed_hash": "2bf8bad2184e8f2e",
  "duplicate_microbatches": 0
}
```

### replay  — PASS

```json
{
  "compared": 30,
  "hash_matched": 30,
  "span_matched": 30
}
```

### fork_divergence  — PASS

```json
{
  "steps": 3,
  "diverged": 3
}
```

### learning_trace  — PASS

```json
{
  "learning_records": 2462,
  "microbatches_linked": 320,
  "every_shard_resolves": true
}
```

### throughput_reconstructible  — PASS

```json
{
  "ledger_raw_tokens": 655360,
  "reported_raw_tokens": 655360,
  "ledger_loss_tokens": 564087,
  "reported_loss_tokens": 564087,
  "ledger_utilization": 0.980699,
  "useful_tokens_per_sec": 3313.0
}
```
