<h1 align="center">V5 Training Data Execution System</h1>

<p align="center"><b>Tejaskumar Reddy J</b> · ERA V5 · Session 6 — Building the Training Dataset</p>

<p align="center">
  <img alt="requirements" src="https://img.shields.io/badge/evidence-12%2F12%20PASS-24b36b">
  <img alt="tests" src="https://img.shields.io/badge/tests-21%20passing-4cc9f0">
  <img alt="runtime" src="https://img.shields.io/badge/one%20command-5--8%20min%20CPU-7c5cff">
  <img alt="offline" src="https://img.shields.io/badge/demo-fully%20offline-5a44c8">
</p>

A small but complete data plane that can **prove what it consumed, why it consumed it, what
the model learned from it, and reconstruct the run exactly**.

```bash
pip install -r requirements.txt
python run_demo.py
```

Regenerates `submission_artifacts/` end to end — shards, manifests, ledgers, checkpoints,
a real crash and resume, a replay, a fork, an audit, a throughput report and the evidence
bundle. No network, no arguments, no manual steps.

### Generated artifacts

| Artifact | What it holds |
|---|---|
| [`submission_artifacts/run.log`](submission_artifacts/run.log) | full event sequence with `[PASS]`/`[FAIL]` markers |
| [`submission_artifacts/evidence.json`](submission_artifacts/evidence.json) | machine-readable result + evidence pointer per requirement |
| [`submission_artifacts/evidence.md`](submission_artifacts/evidence.md) | human-readable requirement table |
| [`submission_artifacts/manifests/`](submission_artifacts/manifests) | one sealed manifest per shard (22) |
| [`submission_artifacts/ledgers/`](submission_artifacts/ledgers) | consumption, learning and OPUS-decision ledgers |
| [`submission_artifacts/checkpoints/`](submission_artifacts/checkpoints) | weights + metadata bound to ledger offsets |
| [`submission_artifacts/performance.json`](submission_artifacts/performance.json) | throughput, reconcilable against the ledger |
| `replay_report` · `fork_report` · `audit_report` · `feasibility` | supporting reports |

---

## The one design decision everything rests on

**The batch stream is a pure function of `(seed, branch_id, global_step, microbatch)`.**

There is no iterator position, no shuffle buffer, no worker state, no mutable consumed-set.
`v5flow/stream.py` derives each batch from a counter-based RNG keyed on those coordinates.

Three of the assignment's hardest requirements then stop being features and become
consequences of one property:

| Requirement | Why it holds |
|---|---|
| **Resume** must not skip or repeat a batch | the batch at step *N* is the same object in any process |
| **Replay** must reproduce ids, spans and hashes | recomputing the coordinates yields the identical batch |
| **Fork** must diverge from its parent | a different `branch_id` is a different point in the function's domain |

A stateful dataloader can only *assert* these. A pure function makes them true, and the
evidence pass verifies it rather than trusting it.

> This bit us during development, which is the best argument for it. The lane salt originally
> used Python's builtin `hash(lane)`, which is randomised per process. Everything passed
> inside one process and replay failed 40/40 across processes. `tests/test_invariants.py`
> now has a test asserting `hash(lane)` never reappears in `stream.py`.

## Pipeline

```
data/raw/*.jsonl            real corpora, vendored so the demo is offline
   │  clean  (S4 contract: NFC, invisibles, ghost markers, PII)
   │  encode (S2 contract: frozen 10k BPE, hash recorded)
   ▼
shards + manifests          immutable; content hash, tokenizer hash, cleaning hash
   │  admission gate + evaluation firewall
   ▼
mixture timeline            S5 stages/floors compiled into per-step integer quotas
   │  OPUS: accept · reject · defer · protected-floor override
   ▼
packing                     loss mask · block-diagonal attention · per-document positions
   ▼
training                    tiny LM; every microbatch ledgered
   │
   ├── consumption ledger   what was consumed (append-only JSONL)
   ├── learning ledger      what came of it (loss attributed back to shard + lane)
   └── checkpoints          model + optimizer + ledger offset + throughput counters
        │
        └── crash → resume → replay → fork → audit
```

## Design decisions worth defending

**1. Checkpoints bind model state to data position.** A checkpoint stores the consumption
ledger offset. On resume the ledger is truncated back to it, so microbatches written by a
process that then died are re-derived rather than double-counted. The checkpoint is written
*after* its step is ledgered, so resume starts at `step + 1` — the off-by-one here produced
duplicate microbatches until it was fixed, and a test now guards it.

**2. Throughput counters live in the checkpoint.** A crashed process never flushes its
metrics. Treating counters as run state means they roll back and re-accumulate in lockstep
with the ledger, which is what lets `performance.json` be reconciled against the ledger
exactly instead of approximately.

**3. Protected floors need deficit carry-forward.** A 4% lane in an 8-sequence microbatch
rounds to zero every step, silently emptying a protected lane. `mixture.quota_timeline`
carries the rounding deficit forward, so agentic is served every few steps and realised
shares land within **0.0007** of plan. The timeline is computed from config alone, so it
stays a pure function.

**4. OPUS scores, it does not roll dice.** Scores come from a candidate's own token
signature projected onto a frozen proxy direction, plus a lane affinity that is deliberately
English/code-leaning — reproducing the bias Session 5 warns about. The protected floors then
visibly rescue Indic, reasoning and agentic batches the selector wanted to drop; those
rescues are recorded as `floor_override` with a reason.

**5. Evidence is recomputed, never reported.** `v5flow/evidence.py` takes no in-memory state
from the run. It reopens manifests, ledgers, shard payloads and checkpoints from disk,
rebuilds batches, re-hashes shard contents, and recomputes every claim. If a subsystem
silently did nothing, the recomputation disagrees and the requirement fails.

**6. Packing policy follows data type.** Plain text is concatenated and chopped with EOS
boundaries; reasoning and agentic samples are structure-preserving, one per sequence, because
merging them would let unrelated samples leak through attention.

## The three masks

| Mask | Rule | Failure it prevents |
|---|---|---|
| loss | pad never; `context` segments never; `model` segments do | training on a tool observation teaches the model to invent tool results |
| attention | causal **and** same-document | packed documents attending across boundaries teaches unrelated text as natural continuation |
| position | resets to 0 at each document start | positions that lie about distance |

The agentic lane uses real `glaive-function-calling-v2` trajectories, where
`FUNCTION RESPONSE:` is a genuine tool observation — so the masking rule is exercised against
real data, not a synthetic stand-in.

## Corpus — all lanes are real published data

| Lane | Source | License | Role |
|---|---|---|---|
| web | `HuggingFaceFW/fineweb-edu` | ODC-By-1.0 | plain pretraining |
| code | `codeparrot/codeparrot-clean-valid` | Apache-2.0 | plain pretraining |
| indic | `ai4bharat/sangraha` verified tel+hin | CC-BY-4.0 | protected lane, Tier A |
| reasoning | `openai/gsm8k` train + `open-r1/OpenR1-Math-220k` | MIT / Apache-2.0 | short + long traces |
| agentic | `glaiveai/glaive-function-calling-v2` | Apache-2.0 | real tool-call trajectories |
| **eval** | `openai/gsm8k` **test** split | eval-only | `never_train`, firewall target |

The evaluation shard is the *test* split of the same benchmark whose *train* split feeds the
reasoning lane. That is a real train/test boundary for the firewall to hold, not a synthetic
placeholder. `tools/build_vendor_corpus.py` (author-side, needs network) produced
`data/raw/`; the demo itself never downloads anything.

## What the demo proves

```
[PASS] tokenizer_hash_verified            frozen tokenizer, hash on every manifest
[PASS] eval_shard_blocked                 held-out data actively submitted, refused
[PASS] tampered_tokenizer_hash_rejected   negative control: gate reads the manifest
[PASS] mixture_compiled                   160 steps of integer quotas
[PASS] checkpoint_saved                   bound to a ledger offset
[PASS] crash_simulated                    child process hard-killed (os._exit)
[PASS] resume_next_batch_matched          hash declared BEFORE the crash, matched after
[PASS] replay_hash_matched                40/40 batches, hashes and token spans
[PASS] fork_branch_diverged               new branch from an earlier checkpoint
[PASS] audit_completed                    which shards trained which interval
[PASS] performance_measured               useful loss-bearing tokens/sec
[PASS] evidence_bundle_generated          12/12, recomputed from disk
```

The crash is a real `os._exit(137)` inside a child process — not a caught exception — so
recovery is demonstrated rather than simulated. The expected next batch hash is written to
`expected_next_batch.json` *before* the process dies, and checked against the ledger after
the resumed process has produced it.

## Layout

```
run_demo.py                  one command, full demonstration
v5flow/
  config.py                  contracts inherited from Sessions 1-5
  hashing.py                 canonical hashing + append-only JSONL
  cleaning.py                S4 admission contract (its own hash goes on every manifest)
  tokenizer_gate.py          S2 frozen tokenizer + verification
  shards.py                  documents -> immutable shards + manifests
  registry.py                admission gate + evaluation firewall
  mixture.py                 S5 stages -> per-step quotas, floors, feasibility
  opus.py                    scoring, four verdicts, audit trail
  packing.py                 policies + the three masks
  stream.py                  the pure-function batch stream
  ledger.py                  consumption + learning ledgers
  model.py                   tiny mask-aware LM
  trainer.py                 loop, checkpoints, restore
  recovery.py                replay / fork / audit
  evidence.py                independent verification pass
tests/test_invariants.py     21 invariant tests
tools/build_vendor_corpus.py author-side corpus download (not in the demo path)
submission_artifacts/        generated: run.log, evidence.*, manifests, ledgers, checkpoints
```

## Tests

```bash
python -m pytest tests -q
```

21 tests covering stream purity, fork divergence, the process-stable-salt regression,
causal + block-diagonal attention, padding and tool-observation masking, position resets,
firewall enforcement, tokenizer tampering, quota sums, protected-lane service, realised-vs-
planned drift, shard content hashes, manifest lineage completeness, and the ledger having
no duplicate or missing microbatches.

## Known limits

- The model is deliberately tiny; loss values are not meaningful, only their attribution is.
- Deduplication status on manifests is asserted by the corpus builder, not recomputed here —
  Session 4 owns that stage and this system consumes its verdict.
- Single process, single rank. Ledger records carry a `rank` field so the schema extends to
  multi-rank, but no distributed behaviour is exercised.
- Throughput is measured on CPU, so absolute tokens/sec are not representative; the
  reconcilability of the number against the ledger is what is being demonstrated.
