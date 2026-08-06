"""Training loop, checkpointing and crash recovery.

A checkpoint here stores model + optimizer + RNG **and the consumption-ledger
offset**. On resume, the ledger is rolled back to that offset, so microbatches
written after the last checkpoint but before the crash are not double-counted,
and the stream is recomputed from the step recorded in the checkpoint.

Because the stream is a pure function of (seed, branch, step, micro), resuming
cannot skip or repeat a batch: the batch at a given step is the same object no
matter which process computes it.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import config, mixture
from .hashing import canonical_json
from .ledger import Ledgers
from .model import TinyLM
from .packing import LOSS_IGNORE
from .stream import StreamBuilder


def checkpoint_path(step: int, branch: str = "main") -> Path:
    return config.CHECKPOINTS / f"ckpt_{branch}_step{step:05d}.pt"


def save_checkpoint(model, opt, step: int, ledgers: Ledgers, log, branch: str = "main",
                    perf: dict | None = None) -> str:
    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ckpt_id = f"ckpt_{branch}_step{step:05d}"
    payload = {
        "checkpoint_id": ckpt_id, "global_step": step, "branch_id": branch,
        "run_id": ledgers.run_id,
        "consumption_offset": ledgers.consumption.count(),
        "learning_offset": ledgers.learning.count(),
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "seed": config.SEED,
        # Throughput counters are run state, not a side effect. A crashed
        # process never flushes them, so they are checkpointed alongside the
        # ledger offset and roll back with it -- which is what keeps the
        # performance report reconcilable against the consumption ledger.
        "perf": dict(perf or {}),
    }
    torch.save(payload, checkpoint_path(step, branch))
    meta = {k: v for k, v in payload.items()
            if k not in ("model", "optimizer", "torch_rng")}
    meta["perf"] = {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in meta.get("perf", {}).items()}
    (config.CHECKPOINTS / f"{ckpt_id}.json").write_text(canonical_json(meta), encoding="utf-8")
    log.check("checkpoint_saved", checkpoint_path(step, branch).exists(),
              checkpoint_id=ckpt_id, step=step,
              consumption_offset=meta["consumption_offset"])
    _prune_weights(branch)
    return ckpt_id


def _prune_weights(branch: str) -> None:
    """Keep every checkpoint's .json metadata (it is the evidential record that
    binds model state to a ledger offset) but retain only the most recent few
    .pt weight files, which are large and not needed to audit the run."""
    pts = sorted(config.CHECKPOINTS.glob(f"ckpt_{branch}_step*.pt"))
    for old in pts[:-config.KEEP_CHECKPOINT_WEIGHTS]:
        old.unlink(missing_ok=True)


def load_checkpoint(step: int, branch: str = "main") -> dict:
    return torch.load(checkpoint_path(step, branch), map_location="cpu", weights_only=False)


def latest_checkpoint(branch: str = "main") -> dict | None:
    metas = sorted(config.CHECKPOINTS.glob(f"ckpt_{branch}_step*.json"))
    if not metas:
        return None
    return json.loads(metas[-1].read_text(encoding="utf-8"))


class Trainer:
    def __init__(self, gate, registry, opus_engine, log, run_id: str,
                 branch_id: str = "main", ledger_suffix: str = ""):
        self.gate, self.reg, self.opus, self.log = gate, registry, opus_engine, log
        self.branch_id = branch_id
        self.ledgers = Ledgers(run_id, branch_id, ledger_suffix)
        self.stream = StreamBuilder(registry, gate, opus_engine,
                                    seed=config.SEED, branch_id=branch_id)
        self.model = TinyLM(gate.vocab_size)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE,
                                     betas=(0.9, 0.95), weight_decay=0.1)
        self.perf = {"batches": 0, "raw_tokens": 0, "loss_tokens": 0,
                     "pad_tokens": 0, "train_seconds": 0.0, "loader_seconds": 0.0,
                     "utilization_sum": 0.0}
        self.last_ckpt: str | None = None

    # ---- restore ------------------------------------------------------------
    def restore(self, step: int, branch: str | None = None, roll_back_ledger: bool = True) -> int:
        ck = load_checkpoint(step, branch or self.branch_id)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        self.last_ckpt = ck["checkpoint_id"]
        if ck.get("perf"):
            self.perf = dict(ck["perf"])          # counters roll back with the ledger
        if roll_back_ledger:
            # discard anything written after the checkpoint: those microbatches
            # were consumed by a process that then died, and will be produced
            # again by the resumed run.
            self.ledgers.consumption.truncate_to(ck["consumption_offset"])
            self.ledgers.learning.truncate_to(ck["learning_offset"])
        self.log.event("checkpoint_restored", checkpoint_id=ck["checkpoint_id"],
                       step=ck["global_step"], consumption_offset=ck["consumption_offset"],
                       branch=ck["branch_id"])
        return int(ck["global_step"])

    # ---- one optimizer step -------------------------------------------------
    def train_step(self, step: int) -> dict:
        phase = ("early" if step < config.TOTAL_STEPS * 0.4 else
                 "mid" if step < config.TOTAL_STEPS * 0.75 else
                 "anneal" if mixture.compile_step(step, 1).is_anneal else "late")
        self.opt.zero_grad(set_to_none=True)
        losses = []
        for micro in range(config.GRAD_ACCUM):
            t_load = time.perf_counter()
            batch, picks = self.stream.build(step, micro, self.log)
            self.perf["loader_seconds"] += time.perf_counter() - t_load
            if not batch.seqs:
                continue
            ids, labels, seg, pos = batch.tensors()
            t_tr = time.perf_counter()
            loss, per_tok, mask = self.model(
                torch.from_numpy(ids), torch.from_numpy(labels),
                torch.from_numpy(seg), torch.from_numpy(pos))
            (loss / config.GRAD_ACCUM).backward()
            self.perf["train_seconds"] += time.perf_counter() - t_tr

            self.ledgers.record_batch(step=step, micro=micro, batch=batch, picks=picks,
                                      plan=mixture.compile_step(step, config.MICRO_BATCH),
                                      checkpoint_id=self.last_ckpt,
                                      tokenizer_hash=self.gate.hash)
            self.ledgers.record_learning(step=step, micro=micro, picks=picks,
                                         per_token_loss=per_tok, loss_mask=mask,
                                         batch=batch, model_phase=phase)
            losses.append(float(loss))
            self.perf["batches"] += 1
            self.perf["raw_tokens"] += int(ids.size)
            self.perf["loss_tokens"] += int((labels != LOSS_IGNORE).sum())
            self.perf["pad_tokens"] += int((ids == 0).sum())
            self.perf["utilization_sum"] += batch.utilization
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return {"step": step, "loss": float(np.mean(losses)) if losses else 0.0}

    # ---- run ----------------------------------------------------------------
    def run(self, start_step: int, end_step: int, crash_at: int | None = None) -> None:
        for step in range(start_step, end_step):
            if crash_at is not None and step == crash_at:
                # state the expected next batch BEFORE dying, so the resumed run
                # can be checked against a claim made in advance
                nxt = self.stream.expected_batch_hash(step, 0, self.log)
                (config.ARTIFACTS / "expected_next_batch.json").write_text(canonical_json({
                    "expected_step": step, "expected_micro": 0,
                    "expected_batch_hash": nxt, "branch_id": self.branch_id,
                    "declared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }), encoding="utf-8")
                self.log.event("crash_simulated", step=step, expected_next_batch_hash=nxt[:16],
                               mode="hard_process_exit")
                self.log.event("process_exit", code=137)
                os._exit(137)          # real kill: no unwinding, no cleanup
            out = self.train_step(step)
            if step % 10 == 0:
                self.log.event("train_step", step=step, loss=round(out["loss"], 4),
                               stage=mixture.compile_step(step, 1).stage)
            if step > 0 and step % config.CHECKPOINT_EVERY == 0:
                self.last_ckpt = save_checkpoint(self.model, self.opt, step,
                                                 self.ledgers, self.log, self.branch_id,
                                                 perf=self.perf)

    def perf_counters(self) -> dict:
        return dict(self.perf)


def perf_report(counters: dict) -> dict:
    """Build the throughput report from raw counters.

    Kept as a free function because a run that crashed and resumed produces two
    sets of counters, and the report must describe the whole consumed stream --
    otherwise it cannot be reconciled against the consumption ledger.
    """
    secs = max(1e-9, counters["train_seconds"] + counters["loader_seconds"])
    b = max(1, counters["batches"])
    return {
        "batches": counters["batches"],
        "raw_tokens": counters["raw_tokens"],
        "loss_bearing_tokens": counters["loss_tokens"],
        "pad_tokens": counters["pad_tokens"],
        "wall_seconds": round(secs, 3),
        "loader_seconds": round(counters["loader_seconds"], 3),
        "train_seconds": round(counters["train_seconds"], 3),
        "raw_tokens_per_sec": round(counters["raw_tokens"] / secs, 1),
        "useful_loss_bearing_tokens_per_sec": round(counters["loss_tokens"] / secs, 1),
        "loss_bearing_fraction": round(counters["loss_tokens"] /
                                       max(1, counters["raw_tokens"]), 6),
        "mean_packing_utilization": round(counters["utilization_sum"] / b, 6),
        "loader_share_of_wall": round(counters["loader_seconds"] / secs, 6),
    }


