"""E3 -- the control. Text language modelling, FIREFLY against Kronecker.

PRE-REGISTERED PREDICTION: a tie.

This is not hedging, it is a consequence of the algebra. For tokens of at most
d_p bytes, Phi = kappa @ F for a fixed matrix F (proved in tests/, and F is
returned by encoder.fourier_from_kronecker_matrix). F is full rank, so a learned
projection D -> d_model can absorb it exactly. The two codecs therefore carry
identical information about short tokens, and no difference in validation loss
should survive.

Reporting this arm matters precisely because it should show nothing. Every real
FIREFLY gain lives where F does not apply -- length past the cap, 2-D and
continuous index domains, magnitude features, the numeric block -- and a paper
that also claimed a win on ordinary short-token English would be claiming
something its own mathematics forbids.

The arms share one body, one optimiser, one data order. Only the frozen codebook
differs. A learned embedding table is included as a reference point.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firefly import data
from firefly import encoder as E
from firefly import plots
from firefly.model import LMWithFrozenEmbedding
from firefly.plots import plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "e3_textlm.json"

VOCAB = 8192
SEQ = 96
D_MODEL, N_LAYER, N_HEAD = 160, 3, 4
STEPS = 900
BATCH = 16
EVAL_EVERY = 150
SEEDS = [0, 1]
ARMS = ["learned", "kronecker", "firefly"]


def codebook(arm: str) -> np.ndarray | None:
    if arm == "learned":
        return None
    toks = data.token_vocab_bytes(VOCAB)
    fn = E.embed_kronecker if arm == "kronecker" else E.embed
    return np.stack([fn(t if t else b"\x00") for t in toks]).astype(np.float32)


def batches(ids: np.ndarray, rng, n: int):
    i = rng.integers(0, len(ids) - SEQ - 1, size=n)
    return np.stack([ids[j:j + SEQ + 1] for j in i]).astype(np.int64)


@torch.no_grad()
def evaluate(model, ids, n_batches: int = 24, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x = torch.from_numpy(batches(ids, rng, BATCH))
        tot += float(model.loss(x))
    model.train()
    return tot / n_batches


def train_arm(arm: str, seed: int, tr, va, log=print) -> dict:
    torch.manual_seed(seed)
    codes = codebook(arm)
    model = LMWithFrozenEmbedding(VOCAB, codes, D_MODEL, N_LAYER, N_HEAD, SEQ)
    if codes is not None:
        model.embed.codes.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(seed)
    curve, t0 = [], time.time()
    for step in range(STEPS):
        x = torch.from_numpy(batches(tr, rng, BATCH))
        loss = model.loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            v = evaluate(model, va)
            curve.append({"step": step, "val_loss": v})
            log(f"      {arm:10s} seed{seed} step {step:5d}  val {v:.4f}")
    final = evaluate(model, va, n_batches=64)
    n_emb = (model.embed.n_trainable() if codes is not None
             else model.tok.weight.numel())
    return {"final_val_loss": final, "curve": curve,
            "input_side_trainable_params": int(n_emb),
            "seconds": round(time.time() - t0, 1)}


def run(log=print) -> dict:
    tr = data.token_stream("train", VOCAB)
    va = data.token_stream("validation", VOCAB)
    log(f"  {len(tr):,} train tokens / {len(va):,} val tokens, vocab {VOCAB}")
    out = {"vocab": VOCAB, "seq": SEQ, "steps": STEPS, "seeds": SEEDS,
           "d_model": D_MODEL, "n_layer": N_LAYER,
           "prediction": "tie between kronecker and firefly (Phi = kappa @ F)",
           "arms": {}}
    for arm in ARMS:
        runs = [train_arm(arm, s, tr, va, log) for s in SEEDS]
        losses = [r["final_val_loss"] for r in runs]
        out["arms"][arm] = {
            "runs": runs,
            "mean_val_loss": float(np.mean(losses)),
            "std_val_loss": float(np.std(losses)),
            "input_side_trainable_params": runs[0]["input_side_trainable_params"],
        }
        log(f"    {arm:10s} val {np.mean(losses):.4f} +/- {np.std(losses):.4f}"
            f"   input-side params {runs[0]['input_side_trainable_params']:,}")

    k, c = out["arms"]["kronecker"], out["arms"]["firefly"]
    gap = c["mean_val_loss"] - k["mean_val_loss"]
    noise = float(np.hypot(k["std_val_loss"], c["std_val_loss"]))
    out["firefly_minus_kronecker"] = gap
    out["seed_noise"] = noise
    out["tie_confirmed"] = bool(abs(gap) <= max(noise, 0.02))
    data.save_json(out, RESULTS)
    _figure(out)
    log(f"[{'PASS' if out['tie_confirmed'] else 'NOTE'}] e3_textlm  "
        f"firefly - kronecker = {gap:+.4f} nats (seed noise {noise:.4f}) -> "
        f"{'tie as predicted' if out['tie_confirmed'] else 'NOT a tie'}")
    return out


def _figure(out: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colours = {"learned": plots.WARN, "kronecker": plots.KRON, "firefly": plots.FIREFLY}
    for arm in ARMS:
        curves = [[p["val_loss"] for p in r["curve"]] for r in out["arms"][arm]["runs"]]
        steps = [p["step"] for p in out["arms"][arm]["runs"][0]["curve"]]
        m = np.mean(curves, axis=0)
        ax.plot(steps, m, "-", color=colours[arm], lw=2,
                label=f"{arm}  ({out['arms'][arm]['mean_val_loss']:.3f})")
        ax.fill_between(steps, np.min(curves, 0), np.max(curves, 0),
                        color=colours[arm], alpha=0.18)
    ax.set_xlabel("step"); ax.set_ylabel("validation loss (nats)")
    ax.set_title("E3 control: a tie is the predicted result", color=plots.FG,
                 fontsize=10)
    ax.legend(fontsize=8)
    plots.save(fig, "e3_textlm.png")


if __name__ == "__main__":
    run()
