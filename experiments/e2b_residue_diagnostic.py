"""E2b -- why the residue readout did not rescue length generalisation either.

E2 tried two fixes and both failed. Digit outputs cap out at 3 digits; swapping
to a residue head + CRT did not change that. Before concluding anything about
representations, this file separates the two candidate explanations:

  (a) the heads cannot compute residues at all
  (b) the heads are individually fine, and EXACTNESS is what kills it --
      CRT needs all 12 residues simultaneously correct, so per-head accuracy p
      gives exact-integer accuracy p^12. At p = 0.95 that is already 0.54; at
      p = 0.80 it is 0.07.

These have opposite implications. Under (a) the representation is not being
read; under (b) it is being read fine and the failure is a compounding
requirement that no amount of representation fixes.

Trains one firefly->residues model on addition and reports per-head accuracy
alongside the joint. ~2 minutes.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firefly import data
from firefly import numeric as N
from firefly import plots
from firefly.model import ResidueReadout, TinyTransformer
from firefly.plots import plt

import experiments.e2_arithmetic as E2

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "e2b_residue_diagnostic.json"
STEPS = 2500
BATCH = 64


def run(log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    arm, op = "firefly", "+"
    ids, feats, pos, mask, res, prompt_len, _ = E2.build(
        30_000, 1, 3, arm, "residues", op, 0)
    body = TinyTransformer(E2.VOCAB, E2.D_MODEL, E2.N_LAYER, E2.N_HEAD, E2.MAXLEN,
                           feat_dim=E2.FEAT_DIM, n_positions=E2.MAXLEN)
    model = ResidueReadout(body, N.PRIMES, E2.D_MODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(0)
    model.train()
    for step in range(STEPS):
        i = rng.choice(len(ids), BATCH)
        logits = model(torch.from_numpy(ids[i][:, :prompt_len]),
                       torch.from_numpy(feats[i][:, :prompt_len]),
                       torch.from_numpy(pos[i][:, :prompt_len]))
        y = torch.from_numpy(res[i])
        loss = sum(F.cross_entropy(lg, y[:, j]) for j, lg in enumerate(logits))
        opt.zero_grad(set_to_none=True)
        (loss / len(logits)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    out = {"steps": STEPS, "primes": list(N.PRIMES), "by_digits": {}}
    model.eval()
    for d in E2.TEST_DIGITS:
        ids, feats, pos, mask, res, pl, truth = E2.build(
            1000, d, d, arm, "residues", op, 99)
        with torch.no_grad():
            logits = model(torch.from_numpy(ids[:, :pl]),
                           torch.from_numpy(feats[:, :pl]),
                           torch.from_numpy(pos[:, :pl]))
        per_head = [float((lg.argmax(-1).numpy() == res[:, j]).mean())
                    for j, lg in enumerate(logits)]
        pred = np.stack([lg.argmax(-1).numpy() for lg in logits], 1)
        joint = float(np.mean([N.crt(r) == int(t) % N.MODULUS
                               for r, t in zip(pred, truth)]))
        all_right = float(np.mean((pred == res).all(axis=1)))
        out["by_digits"][str(d)] = {
            "per_head": per_head,
            "mean_per_head": float(np.mean(per_head)),
            "min_per_head": float(np.min(per_head)),
            "product_of_heads": float(np.prod(per_head)),
            "all_heads_correct": all_right,
            "exact_integer": joint,
        }
        log(f"    {d}d  mean per-head {np.mean(per_head):.3f}  "
            f"min {np.min(per_head):.3f}  prod {np.prod(per_head):.4f}  "
            f"exact {joint:.4f}")

    tr = out["by_digits"]["3"]
    out["diagnosis"] = (
        "compounding" if tr["mean_per_head"] > 0.7 else "heads_cannot_read_residues")
    out["wall_seconds"] = round(time.time() - t0, 1)
    _figure(out)
    data.save_json(out, RESULTS)
    log(f"[PASS] e2b_residue_diagnostic  diagnosis: {out['diagnosis']}")
    return out


def _figure(out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ds = sorted(int(d) for d in out["by_digits"])
    ax = axes[0]
    ax.plot(ds, [out["by_digits"][str(d)]["mean_per_head"] for d in ds], "o-",
            color=plots.FIREFLY, lw=2, label="mean per-head residue accuracy")
    ax.plot(ds, [out["by_digits"][str(d)]["product_of_heads"] for d in ds], "s--",
            color=plots.WARN, lw=2, label="product of the 12 heads")
    ax.plot(ds, [out["by_digits"][str(d)]["exact_integer"] for d in ds], "^-",
            color=plots.KRON, lw=2, label="exact integer after CRT")
    ax.set_xlabel("operand digits"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(ds); ax.legend(fontsize=8)
    ax.set_title("per-head vs joint", color=plots.FG, fontsize=10)

    ax = axes[1]
    p = np.linspace(0.5, 1.0, 200)
    for k in (4, 8, 12):
        ax.plot(p, p ** k, lw=2, label=f"{k} heads")
    for d in ds:
        v = out["by_digits"][str(d)]
        ax.scatter([v["mean_per_head"]], [v["exact_integer"]], s=42,
                   color=plots.KRON, zorder=5)
    ax.set_xlabel("per-head accuracy p"); ax.set_ylabel("p^k  (all heads correct)")
    ax.legend(fontsize=8)
    ax.set_title("why exactness is brutal", color=plots.FG, fontsize=10)
    plots.save(fig, "e2b_residue_diagnostic.png")


if __name__ == "__main__":
    run()
