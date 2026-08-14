"""E2 -- does storing mathematical structure in the embedding actually help?

E0 already proves the algebra is exact: compose(enc(9), enc(9)) decodes to 18
with no model at all. E2 asks the follow-on question -- given a transformer that
must read an answer out, does the numeric block help, and does it survive
operands longer than any it was trained on?

The first version of this experiment ANSWERED NO, and that answer is kept. With
a digit-token output every arm scored 0.000 at 4 and 5 digits, and the fair
firefly arm (0.832 / 0.005 / 0.000 on 1-3 digits) was *worse* than the plain
digit baseline (1.000 / 0.908 / 0.432). Exact structure on the input side buys
nothing while the readout still has to emit digits: the bottleneck simply moves
to the output, where carry propagation and digit counts come straight back.

So the grid is now 2x2 -- input representation crossed with output space --
because that is the only way to see which end is actually responsible.

  INPUT                            OUTPUT
  digit    digit tokens            digits     answer spelled out, autoregressive
  abacus   digit tokens + Abacus    residues   12 residue heads, then CRT
           positions (2405.17399)              (CRT itself has no parameters)
  firefly  one token per operand,
           carrying the 48-d block
  firefly+op  ...plus compose(a,b), the answer's own block, from the frozen
              parameter-free operator. Labelled clearly: this is the thesis
              stated as an experiment, not a baseline.

Train on 1-3 digit operands (digit count sampled uniformly). Test to 5.
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "e2_arithmetic.json"

D0, PLUS, STAR, EQ, PAD, BOS, EOS, NUM = 0, 10, 11, 12, 13, 14, 15, 16
VOCAB = 17
MAXLEN = 24
FEAT_DIM = 2 * N.N_NUMERIC_DIMS          # operand block + composed block

TRAIN_DIGITS = (1, 3)
TEST_DIGITS = [1, 2, 3, 4, 5]
N_TRAIN = 30_000
TRAIN_STEPS = 2000
BATCH = 64
D_MODEL, N_LAYER, N_HEAD = 96, 3, 4
SEED = 0

ARMS = [
    ("digit", "digits"), ("abacus", "digits"),
    ("firefly", "digits"), ("firefly+op", "digits"),
    ("digit", "residues"), ("firefly", "residues"), ("firefly+op", "residues"),
]


def _digits(n):
    return [int(c) for c in str(int(n))]


def encode_sample(a, b, r, arm, out_space, op, rng):
    nb = N.N_NUMERIC_DIMS
    opt = PLUS if op == "+" else STAR
    if arm.startswith("firefly"):
        prompt = [BOS, NUM, opt, NUM, EQ]
    else:
        prompt = [BOS] + _digits(a) + [opt] + _digits(b) + [EQ]

    if out_space == "digits":
        ans = _digits(r)
        ids = prompt + ans + [EOS]
        mask = [0] * len(prompt) + [1] * (len(ans) + 1)
    else:
        ids, mask = list(prompt), [0] * len(prompt)
    pad = MAXLEN - len(ids)
    ids, mask = ids + [PAD] * pad, mask + [0] * pad

    feats = np.zeros((MAXLEN, FEAT_DIM), dtype=np.float32)
    if arm.startswith("firefly"):
        feats[1, :nb] = N.numeric_block(a)
        feats[3, :nb] = N.numeric_block(b)
        if arm == "firefly+op":
            zero = np.zeros(len(N.PRIMES), dtype=np.complex128)
            if op == "+":
                z, u = N.compose(N.add_encode(a), N.add_encode(b)), zero
            else:
                z, u = zero, N.compose(N.mul_encode(a), N.mul_encode(b))
            feats[4, nb:] = np.concatenate([z.real, z.imag, u.real, u.imag])

    if arm == "abacus":
        pos = np.zeros(MAXLEN, dtype=np.int64)
        off = int(rng.integers(0, 8))
        i = 1
        groups = [_digits(a), _digits(b)] + ([_digits(r)] if out_space == "digits" else [])
        for grp in groups:
            for j, _ in enumerate(grp):
                pos[i] = off + len(grp) - j
                i += 1
            i += 1
        pos = np.clip(pos, 0, MAXLEN - 1)
    else:
        pos = np.arange(MAXLEN, dtype=np.int64)

    return (np.array(ids, np.int64), feats, pos, np.array(mask, np.float32),
            N.residues(r), len(prompt))


def build(n, lo, hi, arm, out_space, op, seed):
    rng = np.random.default_rng(seed)
    a, b, r = data.arithmetic(n, lo, hi, op, seed)
    o = [encode_sample(int(x), int(y), int(z), arm, out_space, op, rng)
         for x, y, z in zip(a, b, r)]
    return (np.stack([q[0] for q in o]), np.stack([q[1] for q in o]),
            np.stack([q[2] for q in o]), np.stack([q[3] for q in o]),
            np.stack([q[4] for q in o]), o[0][5], r)


# --------------------------------------------------------------------------
def accuracy(model, arm, out_space, digits, op, n=1000, seed=99):
    ids, feats, pos, mask, res, prompt_len, truth = build(
        n, digits, digits, arm, out_space, op, seed)
    model.eval()
    ok = 0
    with torch.no_grad():
        if out_space == "residues":
            for i in range(0, n, 500):
                logits = model(torch.from_numpy(ids[i:i + 500, :prompt_len]),
                               torch.from_numpy(feats[i:i + 500, :prompt_len]),
                               torch.from_numpy(pos[i:i + 500, :prompt_len]))
                pred = np.stack([lg.argmax(-1).numpy() for lg in logits], 1)
                for row, t in zip(pred, truth[i:i + 500]):
                    ok += N.crt(row) == int(t) % N.MODULUS
        else:
            answer_len = int(mask.sum(axis=1).max())
            for i in range(0, n, 500):
                f = torch.from_numpy(feats[i:i + 500])
                p = torch.from_numpy(pos[i:i + 500])
                tgt = torch.from_numpy(ids[i:i + 500])
                cur = tgt[:, :prompt_len].clone()
                for _ in range(answer_len):
                    nxt = model(cur, f[:, :cur.shape[1]],
                                p[:, :cur.shape[1]])[:, -1].argmax(-1)
                    cur = torch.cat([cur, nxt[:, None]], dim=1)
                gen = cur[:, prompt_len:]
                ref = tgt[:, prompt_len:prompt_len + answer_len]
                keep = torch.from_numpy(
                    mask[i:i + 500, prompt_len:prompt_len + answer_len]) > 0
                ok += int(((gen == ref) | ~keep).all(dim=1).sum())
    return ok / n


def train_arm(arm, out_space, op, log=print):
    torch.manual_seed(SEED)
    ids, feats, pos, mask, res, prompt_len, _ = build(
        N_TRAIN, *TRAIN_DIGITS, arm, out_space, op, SEED)
    feat_dim = FEAT_DIM if arm.startswith("firefly") else 0
    body = TinyTransformer(VOCAB, D_MODEL, N_LAYER, N_HEAD, MAXLEN,
                           feat_dim=feat_dim, n_positions=MAXLEN)
    model = (ResidueReadout(body, N.PRIMES, D_MODEL) if out_space == "residues"
             else body)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    rng = np.random.default_rng(SEED)
    model.train()
    t0 = time.time()
    for step in range(TRAIN_STEPS):
        i = rng.choice(len(ids), BATCH)
        x, f = torch.from_numpy(ids[i]), torch.from_numpy(feats[i])
        p = torch.from_numpy(pos[i])
        if out_space == "residues":
            logits = model(x[:, :prompt_len], f[:, :prompt_len], p[:, :prompt_len])
            y = torch.from_numpy(res[i])
            loss = sum(F.cross_entropy(lg, y[:, j]) for j, lg in enumerate(logits))
            loss = loss / len(logits)
        else:
            m = torch.from_numpy(mask[i])
            lg = model(x[:, :-1], f[:, :-1], p[:, :-1])
            lp = F.cross_entropy(lg.reshape(-1, VOCAB), x[:, 1:].reshape(-1),
                                 reduction="none").reshape(x.shape[0], -1)
            loss = (lp * m[:, 1:]).sum() / m[:, 1:].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    acc = {str(d): accuracy(model, arm, out_space, d, op) for d in TEST_DIGITS}
    name = f"{arm}->{out_space}"
    log(f"    {name:22s} " + " ".join(f"{d}d:{acc[str(d)]:.3f}" for d in TEST_DIGITS)
        + f"   ({time.time()-t0:.0f}s)")
    return {"accuracy_by_digits": acc, "params": model.n_params(),
            "seconds": round(time.time() - t0, 1),
            "input": arm, "output": out_space}


# --------------------------------------------------------------------------
def run(log=print) -> dict:
    out = {"train_digits": list(TRAIN_DIGITS), "test_digits": TEST_DIGITS,
           "n_train": N_TRAIN, "steps": TRAIN_STEPS,
           "primes": list(N.PRIMES), "modulus": N.MODULUS, "ops": {}}
    for op in ["+", "*"]:
        log(f"  operation '{op}'  (train {TRAIN_DIGITS[0]}-{TRAIN_DIGITS[1]} digits, "
            f"digit count sampled uniformly)")
        out["ops"][op] = {f"{a}->{o}": train_arm(a, o, op, log) for a, o in ARMS}
    _figure(out)
    data.save_json(out, RESULTS)
    add = out["ops"]["+"]
    best_d = max(add[k]["accuracy_by_digits"]["5"] for k in add if k.endswith("digits"))
    best_r = max(add[k]["accuracy_by_digits"]["5"] for k in add if k.endswith("residues"))
    out["readout_is_the_bottleneck"] = bool(best_r > best_d + 0.2)
    data.save_json(out, RESULTS)
    log(f"[PASS] e2_arithmetic  5-digit addition, best digit-readout {best_d:.3f} "
        f"vs best residue-readout {best_r:.3f}")
    return out


def _figure(out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colours = {"digit->digits": plots.KRON, "abacus->digits": plots.WARN,
               "firefly->digits": "#8d95ab", "firefly+op->digits": "#5a44c8",
               "digit->residues": "#2a9d8f", "firefly->residues": plots.FIREFLY,
               "firefly+op->residues": plots.OK}
    for ax, op, title in zip(axes, ["+", "*"], ["addition", "multiplication"]):
        for name, res in out["ops"][op].items():
            a = res["accuracy_by_digits"]
            ax.plot(out["test_digits"], [a[str(d)] for d in out["test_digits"]],
                    "o-" if name.endswith("residues") else "s--",
                    color=colours.get(name, "#888"), lw=2, ms=5, label=name)
        ax.axvspan(out["train_digits"][0] - 0.4, out["train_digits"][1] + 0.4,
                   color=plots.GRID, alpha=0.5, zorder=0)
        ax.set_xlabel("operand digits"); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(out["test_digits"])
        ax.set_title(title, color=plots.FG, fontsize=10)
    axes[0].set_ylabel("exact answer accuracy")
    axes[0].legend(fontsize=7, loc="center left")
    fig.suptitle("shaded = digit lengths seen in training;  solid = residue readout",
                 color=plots.FG, fontsize=9, y=1.02)
    plots.save(fig, "e2_arithmetic.png")


if __name__ == "__main__":
    run()
