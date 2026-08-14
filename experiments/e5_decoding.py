"""E5 -- decoding bytes back out, with no output vocabulary and no parameters.

Problem 5 asked for a reverse of the forward map so the final head can be
dropped. The answer turned out to be linear algebra, not a model:

    L     = ( sum_c Phi[c,0] )^2                 length, from the code itself
    bytes = argmax_c  sqrt(L) . Phi . pinv(F)    exact whenever K >= L

Zero parameters. A 1M-token vocabulary costs nothing on the output side.

This file measures where that holds and where it breaks, and it deliberately
attacks its own claim in three ways:

  1. length      exactness is conditional on K >= L. Where is the cliff?
  2. noise       a real head PREDICTS the code, so the code arrives corrupted.
                 Gaussian noise is the friendly case and we also test the
                 unfriendly ones -- int8 quantisation and low-rank error, which
                 are what actually happens in a deployed model.
  3. baseline    a learned decoder was built, measured, and lost. Recorded here
                 so the negative result is not quietly dropped.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firefly import data
from firefly import encoder as E
from firefly import plots
from firefly.decoder import decode, freqs_uniform, infer_length
from firefly.plots import plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "e5_decoding.json"

N_TOKENS = 2000
K_SWEEP = [8, 12, 16, 24, 32, 48]
NOISE = [0.0, 0.01, 0.03, 0.1, 0.3]

# Measured before the learned decoder was deleted. Raw run: results/e5_nca.json.
# A ~630k-parameter NCA decoder, trained with noise and damage augmentation and
# a randomised lattice, against the six lines above.
NCA_RECORD = {
    "params": 628801,
    "wellposed_K24": {"matched_filter_exact": 1.000, "nca_exact": 0.742},
    "compressed_K10": {"matched_filter_byte": 0.991, "nca_byte": 0.977},
    "verdict": ("The learned decoder lost in every regime, including the "
                "underdetermined one built for it. With a uniform grid pinv IS a "
                "matched filter, which is the optimal estimator under additive "
                "white noise, so there was no error to correct. Removed."),
}


def token_set() -> list[bytes]:
    vocab = [t for t in data.token_vocab_bytes() if 1 <= len(t) <= 40]
    rng = np.random.default_rng(0)
    return [vocab[i] for i in rng.choice(len(vocab), min(N_TOKENS, len(vocab)),
                                         replace=False)]


def roundtrip(tokens, k, corrupt=None, rng=None) -> tuple[float, float]:
    om = freqs_uniform(k)
    ok = nb = nr = 0
    for t in tokens:
        c = E.encode_1d(t, om)
        if corrupt is not None:
            c = corrupt(c, rng)
        got = decode(c, om)
        ok += got == t
        nb += len(t)
        nr += sum(a == b for a, b in zip(got, t))
    return ok / len(tokens), nr / max(1, nb)


# ---- the three corruption models -----------------------------------------
def gaussian(sigma):
    def f(c, rng):
        s = sigma * np.abs(c).mean()
        return c + s * (rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape))
    return f


def int8_quant(c, rng):
    """What a deployed low-precision head actually does to a code."""
    scale = np.abs(c).max() / 127.0
    if scale == 0:
        return c
    return (np.round(c.real / scale) + 1j * np.round(c.imag / scale)) * scale


def low_rank(rank):
    """Structured error: a head's prediction lives in a low-dimensional subspace,
    so its residual is correlated across channels rather than white."""
    def f(c, rng):
        u = rng.standard_normal((c.shape[0], rank))
        v = rng.standard_normal((rank, c.shape[1]))
        e = (u @ v) / np.sqrt(rank)
        return c + 0.05 * np.abs(c).mean() * e
    return f


def run(log=print) -> dict:
    t0 = time.time()
    toks = token_set()
    log(f"  {len(toks)} held-out tokens, lengths {min(map(len,toks))}-{max(map(len,toks))}")

    out = {"n_tokens": len(toks), "params": 0, "nca_baseline": NCA_RECORD,
           "k_sweep": {}, "by_length": {}, "noise": {}, "structured_noise": {}}

    # ---- 1. exactness vs K ------------------------------------------------
    log("  exact round trip vs number of frequencies:")
    for k in K_SWEEP:
        ex, by = roundtrip(toks, k)
        out["k_sweep"][str(k)] = {"exact": ex, "byte": by}
        log(f"    K={k:3d}   exact {ex:.4f}   byte {by:.4f}")

    # ---- 2. where the K >= L condition bites ------------------------------
    # BPE tokens top out around 14 bytes, so a vocabulary alone never reaches the
    # cliff and the sweep would look flawlessly exact for the wrong reason. Use
    # real text spans of controlled length instead, which is also the regime a
    # vocabulary-free head would actually operate in.
    corpus = " ".join(data.wikitext("validation", max_docs=400)).encode("utf-8")
    rng = np.random.default_rng(3)
    by_len: dict[int, list[bytes]] = {}
    for ln in range(1, 41):
        starts = rng.integers(0, len(corpus) - ln - 1, size=120)
        by_len[ln] = [bytes(corpus[s:s + ln]) for s in starts]
    k_fixed = 16
    log(f"  with K={k_fixed} fixed, exactness by token length "
        f"(the cliff should be at L={k_fixed}):")
    for ln in sorted(by_len):
        group = by_len[ln]
        if len(group) < 10:
            continue
        ex, by = roundtrip(group, k_fixed)
        out["by_length"][str(ln)] = {"exact": ex, "byte": by, "n": len(group),
                                     "k_ge_l": ln <= k_fixed}
        if ln in (1, 8, 14, 16, 17, 20, 24, 32, 40):
            log(f"    L={ln:3d}  exact {ex:.4f}  {'K>=L' if ln <= k_fixed else 'K<L'}")

    # ---- 3. corrupted codes ------------------------------------------------
    log("  Gaussian noise on the code (relative to mean |Phi|):")
    for s in NOISE:
        rng = np.random.default_rng(1)
        ex, by = roundtrip(toks[:600], 24, gaussian(s) if s else None, rng)
        out["noise"][str(s)] = {"exact": ex, "byte": by}
        log(f"    sigma {s:4.2f}   exact {ex:.4f}   byte {by:.4f}")

    log("  structured corruption -- what a real head actually produces:")
    for name, fn in [("int8_quantisation", int8_quant),
                     ("low_rank_residual_r8", low_rank(8)),
                     ("low_rank_residual_r2", low_rank(2))]:
        rng = np.random.default_rng(2)
        ex, by = roundtrip(toks[:600], 24, fn, rng)
        out["structured_noise"][name] = {"exact": ex, "byte": by}
        log(f"    {name:24s} exact {ex:.4f}   byte {by:.4f}")

    # ---- 4. what it replaces ----------------------------------------------
    out["head_comparison"] = {
        "softmax_head_params_131k_vocab_4096_dmodel": 131072 * 4096,
        "firefly_decoder_params": 0,
        "length_inference_exact": all(
            infer_length(E.encode_1d(t, freqs_uniform(48))) == len(t)
            for t in toks[:400]),
    }
    out["wall_seconds"] = round(time.time() - t0, 1)
    _figure(out)
    data.save_json(out, RESULTS)
    k16 = out["k_sweep"]["16"]["exact"]
    log(f"[PASS] e5_decoding  K=16 exact {k16:.4f} with 0 parameters, "
        f"replacing a {131072*4096:,}-parameter head")
    return out


def _figure(out: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ks = [int(k) for k in out["k_sweep"]]
    ax.plot(ks, [out["k_sweep"][str(k)]["exact"] for k in ks], "o-",
            color=plots.FIREFLY, lw=2, label="exact token")
    ax.plot(ks, [out["k_sweep"][str(k)]["byte"] for k in ks], "s--",
            color=plots.FIREFLY2, lw=2, label="per byte")
    ax.set_xlabel("frequencies K"); ax.set_ylabel("round-trip accuracy")
    ax.set_title("zero-parameter decode", color=plots.FG, fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    lens = sorted(int(x) for x in out["by_length"])
    ok = [l for l in lens if out["by_length"][str(l)]["k_ge_l"]]
    bad = [l for l in lens if not out["by_length"][str(l)]["k_ge_l"]]
    for grp, col, lab in [(ok, plots.OK, "K ≥ L (exact)"),
                          (bad, plots.KRON, "K < L (underdetermined)")]:
        if grp:
            ax.plot(grp, [out["by_length"][str(l)]["byte"] for l in grp], "o-",
                    color=col, lw=2, label=lab)
    ax.axvline(16.5, color=plots.GRID, ls=":")
    ax.set_xlabel("token length (bytes), K = 16")
    ax.set_ylabel("byte accuracy")
    ax.set_title("the K ≥ L cliff", color=plots.FG, fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    names = list(out["structured_noise"]) + [f"gauss σ={k}" for k in out["noise"]]
    vals = ([out["structured_noise"][n]["byte"] for n in out["structured_noise"]]
            + [out["noise"][k]["byte"] for k in out["noise"]])
    cols = [plots.WARN] * len(out["structured_noise"]) + [plots.FIREFLY] * len(out["noise"])
    ax.barh(range(len(vals)), vals, color=cols)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    ax.set_xlabel("byte accuracy"); ax.set_xlim(0, 1.05)
    ax.set_title("corrupted codes", color=plots.FG, fontsize=10)
    plots.save(fig, "e5_decoding.png")


if __name__ == "__main__":
    run()
