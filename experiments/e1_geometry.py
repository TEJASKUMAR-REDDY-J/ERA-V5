"""E1 -- the geometry of the two codes, measured on a real vocabulary.

No training. This probe tests the defect the Kronecker Embeddings paper names in
its own limitations section: "Insertion/deletion shifts all following bytes,
reducing cosine more sharply." A delta position basis has no representation of
relative shift, so inserting one byte relocates every subsequent spike to a
coordinate that did not previously exist. FIREFLY turns the same operation into a
phase rotation, whose magnitude is invariant.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firefly import data
from firefly import encoder as E
from firefly import plots
from firefly.plots import plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "e1_geometry.json")

N_PROBES = 400
MAX_SHIFT = 6


def _codes(words):
    return (np.stack([E.embed_kronecker(w) for w in words]),
            np.stack([E.embed(w) for w in words]),
            np.stack([E.magnitude_code(w) for w in words]))


def _cos_rows(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(1)


def run(log=print) -> dict:
    vocab = [w for w in data.token_vocab_bytes() if 3 <= len(w) <= 12]
    rng = np.random.default_rng(0)
    probes = [vocab[i] for i in rng.choice(len(vocab), N_PROBES, replace=False)]

    base_k, base_c, base_m = _codes(probes)
    out = {"n_probes": len(probes), "insertion": {}, "substitution": {}, "suffix": {}}

    # ---- prefix insertion: the failure mode the source paper names ----------
    for d in range(MAX_SHIFT + 1):
        shifted = [b"_" * d + w for w in probes]
        k, c, m = _codes(shifted)
        out["insertion"][d] = {
            "kronecker": float(_cos_rows(base_k, k).mean()),
            "firefly_phase": float(_cos_rows(base_c, c).mean()),
            "firefly_magnitude": float(_cos_rows(base_m, m).mean()),
        }

    # ---- substitution: same length, so the delta basis should cope ---------
    for d in range(MAX_SHIFT + 1):
        sub = []
        for w in probes:
            a = bytearray(w)
            for i in rng.choice(len(w), min(d, len(w)), replace=False):
                a[i] = int(rng.integers(97, 123))
            sub.append(bytes(a))
        k, c, m = _codes(sub)
        out["substitution"][d] = {
            "kronecker": float(_cos_rows(base_k, k).mean()),
            "firefly_phase": float(_cos_rows(base_c, c).mean()),
            "firefly_magnitude": float(_cos_rows(base_m, m).mean()),
        }

    # ---- suffix growth: morphology (run -> runs -> running) ----------------
    for d in range(MAX_SHIFT + 1):
        suf = [w + b"s" * d for w in probes]
        k, c, m = _codes(suf)
        out["suffix"][d] = {
            "kronecker": float(_cos_rows(base_k, k).mean()),
            "firefly_phase": float(_cos_rows(base_c, c).mean()),
            "firefly_magnitude": float(_cos_rows(base_m, m).mean()),
        }

    # ---- retrieval: can a word find its own prefixed form? -----------------
    gallery = vocab[:3000]
    gk, gc, gm = _codes(gallery)
    ranks = {"kronecker": [], "firefly_phase": [], "firefly_magnitude": []}
    for w in probes[:150]:
        target = b"re" + w
        for name, gal, fn in [("kronecker", gk, E.embed_kronecker),
                              ("firefly_phase", gc, E.embed),
                              ("firefly_magnitude", gm, E.magnitude_code)]:
            q = fn(target)
            q = q / (np.linalg.norm(q) + 1e-12)
            g = gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-12)
            sims = g @ q
            true_i = gallery.index(w) if w in gallery else None
            if true_i is None:
                continue
            ranks[name].append(int((sims > sims[true_i]).sum()) + 1)
    out["prefixed_form_retrieval_median_rank"] = {
        k: (float(np.median(v)) if v else None) for k, v in ranks.items()}
    out["prefixed_form_retrieval_top10_rate"] = {
        k: (float(np.mean(np.array(v) <= 10)) if v else None) for k, v in ranks.items()}

    # ---- devil's advocate --------------------------------------------------
    # "The magnitude code is just a nonlinear feature. Bolt one onto Kronecker
    #  and you get the same thing -- Fourier is doing nothing special."
    #
    # The Kronecker architecture applies ONE LEARNED LINEAR MAP to its codec.
    # So the question is not whether some nonlinearity could recover shift
    # invariance, it is whether a *linear* map can -- because that is all the
    # projection is. Fit kappa -> |Phi| by least squares and see.
    tr, te = gallery[:2200], gallery[2200:3000]
    xk = np.stack([E.embed_kronecker(w) for w in tr])
    yk = np.stack([E.magnitude_code(w) for w in tr])
    sol, *_ = np.linalg.lstsq(xk, yk, rcond=None)
    xt = np.stack([E.embed_kronecker(w) for w in te])
    yt = np.stack([E.magnitude_code(w) for w in te])
    pred = xt @ sol
    ss_res = float(((yt - pred) ** 2).sum())
    ss_tot = float(((yt - yt.mean(0)) ** 2).sum())
    out["magnitude_from_kronecker_linear_r2"] = 1 - ss_res / ss_tot

    # R^2 turns out to be the WRONG question, and it nearly cost us the claim:
    # the fit reaches R2 ~ 0.93, which looks like the projection can manufacture
    # the magnitude feature after all. But R2 is measured on unshifted words, and
    # the property we actually want is invariance UNDER SHIFT. So apply the fitted
    # linear map to shifted inputs and re-run the insertion probe on its output.
    # kappa(shifted) is nearly orthogonal to anything the fit ever saw, so a
    # linear map has nothing to extrapolate from.
    probe_words = te[:400]
    recon_ins = {}
    for d in [0, 1, 2, 3]:
        base = np.stack([E.embed_kronecker(w) for w in probe_words]) @ sol
        shft = np.stack([E.embed_kronecker(b"_" * d + w) for w in probe_words]) @ sol
        true = np.stack([E.magnitude_code(b"_" * d + w) for w in probe_words])
        real = np.stack([E.magnitude_code(w) for w in probe_words])
        recon_ins[str(d)] = {
            "linear_reconstruction": float(_cos_rows(base, shft).mean()),
            "true_magnitude": float(_cos_rows(real, true).mean()),
        }
    out["magnitude_reconstruction_under_shift"] = recon_ins
    log("  ...and the reconstruction KEEPS the invariance: "
        + " ".join(f"d={d}:{v['linear_reconstruction']:.3f}(true {v['true_magnitude']:.3f})"
                   for d, v in recon_ins.items()))

    # The rebuttal failed, so here is why, stated as evidence rather than buried.
    # |Phi[c,:]| is the magnitude of a sum of unit phasors, one per occurrence of
    # byte c. If c occurs ONCE, that magnitude is 1 whatever the position -- so
    # for any string with no repeated byte, |Phi| is exactly the byte histogram,
    # which a linear map reads straight off kappa by summing its position axis.
    # The magnitude code therefore only carries information beyond a histogram
    # where bytes repeat, and its advantage over Kronecker is an advantage over
    # an UNTRAINED baseline -- but Kronecker is never deployed untrained.
    def _hist(w):
        h = np.zeros(256)
        for b in E.as_bytes(w):
            h[b] += 1
        return h / np.sqrt(max(1, len(E.as_bytes(w))))

    anagrams = [(b"abc", b"cba"), (b"listen", b"silent"), (b"abab", b"baba"),
                (b"aabb", b"abab"), (b"aaab", b"abaa")]
    out["magnitude_is_mostly_a_histogram"] = [
        {"a": a.decode(), "b": b.decode(),
         "magnitude_cos": E.cosine(E.magnitude_code(a), E.magnitude_code(b)),
         "histogram_cos": E.cosine(_hist(a), _hist(b)),
         "phase_cos": E.cosine(E.embed(a), E.embed(b)),
         "has_repeats": len(set(a)) < len(a)}
        for a, b in anagrams]
    no_rep = [r for r in out["magnitude_is_mostly_a_histogram"] if not r["has_repeats"]]
    out["magnitude_equals_histogram_when_no_byte_repeats"] = bool(
        all(abs(r["magnitude_cos"] - r["histogram_cos"]) < 1e-6 for r in no_rep))
    log(f"  |Phi| == byte histogram when no byte repeats: "
        f"{out['magnitude_equals_histogram_when_no_byte_repeats']} "
        f"-> the shift-invariance claim is RETRACTED as a Fourier-only property")
    # control: the phase code IS a fixed linear map of kappa (Phi = kappa @ F),
    # so the same fit should succeed almost perfectly. If it does not, the probe
    # is broken rather than the claim being supported.
    yp = np.stack([E.embed(w) for w in tr])
    solp, *_ = np.linalg.lstsq(xk, yp, rcond=None)
    ytp = np.stack([E.embed(w) for w in te])
    rp = ytp - xt @ solp
    out["phase_from_kronecker_linear_r2"] = 1 - float((rp ** 2).sum()) / float(
        ((ytp - ytp.mean(0)) ** 2).sum())
    log(f"  linear recoverability from the Kronecker code: "
        f"phase R2={out['phase_from_kronecker_linear_r2']:.4f} (control, should be ~1), "
        f"magnitude R2={out['magnitude_from_kronecker_linear_r2']:.4f}")

    # ---- the headline single example --------------------------------------
    out["run_vs_arun"] = {
        "kronecker": E.cosine(E.embed_kronecker("run"), E.embed_kronecker("arun")),
        "firefly_phase": E.cosine(E.embed("run"), E.embed("arun")),
        "firefly_magnitude": E.cosine(E.magnitude_code("run"), E.magnitude_code("arun")),
    }

    _figure(out)
    data.save_json(out, __import__("pathlib").Path(RESULTS))
    log(f"[PASS] e1_geometry  insertion@1 kron={out['insertion'][1]['kronecker']:.3f} "
        f"cicada_mag={out['insertion'][1]['firefly_magnitude']:.3f}")
    return out


def _figure(out: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [("insertion", "bytes inserted at front"),
              ("substitution", "bytes substituted"),
              ("suffix", "suffix bytes appended")]
    for ax, (key, xlabel) in zip(axes, panels):
        xs = sorted(int(k) for k in out[key])
        for name, colour, label in [
                ("kronecker", plots.KRON, "Kronecker (delta basis)"),
                ("firefly_phase", plots.FIREFLY2, "FIREFLY (phase)"),
                ("firefly_magnitude", plots.FIREFLY, "FIREFLY (magnitude)")]:
            ax.plot(xs, [out[key][str(x) if str(x) in out[key] else x][name]
                         for x in xs], "o-", color=colour, label=label, lw=2, ms=4)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("mean cosine to original")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Edit robustness on a real BPE vocabulary  (n=400 tokens)",
                 color=plots.FG, y=1.04)
    plots.save(fig, "e1_geometry.png")


if __name__ == "__main__":
    r = run()
    print(__import__("json").dumps(r["run_vs_arun"], indent=2))
