"""E4 -- one operator, one code width, one projection matrix, three modalities.

The structural claim first, because it needs no accuracy number to be true:

  text   language ID, 10 scripts     bytes over a 1-D index   -> 8192
  audio  spoken digits 0-9, 10 classes  mu-law over a 1-D index  -> 8192
  image  MNIST, 10 classes           pixels over a 2-D index  -> 8192

Same budget every time; only the split between the value axis and the index
axes changes. The Kronecker codec cannot participate at all: its grid is
(byte x position), which has no 2-D form, and its width would scale with the
pixel count even if it did.

The measured claim is the harder one. If the three modalities really do land in
a comparable space, then ONE shared projection W: 8192 -> 128 should serve all
three about as well as three private projections. That is the arrangement a
multimodal model would actually want, and it is what we test.

Also swept here: the value-mode budget, which is the parameter that made
resolution invariance work at all (J=16 -> 0.995, J=256 -> 0.70).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firefly import data
from firefly import encoder as E
from firefly import plots
from firefly.plots import plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "e4_multimodal.json"

N_TEXT, N_IMAGE, N_AUDIO = 6000, 6000, 2700
SHARED_DIM = 128
STEPS = 1500
BATCH = 256
SEED = 0
MODALITIES = ["text", "image", "audio"]


# --------------------------------------------------------------------------
def encode_all(kind: str, log=print):
    """kind='phase' -> the canonical 8192-d complex code (the drop-in one).
       kind='full'  -> 12288-d, phase and magnitude both supplied."""
    fns = {
        "phase": (E.embed, E.embed_image, E.embed_audio),
        "full": (E.embed_full, E.embed_image_full, E.embed_audio_full),
    }[kind]
    log(f"  encoding text (language id, 10 scripts) [{kind}]")
    txt, ty = data.language_id(N_TEXT)
    tx = np.stack([fns[0](s) for s in txt]).astype(np.float32)

    log(f"  encoding images (MNIST, padded 28->32) [{kind}]")
    imgs, iy = data.mnist(N_IMAGE)
    ix = np.stack([fns[1](p) for p in imgs]).astype(np.float32)

    log(f"  encoding audio (free spoken digits 0-9, real recordings) [{kind}]")
    wavs, ay = data.spoken_digits(N_AUDIO)
    ax = np.stack([fns[2](w) for w in wavs]).astype(np.float32)

    return {"text": (tx, ty), "image": (ix, iy), "audio": (ax, ay)}


def split(x, y, frac=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(frac * len(y))
    return (x[idx[:cut]], y[idx[:cut]]), (x[idx[cut:]], y[idx[cut:]])


# --------------------------------------------------------------------------
class Probe(nn.Module):
    """Either one shared projection or three private ones, then a head each."""

    def __init__(self, code_dim: int, n_classes: dict, shared: bool):
        super().__init__()
        self.shared = shared
        if shared:
            self.proj = nn.Linear(code_dim, SHARED_DIM, bias=False)
        else:
            self.proj = nn.ModuleDict(
                {m: nn.Linear(code_dim, SHARED_DIM, bias=False) for m in n_classes})
        self.heads = nn.ModuleDict(
            {m: nn.Linear(SHARED_DIM, c) for m, c in n_classes.items()})

    def forward(self, x, modality: str):
        p = self.proj if self.shared else self.proj[modality]
        return self.heads[modality](F.gelu(p(x)))


def train_probe(sets, shared: bool, log=print) -> dict:
    torch.manual_seed(SEED)
    n_classes = {m: int(sets[m][0][1].max()) + 1 for m in MODALITIES}
    model = Probe(sets["text"][0][0].shape[1], n_classes, shared)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(SEED)
    model.train()
    for step in range(STEPS):
        loss = 0.0
        for m in MODALITIES:                       # one batch per modality per step
            x, y = sets[m][0]
            i = rng.choice(len(y), min(BATCH, len(y)))
            loss = loss + F.cross_entropy(
                model(torch.from_numpy(x[i]), m), torch.from_numpy(y[i]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            log(f"      {'shared' if shared else 'private'} step {step:4d} "
                f"loss {float(loss):.4f}")
    model.eval()
    acc = {}
    with torch.no_grad():
        for m in MODALITIES:
            x, y = sets[m][1]
            pred = model(torch.from_numpy(x), m).argmax(-1).numpy()
            acc[m] = float((pred == y).mean())
    return {"accuracy": acc,
            "projection_params": sum(p.numel() for n, p in model.named_parameters()
                                     if n.startswith("proj"))}


# --------------------------------------------------------------------------
def real_image_resolution(n: int = 400) -> dict:
    """Devil's advocate: the resolution-invariance claim was tested on ONE
    synthetic smooth field, which is the friendliest possible case. Real photos
    have edges, texture and noise -- energy well above the mode budget. Redo it
    on actual MNIST digits resampled to several resolutions.
    """
    from PIL import Image
    imgs, _ = data.mnist(n)
    out = {}
    for lo, hi in [(32, 64), (32, 128), (64, 128)]:
        cs = []
        for p in imgs:
            im = Image.fromarray(p)
            a = np.array(im.resize((lo, lo), Image.BILINEAR), dtype=np.uint8)
            b = np.array(im.resize((hi, hi), Image.BILINEAR), dtype=np.uint8)
            cs.append(E.cosine(E.embed_image(a), E.embed_image(b)))
        out[f"{lo}->{hi}"] = {"mean_cosine": float(np.mean(cs)),
                              "min_cosine": float(np.min(cs)), "n": len(cs)}
    return out


def value_mode_sweep() -> dict:
    """Why J=16. Same continuous field, two sampling resolutions.

    Caveat we state rather than hide: J was chosen using this very sweep, so the
    number below is a selection criterion, not an independent confirmation. The
    real-image check above is the independent one.
    """
    def field(h, w):
        y = np.arange(h)[:, None] / h
        x = np.arange(w)[None, :] / w
        return (128 + 100 * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * x)).astype(np.uint8)

    out = {}
    for j in [2, 4, 8, 16, 32, 64, 128, 256]:
        cs = [E.cosine(E.embed_image(field(lo, lo), kx=16, ky=16, value_modes=j),
                       E.embed_image(field(hi, hi), kx=16, ky=16, value_modes=j))
              for lo, hi in [(32, 64), (32, 128), (64, 128)]]
        out[str(j)] = {"mean_resolution_cosine": float(np.mean(cs)),
                       "code_dim": 16 * 16 * j * 2}
    return out


# --------------------------------------------------------------------------
def run(log=print) -> dict:
    t0 = time.time()
    out = {
        "budget_split": {
            "text": "256 value (delta) x 16 index x 2",
            "audio": f"{E.AUDIO_VALUE_MODES} value x {E.AUDIO_TIME_MODES} time x 2",
            "image": f"{E.IMAGE_VALUE_MODES} value x {E.IMAGE_SPATIAL}^2 spatial x 2",
        },
        "value_mode_sweep": value_mode_sweep(),
        "real_image_resolution": real_image_resolution(),
        "codes": {},
    }
    for kind in ["phase", "full"]:
        raw = encode_all(kind, log)
        sets = {m: split(*raw[m]) for m in MODALITIES}
        dims = {m: int(raw[m][0].shape[1]) for m in MODALITIES}
        # the structural claim: one width for every modality, whichever code
        assert len(set(dims.values())) == 1, dims
        log(f"  [{kind}] one shared projection across all three modalities")
        shared = train_probe(sets, True, log)
        log(f"  [{kind}] three private projections")
        private = train_probe(sets, False, log)
        out["codes"][kind] = {
            "code_dims": dims, "shared": shared, "private": private,
            "n_samples": {m: int(len(raw[m][1])) for m in MODALITIES},
            "shared_vs_private_gap": {
                m: private["accuracy"][m] - shared["accuracy"][m]
                for m in MODALITIES}}
        s, p = shared["accuracy"], private["accuracy"]
        log(f"    [{kind}] shared " + " ".join(f"{m}={s[m]:.3f}" for m in MODALITIES)
            + "  | private " + " ".join(f"{m}={p[m]:.3f}" for m in MODALITIES))

    # backwards-compatible aliases for the canonical code
    out["code_dims"] = out["codes"]["phase"]["code_dims"]
    out["n_samples"] = out["codes"]["phase"]["n_samples"]
    out["shared"] = out["codes"]["phase"]["shared"]
    out["private"] = out["codes"]["phase"]["private"]
    out["phase_vs_magnitude"] = {
        m: {"phase_only": out["codes"]["phase"]["shared"]["accuracy"][m],
            "phase_plus_magnitude": out["codes"]["full"]["shared"]["accuracy"][m]}
        for m in MODALITIES}
    out["wall_seconds"] = round(time.time() - t0, 1)
    _figure(out)
    data.save_json(out, RESULTS)
    f = out["codes"]["full"]["shared"]["accuracy"]
    log("[PASS] e4_multimodal  phase+magnitude, one shared W: " +
        "  ".join(f"{m}={f[m]:.3f}" for m in MODALITIES))
    return out


def _figure(out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    xs = np.arange(len(MODALITIES))
    pv = out["phase_vs_magnitude"]
    ax.bar(xs - 0.19, [pv[m]["phase_only"] for m in MODALITIES], 0.36,
           color=plots.FIREFLY2, label="phase only (8192)")
    ax.bar(xs + 0.19, [pv[m]["phase_plus_magnitude"] for m in MODALITIES], 0.36,
           color=plots.FIREFLY, label="phase + magnitude (12288)")
    ax.axhline(0.1, color=plots.DIM if hasattr(plots, "DIM") else "#8d95ab",
               ls=":", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(MODALITIES)
    ax.set_ylabel("test accuracy, one shared W"); ax.set_ylim(0, 1.05)
    ax.set_title("one code width, three modalities", color=plots.FG, fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    sw = out["value_mode_sweep"]
    js = [int(k) for k in sw]
    ax.semilogx(js, [sw[str(j)]["mean_resolution_cosine"] for j in js], "o-",
                base=2, color=plots.FIREFLY, lw=2)
    ax.axvline(E.IMAGE_VALUE_MODES, color=plots.OK, ls="--", lw=1.5,
               label=f"chosen J={E.IMAGE_VALUE_MODES}")
    ax.set_xlabel("value modes J  (256 = delta basis)")
    ax.set_ylabel("resolution invariance (cosine)")
    ax.set_title("why the value axis needs truncation", color=plots.FG, fontsize=10)
    ax.legend(fontsize=8)
    plots.save(fig, "e4_multimodal.png")


if __name__ == "__main__":
    run()
