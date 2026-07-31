#!/usr/bin/env python3
"""Plots for the plan, straight from proxy_results.json + the inventory. GitHub-embeddable PNGs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spseudo": True} if False else {})
BG, INK, ACC, ACC2, WARN = "#0d0f14", "#e8eaf0", "#4cc9f0", "#7c5cff", "#ff5d73"


def style(ax, fig):
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#33343c")
    ax.tick_params(colors=INK); ax.yaxis.label.set_color(INK); ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK); ax.grid(alpha=.18)


def main():
    d = json.loads((ROOT / "proxy_results.json").read_text())
    arms = [a for a in d["arms"] if not a["arm"].endswith("_s1")]
    arms.sort(key=lambda a: a["shares"]["indic"])
    x = [a["shares"]["indic"] * 100 for a in arms]
    indic = [a["eval"]["indic"]["bits_per_char"] for a in arms]
    web = [a["eval"]["web"]["bits_per_char"] for a in arms]
    code = [a["eval"]["code"]["bits_per_char"] for a in arms]

    # --- fig 1: BPC curves, knee marked ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, indic, "-o", color=ACC, lw=2.4, label="Indic (target)")
    ax.plot(x, web, "-o", color="#7d90a5", lw=1.6, label="web (cost)")
    ax.plot(x, code, "-o", color=WARN, lw=1.6, label="code (cost)")
    ax.axvline(16, color=ACC2, ls="--", lw=1.4)
    ax.annotate("knee → floor = 16%", (16, indic[1]), (18, indic[1] + .18),
                color=ACC2, fontsize=9)
    ax.set_xlabel("Indic share of mixture (%)"); ax.set_ylabel("held-out bits per char (↓ better)")
    ax.set_title("Proxy sweep — Indic improves, cost axes are noise-dominated")
    lg = ax.legend(facecolor=BG, edgecolor="#33343c", labelcolor=INK)
    style(ax, fig); fig.tight_layout(); fig.savefig(FIG / "fig1_bpc_curve.png"); plt.close(fig)

    # --- fig 2: marginal Indic gain per point + efficiency ---
    seg = [f"{int(x[i-1])}→{int(x[i])}%" for i in range(1, len(x))]
    dperpt = [(indic[i] - indic[i - 1]) / (x[i] - x[i - 1]) for i in range(1, len(x))]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(seg, [-v for v in dperpt], color=[ACC, ACC2, "#5a44c8"])
    for b, v in zip(bars, dperpt):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .0006,
                f"{v:+.4f}", ha="center", color=INK, fontsize=9)
    ax.axhline(0.007, color=WARN, ls="--", lw=1.2)
    ax.text(0.02, 0.008, "seed noise floor (0.007)", color=WARN, fontsize=8)
    ax.set_ylabel("Indic BPC gained per +1% share (↑ better)")
    ax.set_title("Marginal return halves after 16% — where the floor sits")
    style(ax, fig); fig.tight_layout(); fig.savefig(FIG / "fig2_marginal.png"); plt.close(fig)

    # --- fig 3: inventory supply vs plan demand (log), the honesty chart ---
    SUP = {"web": 4500, "code": 1100, "Indic": 276, "STEM": 250,
           "long-ctx": 100, "reason": 85, "agentic": 0.63}   # B tokens
    DEM = {"web": 629, "code": 444, "Indic": 296, "STEM": 222,
           "long-ctx": 111, "reason": 111, "agentic": 37}
    lanes = list(SUP)
    import numpy as np
    xi = np.arange(len(lanes)); w = .38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(xi - w / 2, [SUP[l] for l in lanes], w, color=ACC, label="real supply")
    ax.bar(xi + w / 2, [DEM[l] for l in lanes], w, color=WARN, label="plan demand @2T")
    ax.set_yscale("log"); ax.set_xticks(xi); ax.set_xticklabels(lanes, rotation=20)
    ax.set_ylabel("tokens (B, log)")
    ax.set_title("Supply vs demand — agentic is 58× short (must be synthesized)")
    ax.annotate("58× gap", (6, 3), (5.1, 20), color=WARN, fontsize=9,
                arrowprops=dict(color=WARN, arrowstyle="->"))
    lg = ax.legend(facecolor=BG, edgecolor="#33343c", labelcolor=INK)
    style(ax, fig); fig.tight_layout(); fig.savefig(FIG / "fig3_supply_demand.png"); plt.close(fig)

    print("wrote", [p.name for p in FIG.glob("*.png")])


if __name__ == "__main__":
    main()
