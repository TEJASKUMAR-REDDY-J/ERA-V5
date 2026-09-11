# %% [markdown]
# # Adam, by hand and under load
#
# Six exercises: reproduce the optimiser arithmetic exactly, then find out what
# the schedule around it is actually doing.
#
# **Tejaskumar Reddy J** - ERA V5
#
# Runs top to bottom on CPU. Every number quoted in the README is printed here.

# %%
import os
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math, json, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2)
torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device {DEV} | threads {torch.get_num_threads()}")

B1, B2, EPS = 0.9, 0.999, 1e-8


# %% [markdown]
# ---
# # 1. Reproduce Adam by hand
#
# One scalar weight, five gradients. Compute `m`, `v`, `m_hat`, `v_hat` and the
# step by hand, then check every intermediate against PyTorch.
#
# The update is
#
# ```
# m_t     = b1*m_{t-1} + (1-b1)*g_t
# v_t     = b2*v_{t-1} + (1-b2)*g_t^2
# m_hat   = m_t / (1 - b1^t)
# v_hat   = v_t / (1 - b2^t)
# theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
# ```
#
# PyTorch writes the last line differently - it folds the corrections into a step
# size and a denominator - but the algebra is identical, and the check below is
# what establishes that rather than my say-so.

# %%
LR = 1e-2
W0 = 0.7
GRADS = [0.35, -0.12, 0.48, 0.03, -0.27]

# by hand, in float64
m, v, w = 0.0, 0.0, W0
hand = []
for t, g in enumerate(GRADS, start=1):
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    mhat = m / (1 - B1 ** t)
    vhat = v / (1 - B2 ** t)
    step = LR * mhat / (math.sqrt(vhat) + EPS)
    w = w - step
    hand.append({"t": t, "g": g, "m": m, "v": v, "mhat": mhat, "vhat": vhat,
                 "step": step, "w": w})

# PyTorch, same scalar, gradients injected directly
p = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([p], lr=LR, betas=(B1, B2), eps=EPS)
torch_rows = []
for t, g in enumerate(GRADS, start=1):
    opt.zero_grad()
    p.grad = torch.tensor([g], dtype=torch.float64)
    opt.step()
    st = opt.state[p]
    torch_rows.append({"t": t, "m": st["exp_avg"].item(), "v": st["exp_avg_sq"].item(),
                       "w": p.item()})

print(f"lr={LR}  b1={B1}  b2={B2}  eps={EPS}  w0={W0}")
print(f"gradients: {GRADS}\n")
print(f"{'t':<3}{'g':<8}{'m':<14}{'v':<16}{'m_hat':<14}{'v_hat':<16}{'step':<14}{'w':<14}")
print("-" * 100)
for h in hand:
    print(f"{h['t']:<3}{h['g']:<8.2f}{h['m']:<14.10f}{h['v']:<16.12f}"
          f"{h['mhat']:<14.10f}{h['vhat']:<16.12f}{h['step']:<14.10f}{h['w']:<14.10f}")

print(f"\n{'t':<3}{'hand m':<16}{'torch m':<16}{'|diff|':<12}"
      f"{'hand w':<16}{'torch w':<16}{'|diff|':<12}")
print("-" * 94)
worst_m = worst_v = worst_w = 0.0
for h, tr in zip(hand, torch_rows):
    dm, dw = abs(h["m"] - tr["m"]), abs(h["w"] - tr["w"])
    dv = abs(h["v"] - tr["v"])
    worst_m, worst_v, worst_w = max(worst_m, dm), max(worst_v, dv), max(worst_w, dw)
    print(f"{h['t']:<3}{h['m']:<16.12f}{tr['m']:<16.12f}{dm:<12.2e}"
          f"{h['w']:<16.12f}{tr['w']:<16.12f}{dw:<12.2e}")

def decimals(x):
    return 16.0 if x == 0 else max(0.0, -math.log10(x))

print(f"\nworst |diff| over all five steps:")
print(f"  m      {worst_m:.2e}   ({decimals(worst_m):.1f} matching decimals)")
print(f"  v      {worst_v:.2e}   ({decimals(worst_v):.1f} matching decimals)")
print(f"  weight {worst_w:.2e}   ({decimals(worst_w):.1f} matching decimals)")
print("\nPyTorch keeps only m and v in state; m_hat and v_hat are applied on the fly,")
print("so they are checked through the weight they produce.")
ADAM_MATCH = {"m": worst_m, "v": worst_v, "w": worst_w,
              "decimals_w": decimals(worst_w)}


# %% [markdown]
# ---
# # 2. Turn bias correction off
#
# Both variants, same gradients, first twenty steps. The interesting quantity is
# the ratio of the corrected step to the uncorrected one:
#
# ```
# corrected / uncorrected  =  sqrt(1 - b2^t) / (1 - b1^t)      (eps negligible)
# ```
#
# which depends only on `t`, `b1` and `b2` - not on the gradients at all. So the
# question "after how many steps does it stop mattering" has an exact answer.

# %%
def adam_trace(grads, lr=LR, w0=W0, correct=True):
    m = v = 0.0
    w = w0
    out = []
    for t, g in enumerate(grads, start=1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        if correct:
            mh, vh = m / (1 - B1 ** t), v / (1 - B2 ** t)
        else:
            mh, vh = m, v
        step = lr * mh / (math.sqrt(vh) + EPS)
        w -= step
        out.append({"t": t, "step": step, "w": w})
    return out


torch.manual_seed(1)
G20 = (0.3 * torch.randn(20, generator=torch.Generator().manual_seed(7))).tolist()
on = adam_trace(G20, correct=True)
off = adam_trace(G20, correct=False)

print(f"{'t':<4}{'step (corrected)':<20}{'step (uncorrected)':<20}"
      f"{'ratio':<10}{'theory':<10}")
print("-" * 66)
for a, b in zip(on, off):
    t = a["t"]
    ratio = a["step"] / b["step"] if b["step"] != 0 else float("nan")
    theory = math.sqrt(1 - B2 ** t) / (1 - B1 ** t)
    print(f"{t:<4}{a['step']:<20.10f}{b['step']:<20.10f}{ratio:<10.4f}{theory:<10.4f}")

print("\nThe ratio matches the closed form, so it is a property of the schedule of")
print("corrections, not of this particular gradient sequence.\n")

# when does it stop mattering?
def ratio_at(t):
    return math.sqrt(1 - B2 ** t) / (1 - B1 ** t)


THRESH = [0.50, 0.20, 0.10, 0.05, 0.01]
when = {}
for th in THRESH:
    t = 1
    while abs(ratio_at(t) - 1.0) > th and t < 200000:
        t += 1
    when[th] = t
    print(f"  within {100 * th:>4.0f}% of 1.0 after {t:>6} steps   "
          f"(ratio {ratio_at(t):.4f})")

print(f"\nAt step 20 the ratio is still {ratio_at(20):.4f} - the corrected step is")
print(f"{100 * (1 - ratio_at(20)):.0f}% SMALLER than the uncorrected one. Twenty steps is")
print("nowhere near enough for the difference to stop mattering.")
print(f"\nThe binding term is b2. 1 - b1^t reaches 0.99 at t={math.ceil(math.log(0.01) / math.log(B1))},")
print(f"but sqrt(1 - b2^t) needs t={when[0.01]} to get within 1%. Bias correction on the")
print("second moment is the part that persists, and it persists for thousands of steps.")
BIAS_WHEN = when


# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ts = [a["t"] for a in on]
ax[0].plot(ts, [a["step"] for a in on], "o-", label="bias corrected", color="#2a9d8f")
ax[0].plot(ts, [b["step"] for b in off], "s-", label="uncorrected", color="#d1495b")
ax[0].set_xlabel("step"); ax[0].set_ylabel("update size"); ax[0].legend()
ax[0].set_title("First 20 steps"); ax[0].grid(alpha=.3)

ax[1].plot(ts, [a["w"] for a in on], "o-", label="bias corrected", color="#2a9d8f")
ax[1].plot(ts, [b["w"] for b in off], "s-", label="uncorrected", color="#d1495b")
ax[1].set_xlabel("step"); ax[1].set_ylabel("weight"); ax[1].legend()
ax[1].set_title("Weight trajectory"); ax[1].grid(alpha=.3)

tt = list(range(1, 6000))
ax[2].semilogx(tt, [ratio_at(t) for t in tt], color="#7c5cff", lw=2)
ax[2].axhline(1.0, ls=":", c="gray")
for th, c in [(0.10, "#f5a94c"), (0.01, "#2a9d8f")]:
    ax[2].axvline(when[th], ls="--", c=c, lw=1,
                  label=f"within {100*th:.0f}%: t={when[th]}")
ax[2].axvline(20, ls="-", c="#d1495b", lw=1, label="t=20 (the plot on the left)")
ax[2].set_xlabel("step"); ax[2].set_ylabel("corrected / uncorrected")
ax[2].set_title("When it stops mattering"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("bias_correction.png", dpi=130)
print("saved bias_correction.png")


# %% [markdown]
# ---
# ## A model, for the rest of the exercises
#
# Character-level, so the vocabulary is small and a learning-rate sweep at three
# widths is affordable on CPU.

# %%
from datasets import load_dataset

_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
TEXT = "\n".join(t.strip() for t in _ds["text"] if len(t.strip()) > 200)[:400_000]
CHARS = sorted(set(TEXT))
VOCAB = len(CHARS)
STOI = {c: i for i, c in enumerate(CHARS)}
SEQ = 32
_ids = torch.tensor([STOI[c] for c in TEXT], dtype=torch.long)
DATA = _ids[: (_ids.numel() // SEQ) * SEQ].view(-1, SEQ).to(DEV)
print(f"chars {len(TEXT):,} | vocab {VOCAB} | data {tuple(DATA.shape)}")


class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, h, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, m):
        a = s.n1(x)
        x = x + s.att(a, a, a, attn_mask=m, need_weights=False)[0]
        return x + s.mlp(s.n2(x))


class TinyLM(nn.Module):
    def __init__(s, width, n_layer=2, n_head=4, seq=SEQ, vocab=VOCAB):
        super().__init__()
        s.emb = nn.Embedding(vocab, width)
        s.pos = nn.Embedding(seq, width)
        s.blocks = nn.ModuleList([Blk(width, n_head) for _ in range(n_layer)])
        s.norm = nn.LayerNorm(width)
        s.head = nn.Linear(width, vocab, bias=False)
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(s, ids):
        T = ids.shape[1]
        x = s.emb(ids) + s.pos(torch.arange(T, device=ids.device))[None]
        cm = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        for b in s.blocks:
            x = b(x, cm)
        return s.head(s.norm(x))


def batch(bs, gen):
    idx = torch.randint(0, DATA.shape[0], (bs,), generator=gen)
    return DATA[idx.to(DATA.device)]


def step_loss(model, b):
    lg = model(b)
    return F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1))


# %% [markdown]
# ---
# # 3. Update-to-weight ratio per layer, and where warmup stops changing it
#
# For each layer, `||update|| / ||weight||` every step. Under warmup the learning
# rate is the only thing moving, so the ratio should track it and then detach.

# %%
WARMUP = 60
TOTAL3 = 200
WIDTH3 = 256


def lr_warmup(t, peak=3e-3, warm=WARMUP):
    return peak * min(1.0, (t + 1) / warm)


torch.manual_seed(0)
m3 = TinyLM(WIDTH3).to(DEV)
opt3 = torch.optim.Adam(m3.parameters(), lr=1e-3, betas=(B1, B2), eps=EPS)
gen3 = torch.Generator().manual_seed(3)

TRACK = {"emb": m3.emb.weight, "blk0.attn": m3.blocks[0].att.in_proj_weight,
         "blk0.mlp": m3.blocks[0].mlp[0].weight, "blk1.mlp": m3.blocks[1].mlp[0].weight,
         "head": m3.head.weight}
ratios = {k: [] for k in TRACK}
lrs = []

for t in range(TOTAL3):
    lr = lr_warmup(t)
    for gparam in opt3.param_groups:
        gparam["lr"] = lr
    lrs.append(lr)
    before = {k: w.detach().clone() for k, w in TRACK.items()}
    loss = step_loss(m3, batch(16, gen3))
    opt3.zero_grad(set_to_none=True)
    loss.backward()
    opt3.step()
    for k, w in TRACK.items():
        d = (w.detach() - before[k]).norm().item()
        ratios[k].append(d / max(before[k].norm().item(), 1e-12))

print(f"warmup configured for {WARMUP} steps, peak lr {lr_warmup(WARMUP - 1):.1e}\n")
print(f"{'step':<7}" + "".join(f"{k:<13}" for k in TRACK) + "lr")
print("-" * 82)
for t in [0, 10, 30, 50, 58, 59, 60, 61, 70, 100, 150, 199]:
    print(f"{t:<7}" + "".join(f"{ratios[k][t]:<13.3e}" for k in TRACK) + f"{lrs[t]:.2e}")


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) ** .5
    vb = sum((x - mb) ** 2 for x in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / max(va * vb, 1e-12)


print("\nDuring warmup the lr is the only thing moving, so correlating the ratio")
print("against it is meaningful. AFTER warmup the lr is constant, so that")
print("correlation is undefined - not zero. Reporting the ratio's own drift instead.\n")
print(f"{'layer':<14}{'corr w/ lr (warmup)':<22}{'mean after':<14}{'CV after':<12}{'drift':<10}")
print("-" * 74)
CORR = {}
for k in TRACK:
    c_in = pearson(ratios[k][:WARMUP], lrs[:WARMUP])
    post = ratios[k][WARMUP:]
    mu = sum(post) / len(post)
    sd = (sum((x - mu) ** 2 for x in post) / len(post)) ** .5
    first = sum(post[:20]) / 20
    last = sum(post[-20:]) / 20
    CORR[k] = (c_in, mu, sd / max(mu, 1e-12), last / max(first, 1e-12))
    print(f"{k:<14}{c_in:<22.3f}{mu:<14.3e}{sd / max(mu, 1e-12):<12.3f}"
          f"{last / max(first, 1e-12):<10.2f}x")

# The knee: where the ratio stops rising. Take the argmax of the smoothed mean,
# which is the last point at which warmup was still pushing it up.
def smooth(x, w=7):
    return [sum(x[max(0, i - w + 1):i + 1]) / len(x[max(0, i - w + 1):i + 1])
            for i in range(len(x))]


mean_ratio = [sum(ratios[k][t] for k in TRACK) / len(TRACK) for t in range(TOTAL3)]
sm_ratio = smooth(mean_ratio)
KNEE = max(range(TOTAL3), key=lambda t: sm_ratio[t])
print(f"\nthe smoothed mean ratio peaks at step {KNEE}; warmup was configured to end "
      f"at {WARMUP}")
print(f"mean ratio over steps 0-{WARMUP}: {sum(mean_ratio[:WARMUP]) / WARMUP:.3e}")
print(f"mean ratio over steps {WARMUP}-{TOTAL3}: "
      f"{sum(mean_ratio[WARMUP:]) / (TOTAL3 - WARMUP):.3e}")
print("\nNot every layer tracks the lr equally. The embedding and the first attention")
print("block follow it closely; the deeper MLP and the head barely do, because their")
print("ratio is dominated by how fast their own gradients are changing rather than by")
print("the schedule. So 'warmup stops changing the ratio' is a per-layer statement,")
print("and the layers that were never lr-driven never had a knee to begin with.")

fig, ax = plt.subplots(figsize=(9, 4))
for k in TRACK:
    ax.plot(ratios[k], lw=1.4, label=k)
ax.axvline(WARMUP, ls="--", c="#d1495b", label=f"warmup ends ({WARMUP})")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("||update|| / ||weight||")
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Update-to-weight ratio per layer")
plt.tight_layout(); plt.savefig("update_ratio.png", dpi=130)
print("saved update_ratio.png")


# %% [markdown]
# ---
# # 5. Learning-rate sweep at three widths
#
# Done before the schedule comparison on purpose: item 6 says tune both sides
# before accepting a comparison, so the tuned learning rates have to exist first.

# %%
def train(width, lr, steps=120, sched=None, bs=16, seed=0, warm=20, log_every=None):
    torch.manual_seed(seed)
    m = TinyLM(width).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(B1, B2), eps=EPS)
    gen = torch.Generator().manual_seed(100 + seed)
    hist = []
    for t in range(steps):
        if sched is None:
            cur = lr * min(1.0, (t + 1) / warm)
        else:
            cur = sched(t, steps, lr, warm)
        for g in opt.param_groups:
            g["lr"] = cur
        loss = step_loss(m, batch(bs, gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
    return m, hist


def tail(h, k=15):
    return sum(h[-k:]) / k


WIDTHS = [256, 512, 1024]
# Extended below 3e-4 because the first pass put width 1024's minimum at the very
# edge of the grid, which means it was never bracketed and the "optimum" was just
# the smallest value tried.
LR_GRID = [7.5e-5, 1.5e-4, 3e-4, 6e-4, 1.2e-3, 2.5e-3, 5e-3, 1e-2]
fmt_lr = lambda x: f"{x:.2g}"
SWEEP = {}
print("sweeping (final loss = mean of last 15 steps of 120)\n")
print(f"{'width':<8}" + "".join(f"{fmt_lr(lr):<10}" for lr in LR_GRID) + "  best")
print("-" * 98)
t0 = time.perf_counter()
for w in WIDTHS:
    row = {}
    for lr in LR_GRID:
        _, h = train(w, lr, steps=120, seed=0)
        row[lr] = tail(h)
    SWEEP[w] = row
    best = min(row, key=row.get)
    edge = " EDGE" if best in (LR_GRID[0], LR_GRID[-1]) else ""
    print(f"{w:<8}" + "".join(f"{row[lr]:<10.3f}" for lr in LR_GRID)
          + f"  {fmt_lr(best)} ({row[best]:.3f}){edge}")
print(f"\nsweep took {time.perf_counter() - t0:.0f}s")

BEST_LR = {w: min(SWEEP[w], key=SWEEP[w].get) for w in WIDTHS}
print("minima:", {w: fmt_lr(BEST_LR[w]) for w in WIDTHS})
EDGE = [w for w in WIDTHS if BEST_LR[w] in (LR_GRID[0], LR_GRID[-1])]
if EDGE:
    print(f"WARNING: width(s) {EDGE} put their minimum at a grid edge - not bracketed,")
    print("so that 'optimum' is only the smallest value tried and the fit is unsafe.")

# fit a power law  lr* = C * width^a  through the three minima
lw = [math.log(w) for w in WIDTHS]
ll = [math.log(BEST_LR[w]) for w in WIDTHS]
n = len(lw)
mw, ml = sum(lw) / n, sum(ll) / n
ALPHA = sum((x - mw) * (y - ml) for x, y in zip(lw, ll)) / sum((x - mw) ** 2 for x in lw)
LOGC = ml - ALPHA * mw
PRED_4096 = math.exp(LOGC + ALPHA * math.log(4096))
ss_res = sum((y - (LOGC + ALPHA * x)) ** 2 for x, y in zip(lw, ll))
ss_tot = sum((y - ml) ** 2 for y in ll)
R2 = 1 - ss_res / max(ss_tot, 1e-12)

print(f"\npower-law fit  lr* = {math.exp(LOGC):.3e} * width^{ALPHA:.3f}   (R2 = {R2:.4f})")
print(f"extrapolated to width 4096: lr* = {PRED_4096:.2e}")

# How much to trust that number.
import itertools

GRID_RATIO = LR_GRID[1] / LR_GRID[0]
print(f"\nHOW MUCH TO TRUST IT. The grid is geometric with ratio {GRID_RATIO:.2g}, so every")
print("minimum is snapped to the nearest grid point and carries roughly")
print(f"+/-{100 * (GRID_RATIO - 1) / 2:.0f}% of quantisation error. Three snapped points lying on a")
print("straight line in log-log is close to guaranteed: R2 = 1.0000 here is a")
print("property of the grid, NOT evidence that the exponent is measured precisely.")
print(f"Extrapolating to 4096 is a {4096 / max(WIDTHS):.0f}x reach past the largest width tried,")
print("from 3 points, at one depth, one dataset, one step budget, one seed.")

alts = []
for shifts in itertools.product([-1, 0, 1], repeat=3):
    lls, ok = [], True
    for wid, sh in zip(WIDTHS, shifts):
        i = LR_GRID.index(BEST_LR[wid]) + sh
        if not (0 <= i < len(LR_GRID)):
            ok = False
            break
        lls.append(math.log(LR_GRID[i]))
    if not ok:
        continue
    ml2 = sum(lls) / 3
    a = (sum((x - mw) * (y - ml2) for x, y in zip(lw, lls))
         / sum((x - mw) ** 2 for x in lw))
    alts.append(math.exp((ml2 - a * mw) + a * math.log(4096)))
PRED_RANGE = (min(alts), max(alts))
print(f"\nIf any one minimum is off by a single grid notch, the width-4096 answer")
print(f"ranges over {PRED_RANGE[0]:.1e} to {PRED_RANGE[1]:.1e} - a factor of "
      f"{PRED_RANGE[1] / PRED_RANGE[0]:.0f}.")
print("\nThe exponent landing on exactly -1.000 is worth one more caution: that is")
print("the textbook standard-parameterisation result, which makes it the answer I")
print("was most likely to accept without checking. It is consistent with theory,")
print("but three grid-snapped points cannot distinguish -1.0 from, say, -0.85.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cols = {256: "#2a9d8f", 512: "#7c5cff", 1024: "#f5a94c"}
for w in WIDTHS:
    xs = sorted(SWEEP[w])
    ax[0].semilogx(xs, [SWEEP[w][x] for x in xs], "o-", color=cols[w], label=f"width {w}")
    ax[0].scatter([BEST_LR[w]], [SWEEP[w][BEST_LR[w]]], s=160, facecolors="none",
                  edgecolors=cols[w], linewidths=2.2, zorder=5)
ax[0].set_xlabel("learning rate"); ax[0].set_ylabel("loss (mean of last 15 steps)")
ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Sweep, minima circled")

ax[1].loglog(WIDTHS, [BEST_LR[w] for w in WIDTHS], "o", ms=9, color="#2a9d8f",
             label="measured minima")
xs = [200, 4096]
ax[1].loglog(xs, [math.exp(LOGC + ALPHA * math.log(x)) for x in xs], "--",
             color="#7c5cff", label=f"fit: width^{ALPHA:.2f}")
ax[1].scatter([4096], [PRED_4096], marker="*", s=260, color="#d1495b",
              zorder=5, label=f"predicted {PRED_4096:.1e}")
ax[1].set_xlabel("width"); ax[1].set_ylabel("optimal lr"); ax[1].legend(fontsize=8)
ax[1].grid(alpha=.3, which="both"); ax[1].set_title("Extrapolation to width 4096")
plt.tight_layout(); plt.savefig("lr_sweep.png", dpi=130)
print("saved lr_sweep.png")


# %% [markdown]
# ---
# # 4 and 6. Cosine against WSD, with both sides tuned
#
# Item 6 is not a separate exercise, it is the condition under which item 4 is
# allowed to mean anything. So each schedule gets its own learning-rate sweep, and
# the comparison is run twice: once at a shared learning rate, once tuned.

# %%
TOTAL, STOP, WARM = 300, 200, 20
W4 = 256


def sched_cosine(t, total, peak, warm):
    if t < warm:
        return peak * (t + 1) / warm
    p = (t - warm) / max(1, total - warm)
    return 0.5 * peak * (1 + math.cos(math.pi * p))


def sched_wsd(t, total, peak, warm, decay_frac=0.2):
    d0 = int(total * (1 - decay_frac))
    if t < warm:
        return peak * (t + 1) / warm
    if t < d0:
        return peak
    p = (t - d0) / max(1, total - d0)
    return peak * (1 - p)


SCHEDS = {"cosine": sched_cosine, "wsd": sched_wsd}
GRID4 = [3e-4, 6e-4, 1.2e-3, 2.5e-3, 5e-3]
print("tuning each schedule separately (loss at step 200 of a 300-step run)\n")
print(f"{'schedule':<10}" + "".join(f"{fmt_lr(lr):<11}" for lr in GRID4) + "  best")
print("-" * 62)
TUNE4, HIST4 = {}, {}
for name, fn in SCHEDS.items():
    row = {}
    for lr in GRID4:
        _, h = train(W4, lr, steps=TOTAL, sched=fn, warm=WARM, seed=0)
        row[lr] = tail(h[:STOP])
        HIST4[(name, lr)] = h
    TUNE4[name] = row
    b = min(row, key=row.get)
    print(f"{name:<10}" + "".join(f"{row[lr]:<11.4f}" for lr in GRID4) + f"  {fmt_lr(b)}")

BEST4 = {k: min(v, key=v.get) for k, v in TUNE4.items()}
print(f"\ntuned learning rates: cosine {BEST4['cosine']:.1e}, wsd {BEST4['wsd']:.1e}")

cos_h = HIST4[("cosine", BEST4["cosine"])]
wsd_h = HIST4[("wsd", BEST4["wsd"])]
COS_200, WSD_200 = tail(cos_h[:STOP]), tail(wsd_h[:STOP])
COS_300, WSD_300 = tail(cos_h), tail(wsd_h)

print(f"\nboth stopped at step {STOP} (each at its own tuned lr):")
print(f"  cosine  {COS_200:.4f}")
print(f"  wsd     {WSD_200:.4f}")
print(f"  gap     {WSD_200 - COS_200:+.4f}")
print(f"\nif allowed to finish all {TOTAL}:")
print(f"  cosine  {COS_300:.4f}")
print(f"  wsd     {WSD_300:.4f}")

# The same comparison run at every shared learning rate. If the verdict depends
# on which one you happened to pick, then a single-lr comparison establishes
# nothing - which is the whole of item 6.
print(f"\nthe same comparison, at each shared learning rate:")
print(f"{'shared lr':<12}{'cosine':<11}{'wsd':<11}{'gap (wsd-cos)':<16}winner")
print("-" * 58)
GAPS = {}
for lr in GRID4:
    c = tail(HIST4[("cosine", lr)][:STOP])
    w_ = tail(HIST4[("wsd", lr)][:STOP])
    GAPS[lr] = (c, w_, w_ - c)
    print(f"{fmt_lr(lr):<12}{c:<11.4f}{w_:<11.4f}{w_ - c:<+16.4f}"
          f"{'cosine' if w_ > c else 'wsd'}")

signs = {1 if g > 0 else -1 for _, _, g in GAPS.values()}
FLIPS = len(signs) > 1
worst_lr = max(GAPS, key=lambda k: abs(GAPS[k][2]))
best_lr_for_wsd = min(GAPS, key=lambda k: GAPS[k][2])
SHARED = best_lr_for_wsd
bad_c, bad_w = GAPS[SHARED][0], GAPS[SHARED][1]
SHIFT = (bad_w - bad_c) - (WSD_200 - COS_200)

print(f"\nthe verdict {'FLIPS' if FLIPS else 'does not flip'} across the grid.")
if FLIPS:
    pro_w = [lr for lr in GAPS if GAPS[lr][2] < 0]
    pro_c = [lr for lr in GAPS if GAPS[lr][2] > 0]
    print(f"  lrs where WSD wins   : {[fmt_lr(x) for x in pro_w]}")
    print(f"  lrs where cosine wins: {[fmt_lr(x) for x in pro_c]}")
    print(f"\nSo 'cosine beats WSD' and 'WSD beats cosine' are both obtainable from this")
    print("same experiment by choosing the shared learning rate. Either claim would")
    print("have replicated for whoever ran it and failed for everyone else.")
print(f"\nspread of the gap across the grid: {min(g for _, _, g in GAPS.values()):+.4f} "
      f"to {max(g for _, _, g in GAPS.values()):+.4f}, a range of "
      f"{max(g for _, _, g in GAPS.values()) - min(g for _, _, g in GAPS.values()):.4f}")
print(f"the tuned-vs-tuned gap is {WSD_200 - COS_200:+.4f}, which is the only one of these")
print("numbers that is about the schedules rather than about the learning rate.")

# what WSD is actually for: decay from wherever you decide to stop
def sched_wsd_stop200(t, total, peak, warm):
    return sched_wsd(t, STOP, peak, warm, decay_frac=0.2)


_, h_stop = train(W4, BEST4["wsd"], steps=STOP, sched=sched_wsd_stop200, warm=WARM, seed=0)
WSD_DECAYED = tail(h_stop)
print(f"\nWSD with its decay moved to end at step {STOP}: {WSD_DECAYED:.4f}")
print(f"  vs cosine stopped early at {STOP}: {COS_200:.4f}")
print(f"  vs WSD interrupted mid-plateau:   {WSD_200:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
sm = lambda h, w=10: [sum(h[max(0, i - w):i + 1]) / len(h[max(0, i - w):i + 1])
                      for i in range(len(h))]
ax[0].plot(sm(cos_h), label=f"cosine @ {BEST4['cosine']:.0e}", color="#2a9d8f")
ax[0].plot(sm(wsd_h), label=f"wsd @ {BEST4['wsd']:.0e}", color="#7c5cff")
ax[0].plot(sm(h_stop), label="wsd decayed at 200", color="#d1495b", ls="--")
ax[0].axvline(STOP, ls=":", c="gray")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss (smoothed)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("Tuned comparison")

ax[1].plot([sched_cosine(t, TOTAL, BEST4["cosine"], WARM) for t in range(TOTAL)],
           label="cosine", color="#2a9d8f")
ax[1].plot([sched_wsd(t, TOTAL, BEST4["wsd"], WARM) for t in range(TOTAL)],
           label="wsd", color="#7c5cff")
ax[1].axvline(STOP, ls=":", c="gray", label="stop at 200")
ax[1].set_xlabel("step"); ax[1].set_ylabel("learning rate")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_title("The schedules")
plt.tight_layout(); plt.savefig("schedules.png", dpi=130)
print("saved schedules.png")


# %% [markdown]
# ---
# ## Summary

# %%
SUMMARY = {
    "1. adam by hand": f"worst |diff| vs torch: m {ADAM_MATCH['m']:.1e}, "
                       f"v {ADAM_MATCH['v']:.1e}, weight {ADAM_MATCH['w']:.1e} "
                       f"({ADAM_MATCH['decimals_w']:.1f} decimals)",
    "2. bias correction": f"ratio at t=20 is {ratio_at(20):.4f}; within 10% at "
                          f"t={BIAS_WHEN[0.10]}, within 1% at t={BIAS_WHEN[0.01]}",
    "3. warmup knee": f"ratio peaks at step {KNEE}, warmup configured {WARMUP}; "
                      f"lr-corr during warmup: emb {CORR['emb'][0]:+.2f}, "
                      f"attn {CORR['blk0.attn'][0]:+.2f}, head {CORR['head'][0]:+.2f}",
    "4. cosine vs wsd @200": f"cosine {COS_200:.4f} vs wsd {WSD_200:.4f} "
                             f"(gap {WSD_200 - COS_200:+.4f}); wsd decayed at 200 "
                             f"{WSD_DECAYED:.4f}",
    "5. lr sweep": f"minima {fmt_lr(BEST_LR[256])}/{fmt_lr(BEST_LR[512])}/"
                   f"{fmt_lr(BEST_LR[1024])}; lr* ~ width^{ALPHA:.2f} -> {PRED_4096:.1e} "
                   f"at 4096 (one-notch range {PRED_RANGE[0]:.1e}-{PRED_RANGE[1]:.1e})",
    "6. tuning both sides": f"gap across the grid ranges "
                            f"{min(g for _, _, g in GAPS.values()):+.4f} to "
                            f"{max(g for _, _, g in GAPS.values()):+.4f}; verdict "
                            f"{'flips' if FLIPS else 'holds'}; tuned gap "
                            f"{WSD_200 - COS_200:+.4f}",
}
print(f"{'item':<24}value")
print("-" * 118)
for k, v in SUMMARY.items():
    print(f"{k:<24}{v}")

with open("results.json", "w") as fh:
    json.dump({
        "device": str(DEV), "summary": SUMMARY,
        "adam_hand": hand, "adam_torch": torch_rows, "adam_match": ADAM_MATCH,
        "bias": {"on": on, "off": off, "thresholds": {str(k): v for k, v in BIAS_WHEN.items()},
                 "ratio_at_20": ratio_at(20)},
        "warmup": {"knee": KNEE, "configured": WARMUP,
                   "corr": {k: list(v) for k, v in CORR.items()},
                   "ratios": ratios, "lrs": lrs},
        "sweep": {str(w): {str(k): v for k, v in SWEEP[w].items()} for w in WIDTHS},
        "sweep_fit": {"alpha": ALPHA, "C": math.exp(LOGC), "r2": R2,
                      "pred_4096": PRED_4096, "pred_range": list(PRED_RANGE),
                      "best": {str(w): BEST_LR[w] for w in WIDTHS}},
        "schedules": {"tuned": {k: {str(a): b for a, b in v.items()}
                                for k, v in TUNE4.items()},
                      "best_lr": {k: v for k, v in BEST4.items()},
                      "cos_200": COS_200, "wsd_200": WSD_200,
                      "cos_300": COS_300, "wsd_300": WSD_300,
                      "wsd_decayed_200": WSD_DECAYED,
                      "shared_lr": SHARED, "bad_cos": bad_c, "bad_wsd": bad_w,
                      "gaps": {str(k): list(v) for k, v in GAPS.items()},
                      "verdict_flips": FLIPS},
    }, fh, indent=2, default=float)
print("\nwrote results.json")
