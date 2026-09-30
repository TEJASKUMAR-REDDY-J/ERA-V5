"""A 20M byte-level LLM trained three ways: standard, reversible, reversible at max batch.

    python train.py                 # SCALE from the env, default "rehearsal"
    SCALE=full python train.py      # the 50M-token runs (needs a GPU)
    python to_notebook.py           # regenerates reversible.ipynb from this file

This file is the source of truth. The notebook is generated from it so the two
cannot drift.
"""

# %% [markdown]
# # Reversible transformers: paying compute for depth-free memory
#
# A standard transformer stores every sublayer's activations for the backward
# pass, so activation memory grows linearly with depth. A reversible one does
# not: given the outputs, the inputs are recomputed by running the coupling
# backwards, so activations are freed going forward and reconstructed going
# back. Depth becomes free in memory and costs roughly one extra forward pass.
#
# The point of this notebook is that **reversibility is not a loss technique**.
# It buys memory, and memory is only worth something if you spend it. So the
# comparison that matters is not run 1 against run 2 — it is run 2 against run 3,
# where the freed memory is turned into batch size.
#
# Everything is byte-level (vocab 256), so there is no tokenizer to download and
# essentially all of the 20M parameters sit in the transformer rather than in an
# embedding table.

# %%
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent if "__file__" in dir() else Path(".")
SCALE = os.environ.get("SCALE", "local20m")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 4)

CFGS = dict(
    # the assignment: 20M parameters, 50M tokens. Needs a GPU.
    full=dict(d=512, n_layer=6, n_head=8, d_ff=2048, seq=512,
              tokens=50_000_000, warmup=500, lr=6e-4),
    # the same 20M model on a 4-core CPU box, with the token budget cut to what
    # fits in a couple of hours. Architecture, memory and throughput ratios are
    # the real thing; only the number of tokens is short.
    local20m=dict(d=512, n_layer=6, n_head=8, d_ff=2048, seq=512,
                  tokens=1_000_000, warmup=100, lr=6e-4),
    rehearsal=dict(d=256, n_layer=4, n_head=4, d_ff=1024, seq=256,
                   tokens=1_500_000, warmup=100, lr=1e-3),
)
CFG = dict(CFGS[SCALE], vocab=256, scale=SCALE)
if os.environ.get("TOKENS"):                      # smoke-test override
    CFG["tokens"] = int(os.environ["TOKENS"])

# On CUDA the max-batch search is empirical: double until it OOMs. On CPU there
# is no fixed ceiling to hit, so we declare one and solve for it using measured
# activation bytes. Same quantity, obtained two honest ways.
MEM_BUDGET = float(os.environ.get("MEM_BUDGET_GIB", "2.0")) * 1024 ** 3
BASE_BATCH = int(os.environ.get("BASE_BATCH", "0"))    # 0 = search
VARIANTS = ["additive", "euler", "midpoint"]

print(f"scale={SCALE}  device={DEV}  cfg={CFG}")


# %% [markdown]
# ## 1. Data
#
# TinyStories, streamed and written out as raw bytes. Byte-level means the
# "tokenizer" is `str.encode`, which costs nothing and downloads nothing.

# %%
def get_data(n_bytes, path=None):
    path = Path(path or HERE / f"data_{n_bytes // 1_000_000}mb.bin")
    if path.exists() and path.stat().st_size >= n_bytes:
        return np.memmap(path, dtype=np.uint8, mode="r")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    written, t0 = 0, time.time()
    with open(path, "wb") as f:
        for row in ds:
            b = (row["text"].strip() + "\n\n").encode("utf-8", "ignore")
            f.write(b)
            written += len(b)
            if written >= n_bytes:
                break
    print(f"wrote {path.name}: {written / 1e6:.1f} MB in {time.time() - t0:.0f}s")
    return np.memmap(path, dtype=np.uint8, mode="r")


# a little slack so the last window is always in range
DATA = get_data(int(CFG["tokens"] * 1.02) + 10_000)
print(f"corpus: {len(DATA) / 1e6:.1f} M bytes")


def get_batch(bs, seq, gen):
    ix = torch.randint(len(DATA) - seq - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(DATA[i:i + seq].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(DATA[i + 1:i + 1 + seq].astype(np.int64)) for i in ix])
    return x.to(DEV), y.to(DEV)


# %% [markdown]
# ## 2. The two sublayers
#
# `F` is attention, `G` is the MLP. Both are ordinary pre-norm sublayers — the
# reversible machinery never touches their internals, which is the point: any
# transformer block can be made reversible by changing how the residual stream
# is wired, not by changing the block.
#
# No dropout anywhere. That is not laziness — the reversible backward *recomputes*
# `F` and `G`, and a recomputation that draws different random numbers than the
# forward pass gives silently wrong gradients. Reversible models need either no
# randomness or carefully replayed RNG state.

# %%
class Attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg["d"])
        self.qkv = nn.Linear(cfg["d"], 3 * cfg["d"], bias=False)
        self.proj = nn.Linear(cfg["d"], cfg["d"], bias=False)
        self.nh = cfg["n_head"]

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln(x)).chunk(3, -1)
        q, k, v = [z.view(B, T, self.nh, D // self.nh).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg["d"])
        self.fc1 = nn.Linear(cfg["d"], cfg["d_ff"], bias=False)
        self.fc2 = nn.Linear(cfg["d_ff"], cfg["d"], bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.ln(x))))


# %% [markdown]
# ## 3. One coupling primitive, three variants
#
# Every reversible scheme here is a sequence of the same elementary move: update
# one stream using the other.
#
# ```
# forward   s[t] <- s[t] + c * fn(s[1-t])
# inverse   s[t] <- s[t] - c * fn(s[1-t])
# ```
#
# The inverse needs `s[1-t]`, which the step did not modify, so it is always
# available. That single fact is the whole of reversibility, and it means the
# variants differ only in which steps they schedule:
#
# | variant | steps per block | sublayer evals |
# |---|---|---|
# | `additive` | `x1 += F(x2)`, `x2 += G(x1)` | 2 |
# | `euler` | same with step size `h` | 2 |
# | `midpoint` | `x2 += (h/2)G(x1)`, `x1 += h F(x2)`, `x2 += (h/2)G(x1)` | 3 |
#
# `additive` is RevNet/Reformer and is `euler` with `h = 1`. `midpoint` is Strang
# splitting — the symmetric, second-order version — and costs 50% more compute
# for the same parameters.

# %%
class Step:
    __slots__ = ("tgt", "fn", "coef")

    def __init__(self, tgt, fn, coef):
        self.tgt, self.fn, self.coef = tgt, fn, coef


def build_steps(blocks, variant, h):
    steps = []
    for f, g in blocks:
        if variant in ("additive", "euler"):
            steps += [Step(0, f, h), Step(1, g, h)]
        elif variant == "midpoint":
            steps += [Step(1, g, h / 2), Step(0, f, h), Step(1, g, h / 2)]
        else:
            raise ValueError(variant)
    return steps


class RevFn(torch.autograd.Function):
    """Run the whole stack storing nothing, then walk it backwards."""

    @staticmethod
    def forward(ctx, x1, x2, steps, *params):
        ctx.steps = steps
        ctx.n_params = len(params)
        with torch.no_grad():
            s = [x1, x2]
            for st in steps:
                s[st.tgt] = s[st.tgt] + st.coef * st.fn(s[1 - st.tgt])
        ctx.save_for_backward(s[0], s[1])
        return s[0], s[1]

    @staticmethod
    def backward(ctx, d1, d2):
        s = [t.detach() for t in ctx.saved_tensors]
        d = [d1, d2]
        for st in reversed(ctx.steps):
            src = 1 - st.tgt
            with torch.enable_grad():
                s_ = s[src].detach().requires_grad_(True)
                out = st.coef * st.fn(s_)
            with torch.no_grad():
                s[st.tgt] = s[st.tgt] - out.detach()        # undo this step
            torch.autograd.backward(out, d[st.tgt])         # params and s_.grad
            d[src] = d[src] + s_.grad
        # parameter grads were accumulated by hand above, so None for each
        return (d[0], d[1], None) + (None,) * ctx.n_params


# %% [markdown]
# ## 4. The model
#
# Two modes over identical parameters. `standard` is a single residual stream
# with ordinary autograd. `reversible` duplicates the embedding into two streams,
# runs the couplings, and averages at the end — so the parameter count is the
# same and only the memory strategy differs.
#
# They are not the same function, and the README says so. The clean control for
# "does reversibility change the answer" is running the *same two-stream model*
# with ordinary autograd, which §5 does.

# %%
class ByteLM(nn.Module):
    def __init__(self, cfg, mode="standard", variant="additive", h=1.0):
        super().__init__()
        self.cfg, self.mode, self.variant, self.h = cfg, mode, variant, h
        self.emb = nn.Embedding(cfg["vocab"], cfg["d"])
        self.pos = nn.Parameter(torch.zeros(cfg["seq"], cfg["d"]))
        self.blocks = nn.ModuleList()
        for _ in range(cfg["n_layer"]):
            self.blocks.append(nn.ModuleList([Attn(cfg), MLP(cfg)]))
        self.lnf = nn.LayerNorm(cfg["d"])
        self.head = nn.Linear(cfg["d"], cfg["vocab"], bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def pairs(self):
        return [(b[0], b[1]) for b in self.blocks]

    def trunk(self, x, force_autograd=False):
        if self.mode == "standard":
            for f, g in self.pairs():
                x = x + f(x)
                x = x + g(x)
            return x
        steps = build_steps(self.pairs(), self.variant, self.h)
        if force_autograd:                       # same math, ordinary autograd
            s = [x, x]
            for st in steps:
                s[st.tgt] = s[st.tgt] + st.coef * st.fn(s[1 - st.tgt])
            return (s[0] + s[1]) * 0.5
        y1, y2 = RevFn.apply(x, x, steps, *self.parameters())
        return (y1 + y2) * 0.5

    def forward(self, idx, targets=None, force_autograd=False):
        x = self.emb(idx) + self.pos[: idx.shape[1]]
        logits = self.head(self.lnf(self.trunk(x, force_autograd)))
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, self.cfg["vocab"]), targets.reshape(-1))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


_probe = ByteLM(CFG)
print(f"parameters: {n_params(_probe):,}")
del _probe

# %% [markdown]
# ## 5. Is it actually reversible?
#
# Two checks, and they are the load-bearing part of the whole notebook. A
# coupling that is only approximately invertible still trains, still produces a
# falling loss curve, and is simply wrong — there is no error message.
#
# 1. **Reconstruction.** Run forward, then run the inverse, and compare against
#    the original input.
# 2. **Gradients.** Compare the reversible backward against an ordinary autograd
#    backward through the identical computation.
#
# Check 1 is run in fp32 *and* fp16, because that is where reversibility gets
# interesting: the inverse subtracts what the forward added, and in low precision
# those do not cancel exactly. The error compounds over depth.

# %%
def check_reversibility(cfg, variant, h=1.0, dtype=torch.float32):
    torch.manual_seed(0)
    m = ByteLM(cfg, "reversible", variant, h).to(dtype)
    steps = build_steps(m.pairs(), variant, h)
    x1 = torch.randn(2, cfg["seq"], cfg["d"], dtype=dtype)
    x2 = x1.clone()
    s = [x1.clone(), x2.clone()]
    with torch.no_grad():
        for st in steps:
            s[st.tgt] = s[st.tgt] + st.coef * st.fn(s[1 - st.tgt])
        for st in reversed(steps):               # walk it back
            s[st.tgt] = s[st.tgt] - st.coef * st.fn(s[1 - st.tgt])
    err = max((s[0] - x1).abs().max().item(), (s[1] - x2).abs().max().item())
    return err / x1.abs().max().item()


def check_gradients(cfg, variant, h=1.0):
    torch.manual_seed(0)
    m = ByteLM(cfg, "reversible", variant, h)
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, cfg["vocab"], (2, cfg["seq"]), generator=g)
    tgt = torch.randint(0, cfg["vocab"], (2, cfg["seq"]), generator=g)

    m.zero_grad()
    m(idx, tgt, force_autograd=True).backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}

    m.zero_grad()
    loss = m(idx, tgt)
    loss.backward()
    worst, where = 0.0, ""
    for n, p in m.named_parameters():
        scale = ref[n].abs().max().item() or 1.0
        rel = (p.grad - ref[n]).abs().max().item() / scale
        if rel > worst:
            worst, where = rel, n
    return worst, where, loss.item()


SMALL = dict(CFG, d=64, n_layer=4, n_head=4, d_ff=128, seq=32)
CHECKS = {}
for v in VARIANTS:
    h = 1.0 if v == "additive" else 0.5
    try:
        fp16 = check_reversibility(SMALL, v, h, torch.float16)
    except (RuntimeError, NotImplementedError):    # fp16 kernels are CPU-patchy
        fp16 = None
    rel, where, _ = check_gradients(SMALL, v, h)
    CHECKS[v] = dict(h=h, recon_fp32=check_reversibility(SMALL, v, h),
                     recon_fp16=fp16, grad_rel=rel, grad_worst_param=where)

print(f"{'variant':10s} {'h':>4s} {'recon fp32':>12s} {'recon fp16':>12s} {'grad rel':>12s}")
for v, c in CHECKS.items():
    f16 = "n/a" if c["recon_fp16"] is None else f"{c['recon_fp16']:.2e}"
    print(f"{v:10s} {c['h']:4.1f} {c['recon_fp32']:12.2e} {f16:>12s} {c['grad_rel']:12.2e}")

# %% [markdown]
# ## 6. Activation memory, measured
#
# The claim is that reversible activation memory is flat in depth. Rather than
# assert it, count what autograd keeps alive: `saved_tensors_hooks` fires on every
# tensor stored for the backward pass, so summing unique storages *is* the
# activation footprint. Parameters are excluded — they are counted separately.

# %%
def activation_bytes(model, idx, tgt, force_autograd=False):
    pptr = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen, total = set(), 0

    def pack(t):
        nonlocal total
        dp = t.untyped_storage().data_ptr()
        if dp not in pptr and dp not in seen:
            seen.add(dp)
            total += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(idx, tgt, force_autograd=force_autograd)
    return total


def depth_sweep(depths=(2, 4, 8, 16)):
    out = {"standard": [], "reversible": []}
    g = torch.Generator().manual_seed(2)
    for L in depths:
        c = dict(CFG, n_layer=L)
        idx = torch.randint(0, c["vocab"], (1, c["seq"]), generator=g)
        tgt = torch.randint(0, c["vocab"], (1, c["seq"]), generator=g)
        for mode in out:
            m = ByteLM(c, mode, "additive", 1.0)
            out[mode].append(activation_bytes(m, idx, tgt))
            del m
    return dict(depths=list(depths), **out)


DEPTH = depth_sweep()
MB = 1024 ** 2
print(f"\n{'depth':>6s} {'standard':>14s} {'reversible':>14s} {'ratio':>8s}")
for i, L in enumerate(DEPTH["depths"]):
    a, b = DEPTH["standard"][i], DEPTH["reversible"][i]
    print(f"{L:6d} {a / MB:11.2f}MiB {b / MB:11.2f}MiB {a / b:8.2f}x")

# %% [markdown]
# ## 7. How big a batch fits
#
# On CUDA this is empirical: double the batch until it OOMs, then back off. On
# CPU there is no device ceiling to collide with, so we declare a budget and
# divide by the measured per-sequence activation cost. The CPU path is what makes
# the rehearsal scale meaningful; the CUDA path is what runs on Colab.

# %%
def max_batch(mode, variant="additive", h=1.0, budget=MEM_BUDGET, cap=8192):
    m = ByteLM(CFG, mode, variant, h).to(DEV)
    g = torch.Generator().manual_seed(3)
    if DEV == "cuda":
        best, bs = 1, 1
        while bs <= cap:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x, y = get_batch(bs, CFG["seq"], g)
                m(x, y).backward()
                m.zero_grad(set_to_none=True)
                best, bs = bs, bs * 2
            except torch.cuda.OutOfMemoryError:
                break
        del m
        torch.cuda.empty_cache()
        return best
    idx = torch.randint(0, CFG["vocab"], (1, CFG["seq"]), generator=g)
    per_seq = activation_bytes(m, idx, idx)
    state = 16 * n_params(m)                      # params + grads + Adam m,v
    del m
    return max(1, min(cap, int((budget - state) // max(per_seq, 1))))


# %% [markdown]
# ## 8. Training
#
# Every run sees the same number of tokens, the same schedule shape, and the same
# seed. Only the batch size and the layer wiring change.

# %%
def train(mode, batch, variant="additive", h=1.0, tokens=None, tag=""):
    tokens = tokens or CFG["tokens"]
    torch.manual_seed(0)
    m = ByteLM(CFG, mode, variant, h).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], betas=(0.9, 0.95),
                            weight_decay=0.1)
    per_step = batch * CFG["seq"]
    steps = max(1, tokens // per_step)
    warm = min(CFG["warmup"], steps // 10 or 1)
    g = torch.Generator().manual_seed(7)
    amp = DEV == "cuda"

    if DEV == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    losses, t0, seen = [], time.time(), 0
    for i in range(steps):
        lr = CFG["lr"] * ((i + 1) / warm if i < warm
                          else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = get_batch(batch, CFG["seq"], g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = m(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        seen += per_step
        if i % max(1, steps // 10) == 0 or i == steps - 1:
            print(f"  [{tag}] step {i + 1}/{steps}  loss {losses[-1]:.4f}  "
                  f"{seen / (time.time() - t0):,.0f} tok/s", flush=True)

    secs = time.time() - t0
    m = m.cpu()
    idx = torch.randint(0, CFG["vocab"], (1, CFG["seq"]), generator=g)
    act = activation_bytes(m, idx, idx)
    peak = (torch.cuda.max_memory_allocated() if DEV == "cuda"
            else 16 * n_params(m) + act * batch)
    del m
    # average the tail, but never so much of it that step 1's loss is included --
    # short runs would otherwise be scored on their own warmup, and the shortest
    # run would be penalised hardest, which is exactly backwards
    tail = max(1, min(20, steps // 5))
    return dict(
        mode=mode, variant=variant, h=h, batch=batch, steps=steps, tail=tail,
        tokens_seen=seen, final_loss=float(np.mean(losses[-tail:])),
        loss_curve=[float(v) for v in losses],
        tokens_per_sec=seen / secs, seconds=secs,
        peak_bytes=int(peak), act_bytes_per_seq=int(act),
        peak_measured=DEV == "cuda",
    )


# %% [markdown]
# ## 9. Pick the variant
#
# `midpoint` does three sublayer evaluations per block where `additive` and
# `euler` do two. Comparing the three at an **equal token budget** would be rigged
# in its favour — it is doing 50% more work per token, so of course it gets a
# lower loss. That comparison answers a question nobody has.
#
# The question that matters is which variant is ahead after the same amount of
# *compute*. So each variant is first benchmarked for throughput, then given a
# token budget sized to spend the same wall-clock, and each runs a complete cosine
# schedule over its own budget. Whoever is lowest at the end wins on merit.

# %%
PROBE_SECONDS = float(os.environ.get("PROBE_SECONDS", "300"))
PROBE_BATCH = max(1, min(16, max_batch("reversible")))


def throughput(mode, batch, variant="additive", h=1.0, iters=4):
    """Tokens/s from a few real steps, used to equalise the probe budgets."""
    torch.manual_seed(0)
    m = ByteLM(CFG, mode, variant, h).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-9)
    g = torch.Generator().manual_seed(11)
    x, y = get_batch(batch, CFG["seq"], g)
    m(x, y).backward()                             # warm the kernels
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    for _ in range(iters):
        loss = m(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    tps = iters * batch * CFG["seq"] / (time.time() - t0)
    del m, opt
    return tps


VSWEEP, TPS = {}, {}
for v in VARIANTS:
    h = 1.0 if v == "additive" else 0.5
    TPS[v] = throughput("reversible", PROBE_BATCH, v, h)
    budget = max(PROBE_BATCH * CFG["seq"] * 8, int(PROBE_SECONDS * TPS[v]))
    print(f"  [probe/{v}] {TPS[v]:,.0f} tok/s measured -> {budget:,} token budget")
    VSWEEP[v] = train("reversible", PROBE_BATCH, v, h, budget, tag=f"probe/{v}")

print(f"\n{'variant':10s} {'tok/s':>10s} {'rel cost':>9s} {'tokens':>10s} "
      f"{'secs':>7s} {'loss':>8s}")
for v, r in VSWEEP.items():
    print(f"{v:10s} {r['tokens_per_sec']:10,.0f} "
          f"{VSWEEP['additive']['tokens_per_sec'] / r['tokens_per_sec']:8.2f}x "
          f"{r['tokens_seen']:10,d} {r['seconds']:7.0f} {r['final_loss']:8.4f}")

def window(c):
    return max(1, min(20, len(c) // 5))


def loss_at_time(r, secs):
    """Loss this run had reached `secs` into training.

    Sizing each variant's budget from a 4-step throughput benchmark does NOT
    equalise wall-clock -- benchmark jitter handed one variant 28% more time than
    another the first time this ran, and it changed which variant 'won'. So the
    comparison is made by reading every curve at the shortest run's elapsed time,
    assuming uniform step cost within a run.
    """
    c = r["loss_curve"]
    step_t = r["seconds"] / max(1, r["steps"])
    i = min(len(c) - 1, max(0, int(secs / step_t) - 1))
    w = window(c)
    return float(np.mean(c[max(0, i - w + 1):i + 1]))


def loss_at(r, tok):
    """Loss this run had reached after `tok` tokens, same tail rule as above."""
    c = r["loss_curve"]
    per = r["batch"] * CFG["seq"]
    i = min(len(c) - 1, max(0, int(tok // per) - 1))
    w = max(1, min(20, len(c) // 5))
    return float(np.mean(c[max(0, i - w + 1):i + 1]))


# the equal-token ranking, read off the same runs at the shortest run's budget.
# Each variant follows its own cosine schedule, so mid-curve losses are only
# roughly comparable -- enough to see whether the two rankings disagree.
COMMON_TOK = min(r["tokens_seen"] for r in VSWEEP.values())
EQ_TOKEN = {v: loss_at(r, COMMON_TOK) for v, r in VSWEEP.items()}
PER_TOKEN_BEST = min(EQ_TOKEN, key=EQ_TOKEN.get)

COMMON_SECS = min(r["seconds"] for r in VSWEEP.values())
EQ_TIME = {v: loss_at_time(r, COMMON_SECS) for v, r in VSWEEP.items()}
BEST = min(EQ_TIME, key=EQ_TIME.get)
BEST_H = VSWEEP[BEST]["h"]

print(f"\nbudgets actually spent: " +
      "  ".join(f"{v}={VSWEEP[v]['seconds']:.0f}s" for v in VARIANTS))
print(f"equal-token  @ {COMMON_TOK:,} tok: " +
      "  ".join(f"{v}={EQ_TOKEN[v]:.4f}" for v in VARIANTS) +
      f"   -> {PER_TOKEN_BEST}")
print(f"equal-time   @ {COMMON_SECS:.0f}s: " +
      "  ".join(f"{v}={EQ_TIME[v]:.4f}" for v in VARIANTS) + f"   -> {BEST}")
print(f"\nchosen: {BEST} (h={BEST_H})  <- used for the three runs")

if os.environ.get("PROBE_ONLY"):
    (HERE / f"results_{SCALE}_probe.json").write_text(json.dumps(dict(
        scale=SCALE, checks=CHECKS, depth_sweep=DEPTH, probe_batch=PROBE_BATCH,
        probe_seconds=PROBE_SECONDS, throughput=TPS,
        variant_sweep={k: {kk: vv for kk, vv in v.items() if kk != "loss_curve"}
                       for k, v in VSWEEP.items()},
        equal_token_loss=EQ_TOKEN, equal_token_budget=COMMON_TOK,
        equal_time_loss=EQ_TIME, equal_time_budget=COMMON_SECS,
        per_token_winner=PER_TOKEN_BEST, chosen=BEST,
    ), indent=2), encoding="utf-8")
    print(f"wrote results_{SCALE}_probe.json")
    raise SystemExit(0)

# %% [markdown]
# ## 10. The three runs

# %%
B_STD = BASE_BATCH or max_batch("standard")
B_REV = max_batch("reversible", BEST, BEST_H)
print(f"max batch  standard={B_STD}  reversible={B_REV}  ({B_REV / B_STD:.2f}x)")

RUNS = {}
RUNS["1_standard"] = train("standard", B_STD, tag="1/standard")
RUNS["2_reversible"] = train("reversible", B_STD, BEST, BEST_H, tag="2/reversible")
RUNS["3_reversible_maxbatch"] = train("reversible", B_REV, BEST, BEST_H,
                                      tag="3/reversible-maxbatch")

print(f"\n{'run':26s} {'batch':>6s} {'loss':>8s} {'tok/s':>10s} {'peak':>12s}")
for k, r in RUNS.items():
    print(f"{k:26s} {r['batch']:6d} {r['final_loss']:8.4f} "
          f"{r['tokens_per_sec']:10,.0f} {r['peak_bytes'] / MB:9.1f}MiB")

# %% [markdown]
# ## 11. Plots and results

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(v, k=25):
    if len(v) < k:
        return v
    c = np.convolve(v, np.ones(k) / k, mode="valid")
    return list(c)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for k, r in RUNS.items():
    a1.plot(smooth(r["loss_curve"]), label=f"{k}  (bs {r['batch']})", lw=1.3)
a1.set_xlabel("step"); a1.set_ylabel("loss (smoothed)")
a1.set_title("Three runs, same token budget")
a1.legend(frameon=False, fontsize=8)
for k, r in RUNS.items():
    toks = np.linspace(0, r["tokens_seen"], len(smooth(r["loss_curve"])))
    a2.plot(toks / 1e6, smooth(r["loss_curve"]), label=k, lw=1.3)
a2.set_xlabel("million tokens"); a2.set_ylabel("loss (smoothed)")
a2.set_title("Same, against tokens seen")
a2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(HERE / f"loss_{SCALE}.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(DEPTH["depths"], [v / MB for v in DEPTH["standard"]], "o-",
        label="standard", color="#444")
ax.plot(DEPTH["depths"], [v / MB for v in DEPTH["reversible"]], "o-",
        label="reversible", color="#d62728")
ax.set_xlabel("layers"); ax.set_ylabel("activation MiB (one sequence)")
ax.set_title("Activation memory vs depth")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(HERE / f"depth_{SCALE}.png", dpi=130)
plt.close(fig)
print("wrote 2 figures")

RESULTS = dict(
    scale=SCALE, device=DEV, cfg=CFG,
    params=n_params(ByteLM(CFG)),
    checks=CHECKS, depth_sweep=DEPTH,
    variant_sweep={k: {kk: vv for kk, vv in v.items() if kk != "loss_curve"}
                   for k, v in VSWEEP.items()},
    chosen_variant=BEST, chosen_h=BEST_H,
    variant_throughput=TPS, equal_token_loss=EQ_TOKEN,
    equal_token_budget=COMMON_TOK, per_token_winner=PER_TOKEN_BEST,
    equal_time_loss=EQ_TIME, equal_time_budget=COMMON_SECS,
    probe_seconds=PROBE_SECONDS, probe_batch=PROBE_BATCH,
    max_batch=dict(standard=B_STD, reversible=B_REV),
    runs=RUNS,
    mem_budget_bytes=MEM_BUDGET if DEV == "cpu" else None,
)
(HERE / f"results_{SCALE}.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
print(f"wrote results_{SCALE}.json")
