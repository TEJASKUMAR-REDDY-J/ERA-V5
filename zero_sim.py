"""32 virtual GPUs, one flat parameter buffer, and the four ZeRO stages.

Run it:

    python zero_sim.py          # prints everything, writes results.json + 4 pngs
    python to_notebook.py       # regenerates zero_sim.ipynb from this file

This file is the source of truth. The notebook is generated from it so the two
cannot drift.
"""

# %% [markdown]
# # ZeRO on 32 virtual GPUs
#
# Thirty-two ranks, a 4.2M-parameter transformer, and the same hand-written Adam
# from the previous session — now sharded four different ways.
#
# **What a simulation on one CPU can and cannot show.** This machine has 4 cores.
# Running 32 ranks on it cannot be faster than running 1, and nothing here claims
# it is. The two things ZeRO actually changes are **bytes resident per device**
# and **bytes moved per step**, and both are exactly measurable without any real
# parallelism:
#
# - memory is measured by really building the shards and summing real tensor
#   sizes, cross-checked against process RSS;
# - communication is counted inside the collectives as they run;
# - correctness is checked by requiring every stage to land on the same weights
#   as a single-device run.
#
# Wall-clock is reported, but only to make the point that it is the wrong metric
# here.

# %%
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

torch.set_num_threads(4)

HERE = Path(__file__).parent if "__file__" in dir() else Path(".")

CFG = dict(vocab=4096, d_model=192, n_head=6, n_layer=6, d_ff=768, seq=64)
WORLD = 32          # virtual GPUs
MICRO_BS = 2        # sequences per rank per step
STEPS = 30
LR = 3e-3
B1, B2, EPS = 0.9, 0.999, 1e-8
SEED = 0

STAGES = ["ddp", "zero1", "zero2", "zero3"]
LABEL = {
    "ddp":   "ZeRO-0 / DDP  (nothing sharded)",
    "zero1": "ZeRO-1        (optimizer states sharded)",
    "zero2": "ZeRO-2        (+ gradients sharded)",
    "zero3": "ZeRO-3        (+ parameters sharded)",
}

MB = 1024 ** 2


def fmt(b):
    return f"{b / MB:8.2f} MiB"


# %% [markdown]
# ## 1. The model, as one flat buffer
#
# Every ZeRO implementation flattens the parameters into a single contiguous
# buffer before sharding it. That is not a micro-optimisation, it is what makes
# sharding tractable: a shard becomes `theta[r*S:(r+1)*S]`, a plain slice, and
# the collectives operate on one tensor instead of a few hundred.
#
# The cost is that shard boundaries fall in the middle of parameter tensors — a
# shard is a bag of bytes, not a list of layers. That is fine for the optimizer
# (Adam is elementwise, so a slice of the state is a valid optimizer problem) and
# it is the reason ZeRO-1 and ZeRO-2 are free lunches. It becomes awkward only in
# ZeRO-3, where the forward pass needs whole tensors back; §7 deals with that.

# %%
def param_spec(cfg):
    """Ordered (name, shape). Order defines the flat layout, so it matters."""
    d, ff, V, L = cfg["d_model"], cfg["d_ff"], cfg["vocab"], cfg["n_layer"]
    spec = [("emb", (V, d)), ("pos", (cfg["seq"], d))]
    for i in range(L):
        spec += [
            (f"b{i}.ln1_g", (d,)), (f"b{i}.ln1_b", (d,)),
            (f"b{i}.qkv", (d, 3 * d)), (f"b{i}.proj", (d, d)),
            (f"b{i}.ln2_g", (d,)), (f"b{i}.ln2_b", (d,)),
            (f"b{i}.fc1", (d, ff)), (f"b{i}.fc2", (ff, d)),
        ]
    spec += [("lnf_g", (d,)), ("lnf_b", (d,)), ("head", (d, V))]
    return spec


SPEC = param_spec(CFG)
SIZES = [math.prod(s) for _, s in SPEC]
PSI = sum(SIZES)                                  # parameter count
SHARD = math.ceil(PSI / WORLD)
PSI_PAD = SHARD * WORLD                           # flat buffer is padded to N*S

# byte offset of every parameter inside the flat buffer
OFFSET, _o = {}, 0
for (name, shape), n in zip(SPEC, SIZES):
    OFFSET[name] = (_o, _o + n, shape)
    _o += n

# contiguous groups, used by the layer-by-layer gather in §7
GROUPS = [("embed", ["emb", "pos"])]
GROUPS += [(f"block{i}", [n for n, _ in SPEC if n.startswith(f"b{i}.")])
           for i in range(CFG["n_layer"])]
GROUPS += [("final", ["lnf_g", "lnf_b", "head"])]
GROUP_RANGE = {g: (OFFSET[ns[0]][0], OFFSET[ns[-1]][1]) for g, ns in GROUPS}


def init_theta(seed=SEED):
    g = torch.Generator().manual_seed(seed)
    t = torch.zeros(PSI_PAD)
    for name, (lo, hi, _) in OFFSET.items():
        if name.endswith("_g"):
            t[lo:hi] = 1.0                        # LayerNorm gain
        elif name.endswith("_b"):
            t[lo:hi] = 0.0                        # LayerNorm bias
        else:
            t[lo:hi] = torch.randn(hi - lo, generator=g) * 0.02
    return t


def views(theta):
    """Name -> view into the flat buffer. No copies."""
    return {n: theta[lo:hi].view(sh) for n, (lo, hi, sh) in OFFSET.items()}


def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + eps) * g + b


def block(h, P, i, cfg):
    B, T, d = h.shape
    nh = cfg["n_head"]
    a = ln(h, P[f"b{i}.ln1_g"], P[f"b{i}.ln1_b"])
    q, k, v = (a @ P[f"b{i}.qkv"]).chunk(3, -1)
    q, k, v = [z.view(B, T, nh, d // nh).transpose(1, 2) for z in (q, k, v)]
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    h = h + y.transpose(1, 2).reshape(B, T, d) @ P[f"b{i}.proj"]
    m = ln(h, P[f"b{i}.ln2_g"], P[f"b{i}.ln2_b"])
    return h + F.gelu(m @ P[f"b{i}.fc1"]) @ P[f"b{i}.fc2"]


def embed(P, x):
    return P["emb"][x] + P["pos"][: x.shape[1]]


def head(P, h, cfg):
    return ln(h, P["lnf_g"], P["lnf_b"]) @ P["head"]


def loss_fn(P, x, y, cfg):
    h = embed(P, x)
    for i in range(cfg["n_layer"]):
        h = block(h, P, i, cfg)
    return F.cross_entropy(head(P, h, cfg).reshape(-1, cfg["vocab"]), y.reshape(-1))


PERIOD, N_MOTIF = 8, 32


def make_batches(world, micro_bs, steps, cfg, seed=1234):
    """Fixed data, identical across every run, so loss curves are comparable.

    Each sequence is one of N_MOTIF fixed motifs, randomly rotated and tiled to
    fill the window. Predicting it means copying from PERIOD positions back, which
    a 6-layer transformer picks up in a few dozen steps.

    That matters more than it looks. On unlearnable noise every stage sits flat at
    ln(V) forever, and "all five loss curves coincide" would pass even if the
    sharding were completely broken. A curve that actually moves is what makes the
    agreement check worth running.

    Only N_MOTIF * PERIOD = 256 of the 4096 token ids ever appear, so most of the
    embedding and head rows see little or no gradient. That turns out to matter
    in §6, where Adam's behaviour on near-zero gradients is the whole story.
    """
    g = torch.Generator().manual_seed(seed)
    n = world * micro_bs
    pool = torch.randint(0, cfg["vocab"], (N_MOTIF, PERIOD), generator=g)
    idx = torch.randint(0, N_MOTIF, (steps, n), generator=g)
    rot = torch.randint(0, PERIOD, (steps, n), generator=g)
    m = torch.gather(pool[idx], 2,
                     (torch.arange(PERIOD) + rot[..., None]) % PERIOD)
    reps = (cfg["seq"] + 1 + PERIOD - 1) // PERIOD
    return m.repeat(1, 1, reps)[:, :, : cfg["seq"] + 1].contiguous()


BATCHES = make_batches(WORLD, MICRO_BS, STEPS, CFG)

print(f"parameters Psi          : {PSI:,}")
print(f"padded to N*S           : {PSI_PAD:,}  (shard {SHARD:,}, waste {PSI_PAD - PSI})")
print(f"fp32 params             : {fmt(4 * PSI)}")
print(f"full model state (16B/p): {fmt(16 * PSI)}")
print(f"tokens / step           : {WORLD * MICRO_BS * CFG['seq']:,}")

# %% [markdown]
# ## 2. A virtual GPU
#
# A rank is not a process here. It is an object that owns some bytes and knows
# how many it owns — which is the only property of a GPU that ZeRO changes.
#
# Memory is **ledgered**, not read off an allocator, and that is deliberate. All
# 32 ranks live in one process, so RSS would report the simulator's footprint,
# not a device's. The ledger instead records real tensor sizes at the real shard
# boundaries: if the arithmetic that splits the buffer were wrong, the ledger
# would be wrong in the same way the training run would be. §6 checks the ledger
# against RSS by allocating one rank's state for real.
#
# The one thing the ledger does *not* charge is the simulator's own convenience:
# for ZeRO-0/1/2 every rank holds a bit-identical copy of `theta`, so the code
# keeps one copy and charges 32. That invariant is what DDP maintains, and it is
# also a hard limit of this machine — 32 real replicas of the full model state
# would be 2.02 GiB, which is the entire point of the exercise.


# %%
class VGPU:
    """One virtual device: a byte ledger with a peak tracker."""

    def __init__(self, rank):
        self.rank = rank
        self.live = {}
        self.peak = 0

    def hold(self, key, nbytes):
        self.live[key] = nbytes
        self.peak = max(self.peak, self.resident)
        return nbytes

    def release(self, key):
        self.live.pop(key, None)

    @property
    def resident(self):
        return sum(self.live.values())


class Fabric:
    """The interconnect. Counts bytes each rank puts on the wire.

    Volumes are the ring-algorithm costs, which is what NCCL and every
    production ZeRO implementation actually run:

        all_reduce      2(N-1)/N * bytes   per rank
        reduce_scatter   (N-1)/N  * bytes   per rank
        all_gather       (N-1)/N  * bytes   per rank

    all_reduce == reduce_scatter + all_gather, in cost and in effect. That
    identity is the whole reason ZeRO-1 and ZeRO-2 are free.
    """

    def __init__(self, world):
        self.world = world
        self.bytes_per_rank = 0
        self.log = []

    def _charge(self, op, nbytes, factor):
        moved = factor * (self.world - 1) / self.world * nbytes
        self.bytes_per_rank += moved
        self.log.append((op, nbytes, moved))
        return moved

    def all_reduce(self, t):
        self._charge("all_reduce", t.numel() * t.element_size(), 2)
        return t

    def reduce_scatter(self, t):
        self._charge("reduce_scatter", t.numel() * t.element_size(), 1)
        return t

    def all_gather(self, t):
        self._charge("all_gather", t.numel() * t.element_size(), 1)
        return t


def check_collectives():
    """reduce_scatter then all_gather must equal all_reduce, bit for bit."""
    g = torch.Generator().manual_seed(7)
    per_rank = [torch.randn(PSI_PAD, generator=g) for _ in range(4)]
    allred = torch.stack(per_rank).mean(0)
    S = PSI_PAD // 4
    scattered = [torch.stack([p[r * S:(r + 1) * S] for p in per_rank]).mean(0)
                 for r in range(4)]
    gathered = torch.cat(scattered)
    return (allred - gathered).abs().max().item()


COLL_ERR = check_collectives()
print(f"reduce_scatter+all_gather vs all_reduce : max|diff| = {COLL_ERR:.3e}")

# %% [markdown]
# ## 3. Sharded Adam
#
# Same update as the previous session, now operating on a slice. Adam is
# elementwise, so a shard of the state is a self-contained optimizer problem —
# rank *r* can step its slice knowing nothing about the other 31. This is the
# single fact that makes ZeRO-1 work, and it is why ZeRO-1 is numerically
# identical to DDP rather than an approximation of it.


# %%
class AdamShard:
    """Adam state for one rank's slice of the flat buffer."""

    def __init__(self, n):
        self.m = torch.zeros(n)
        self.v = torch.zeros(n)
        self.t = 0

    def step(self, w, g, lr):
        self.t += 1
        self.m.mul_(B1).add_(g, alpha=1 - B1)
        self.v.mul_(B2).addcmul_(g, g, value=1 - B2)
        mhat = self.m / (1 - B1 ** self.t)
        vhat = self.v / (1 - B2 ** self.t)
        w.sub_(lr * mhat / (vhat.sqrt() + EPS))

    @property
    def nbytes(self):
        return (self.m.numel() + self.v.numel()) * 4


# %% [markdown]
# ## 4. Activation memory, measured
#
# ZeRO shards *model state*. It does not touch activations. Saying that is easy;
# the number is worth measuring, because it sets the floor that no amount of
# sharding gets under.
#
# The measurement is exact rather than estimated: autograd's `saved_tensors_hooks`
# fires on every tensor the graph keeps alive for backward. Summing their unique
# storages is the activation footprint, by definition. Views into the parameter
# buffer are excluded — those are parameters, already counted.


# %%
def activation_bytes(theta, x, y, cfg):
    seen, total = set(), 0
    theta_ptr = theta.untyped_storage().data_ptr()

    def pack(t):
        nonlocal total
        st = t.untyped_storage()
        if st.data_ptr() != theta_ptr and st.data_ptr() not in seen:
            seen.add(st.data_ptr())
            total += t.numel() * t.element_size()
        return t

    th = theta.detach().clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss_fn(views(th), x, y, cfg)
    return total


_th = init_theta()
_b = BATCHES[0][:MICRO_BS]
ACT_BYTES = activation_bytes(_th, _b[:, :-1], _b[:, 1:], CFG)
print(f"activation bytes / rank (micro_bs={MICRO_BS}): {fmt(ACT_BYTES)}")

# %% [markdown]
# ## 5. The four stages
#
# All four run the same `theta`, the same data, and the same Adam. What differs
# is who stores what and which collective moves it.
#
# | stage | params | grads | opt state | collectives per step |
# |---|---|---|---|---|
# | ZeRO-0 (DDP) | full | full | full | all_reduce(grads) |
# | ZeRO-1 | full | full | **/N** | all_reduce(grads), all_gather(params) |
# | ZeRO-2 | full | **/N** | **/N** | reduce_scatter(grads), all_gather(params) |
# | ZeRO-3 | **/N** | **/N** | **/N** | all_gather(params) ×2, reduce_scatter(grads) |
#
# ZeRO-1 as written here all-reduces and then throws away 31/32 of the result,
# which is what the original paper describes. Every real implementation
# reduce-scatters instead, for the same cost, which is why ZeRO-1 and ZeRO-2 move
# identical bytes.
#
# ZeRO-0/1/2 share the parameter buffer in this simulator (see §2). ZeRO-3 does
# not: its 32 shards are 32 separate tensors, gradients reach them through the
# gather, and nothing reconstructs a full `theta` for the optimizer. If the shard
# arithmetic were wrong anywhere, ZeRO-3 would diverge from the rest — which is
# the check §6 runs.


# %%
def run_stage(stage, world=WORLD, steps=STEPS, cfg=CFG, lr=LR):
    gpus = [VGPU(r) for r in range(world)]
    net = Fabric(world)
    S = SHARD
    P4 = 4 * PSI_PAD                                  # bytes for one full copy
    losses, t0 = [], time.time()

    if stage == "zero3":
        full = init_theta()
        shards = [full[r * S:(r + 1) * S].clone().requires_grad_(True)
                  for r in range(world)]
        opt = [AdamShard(S) for _ in range(world)]
        for g_ in gpus:
            g_.hold("params", 4 * S)
            g_.hold("grads", 4 * S)          # shards[r].grad, one shard wide
            g_.hold("opt", 8 * S)
            g_.hold("activations", ACT_BYTES)
    else:
        theta = init_theta()                          # one copy, charged 32×
        opt = [AdamShard(S) for _ in range(world)]
        for g_ in gpus:
            g_.hold("params", P4)
            g_.hold("activations", ACT_BYTES)
            g_.hold("grads", P4 if stage in ("ddp", "zero1") else 4 * S)
            g_.hold("opt", P4 * 2 if stage == "ddp" else 8 * S)

    for step in range(steps):
        batch = BATCHES[step]
        step_loss = 0.0

        if stage == "zero3":
            for s in shards:
                if s.grad is not None:
                    s.grad = None
            for r in range(world):
                b = batch[r * MICRO_BS:(r + 1) * MICRO_BS]
                flat = torch.cat(shards)              # all_gather, differentiable
                gpus[r].hold("gathered", 4 * PSI_PAD)
                l = loss_fn(views(flat), b[:, :-1], b[:, 1:], cfg)
                (l / world).backward()                # grads land on the owners
                gpus[r].release("gathered")
                step_loss += l.item() / world
            net.all_gather(torch.empty(PSI_PAD))      # forward gather
            net.all_gather(torch.empty(PSI_PAD))      # backward re-gather
            net.reduce_scatter(torch.empty(PSI_PAD))  # grads already scattered
            with torch.no_grad():
                for r in range(world):
                    opt[r].step(shards[r].data, shards[r].grad, lr)
        else:
            gacc = torch.zeros(PSI_PAD)
            for r in range(world):
                b = batch[r * MICRO_BS:(r + 1) * MICRO_BS]
                th = theta.detach().clone().requires_grad_(True)
                l = loss_fn(views(th), b[:, :-1], b[:, 1:], cfg)
                l.backward()
                gacc += th.grad
                step_loss += l.item() / world
            gacc /= world
            if stage == "ddp":
                net.all_reduce(torch.empty(PSI_PAD))
            else:
                net.reduce_scatter(torch.empty(PSI_PAD))
                net.all_gather(torch.empty(PSI_PAD))
            with torch.no_grad():
                for r in range(world):
                    sl = slice(r * S, (r + 1) * S)
                    opt[r].step(theta[sl], gacc[sl], lr)

        losses.append(step_loss)

    final = torch.cat([s.detach() for s in shards]) if stage == "zero3" else theta
    return dict(
        stage=stage, losses=losses, theta=final, secs=time.time() - t0,
        peak=max(g.peak for g in gpus), steady=gpus[0].resident,
        breakdown={k: v for k, v in gpus[0].live.items()},
        comm_per_rank=net.bytes_per_rank / steps,
        ops=[o for o, _, _ in net.log[: len(net.log) // steps]],
    )


def run_single_device(steps=STEPS, cfg=CFG, lr=LR):
    """One GPU, global batch, no collectives. The reference answer."""
    theta = init_theta()
    opt = AdamShard(PSI_PAD)
    losses, t0 = [], time.time()
    for step in range(steps):
        b = BATCHES[step]
        th = theta.detach().clone().requires_grad_(True)
        l = loss_fn(views(th), b[:, :-1], b[:, 1:], cfg)
        l.backward()
        with torch.no_grad():
            opt.step(theta, th.grad, lr)
        losses.append(l.item())
    peak = 16 * PSI_PAD + ACT_BYTES * WORLD
    return dict(stage="1 GPU", losses=losses, theta=theta, secs=time.time() - t0,
                peak=peak, steady=peak, breakdown={}, comm_per_rank=0.0, ops=[])


# %% [markdown]
# ## 6. Run everything and check it agrees
#
# The claim under test: **sharding changes where bytes live, not what the model
# computes.** Four stages, one single-device reference, identical data and seed.
# If any of them lands on different weights, the sharding is wrong.

# %%
RUNS = {"single": run_single_device()}
for st in STAGES:
    RUNS[st] = run_stage(st)
    print(f"{LABEL[st]:44s} loss {RUNS[st]['losses'][-1]:.6f}  "
          f"{RUNS[st]['secs']:5.1f}s  peak/rank {fmt(RUNS[st]['peak'])}")

REF = RUNS["single"]["theta"]
AGREE = {k: (RUNS[k]["theta"] - REF).abs().max().item() for k in RUNS}
LOSS_DIFF = {k: max(abs(a - b) for a, b in zip(RUNS[k]["losses"], RUNS["single"]["losses"]))
             for k in RUNS}

print("\nagreement with the single-device run")
print(f"{'run':10s} {'max|dw|':>12s} {'max|dloss|':>12s}")
for k in RUNS:
    print(f"{k:10s} {AGREE[k]:12.3e} {LOSS_DIFF[k]:12.3e}")

# and with each other: the four stages accumulate the same 32 micro-batches in
# the same order, so they should not differ by even one bit
CROSS = {st: (RUNS[st]["theta"] - RUNS["ddp"]["theta"]).abs().max().item()
         for st in STAGES}
print("\nagreement between stages (vs ZeRO-0/DDP)")
for st, d in CROSS.items():
    print(f"{st:10s} {d:12.3e}")

# %% [markdown]
# The loss agrees to ~1e-6 but the **weights** disagree at ~1e-3, which is 5% of
# their own scale. That looks alarming, so it is worth chasing down rather than
# waving at "float error".
#
# The reference reduces one cross-entropy over 64 sequences; the sharded runs
# reduce 32 losses over 2 sequences each and average them. Same quantity,
# different summation order, so float32 gives a different last bit. The question
# is how a last-bit difference in the gradient becomes 1e-3 in the weights.
#
# The answer is Adam, and it is the same normalisation from the previous session.
# The update is `lr * mhat / (sqrt(vhat) + eps)` — **scale-free**. For a
# parameter whose gradient is genuinely near zero (an embedding row for a token
# that never appeared, say), `mhat` and `sqrt(vhat)` are both noise, their ratio
# is O(1) noise, and the update saturates near ±lr regardless of how small the
# gradient was. A sign flip in the last bit of such a gradient therefore moves
# the weight by up to `2*lr = 6e-3`.
#
# So the test below compares **gradients** instead of post-Adam weights. That
# isolates the thing sharding is responsible for.

# %%
def grad_agreement():
    """Do the 32 micro-batch gradients sum to the single global-batch gradient?"""
    b = BATCHES[0]
    th = init_theta().requires_grad_(True)
    loss_fn(views(th), b[:, :-1], b[:, 1:], CFG).backward()
    ref = th.grad.clone()

    acc = torch.zeros(PSI_PAD)
    for r in range(WORLD):
        mb = b[r * MICRO_BS:(r + 1) * MICRO_BS]
        t2 = init_theta().requires_grad_(True)
        loss_fn(views(t2), mb[:, :-1], mb[:, 1:], CFG).backward()
        acc += t2.grad
    acc /= WORLD

    S = SHARD
    shards = [init_theta()[r * S:(r + 1) * S].clone().requires_grad_(True)
              for r in range(WORLD)]
    for r in range(WORLD):
        mb = b[r * MICRO_BS:(r + 1) * MICRO_BS]
        flat = torch.cat(shards)
        (loss_fn(views(flat), mb[:, :-1], mb[:, 1:], CFG) / WORLD).backward()
    z3 = torch.cat([s.grad for s in shards])

    scale = ref.abs().max().item()
    return dict(
        grad_scale=scale,
        ddp_abs=(acc - ref).abs().max().item(),
        ddp_rel=(acc - ref).abs().max().item() / scale,
        zero3_abs=(z3 - ref).abs().max().item(),
        zero3_rel=(z3 - ref).abs().max().item() / scale,
    )


GRADS = grad_agreement()
print(f"gradient scale (max|g|)            : {GRADS['grad_scale']:.4e}")
print(f"32 micro-batches vs global batch   : {GRADS['ddp_abs']:.3e} "
      f"({GRADS['ddp_rel']:.2e} relative)")
print(f"ZeRO-3 gather/scatter vs global    : {GRADS['zero3_abs']:.3e} "
      f"({GRADS['zero3_rel']:.2e} relative)")


# %%
def amplification():
    """Where the 1e-3 weight differences live: parameters with no gradient."""
    b = BATCHES[0]
    th = init_theta().requires_grad_(True)
    loss_fn(views(th), b[:, :-1], b[:, 1:], CFG).backward()
    g = th.grad.abs()
    dw = (RUNS["ddp"]["theta"] - REF).abs()
    big = dw > 1e-4
    return dict(
        n_big=int(big.sum()), frac=float(big.float().mean()),
        max_dw_over_lr=dw.max().item() / LR,
        median_grad_big=float(g[big].median()) if big.any() else 0.0,
        median_grad_all=float(g.median()),
    )


AMP = amplification()
print(f"\nparameters with |dw| > 1e-4        : {AMP['n_big']:,} "
      f"({AMP['frac'] * 100:.2f}%)")
print(f"largest |dw| as a multiple of lr   : {AMP['max_dw_over_lr']:.2f}")
print(f"median |grad| of those parameters  : {AMP['median_grad_big']:.3e}")
print(f"median |grad| over all parameters  : {AMP['median_grad_all']:.3e}")

# %% [markdown]
# The gradients agree to ~1e-7 relative, which is float32 doing its job. The
# handful of weights that drift are the ones whose gradient is four to five
# orders of magnitude below the median, and the drift stays well under the `2*lr`
# ceiling the argument above allows for. So the ceiling is a bound, not a
# prediction; what it explains is the *direction*, that tiny gradients give large
# weight differences and not the other way round.
#
# The endpoints are worth stating, because the naive version of this argument is
# wrong. A parameter with *exactly* zero gradient does not drift at all: `m` and
# `v` stay at zero, the update is `0 / (0 + eps) = 0`, and both runs leave it
# alone. The completely unused embedding rows are in that category. The drift
# lives in the band between — small enough that reduction order sets the low
# bits, large enough to escape `eps`.
#
# Sharding is exact. Adam is a noise amplifier for dead parameters. A real
# sharding bug would show up in the gradient comparison, which is why that is the
# check worth running and the post-Adam weight comparison is not.

# %% [markdown]
# ## 7. ZeRO-3's transient: gathering one layer at a time
#
# The ZeRO-3 run above gathers all of `theta` at once. That is correct but it
# throws away most of the benefit — while the gathered copy is alive, the rank is
# holding `4Psi/N + 4Psi` bytes, which is *worse* than ZeRO-2.
#
# Real ZeRO-3 (and FSDP) never materialise the whole model. They gather one layer,
# use it, and drop it before gathering the next, so the transient is one layer
# rather than one model. Implementing that under autograd has a catch: a gathered
# tensor that gets freed after the forward pass is still needed for backward.
#
# The way out is recomputation. Gathering **inside** a checkpointed function means
# the gather is not saved — it is re-run during backward, used, and freed again.
# Gradients still reach the shards because `torch.cat` is differentiable, so the
# backward pass through the gather *is* the reduce-scatter. This is the actual
# FSDP design, in about fifteen lines.


# %%
def gather_range(shards, lo, hi, S):
    """Rebuild theta[lo:hi] from only the shards that overlap it."""
    r0, r1 = lo // S, (hi - 1) // S
    return torch.cat([shards[r][max(lo - r * S, 0):min(hi - r * S, S)]
                      for r in range(r0, r1 + 1)])


def zero3_layerwise(shards, x, y, cfg, S):
    """Forward with one group resident at a time. Returns (loss, peak transient)."""
    peak = 0

    def group_views(name):
        lo, hi = GROUP_RANGE[name]
        flat = gather_range(shards, lo, hi, S)
        return {n: flat[OFFSET[n][0] - lo:OFFSET[n][1] - lo].view(OFFSET[n][2])
                for n in dict(GROUPS)[name]}, (hi - lo) * 4

    def stage_fn(name, fn):
        nonlocal peak

        def inner(h, *sh):
            P, nb = group_views(name)
            return fn(h, P)
        lo, hi = GROUP_RANGE[name]
        peak = max(peak, (hi - lo) * 4)
        return inner

    P, nb = group_views("embed")
    peak = max(peak, nb)
    h = embed(P, x)
    del P
    for i in range(cfg["n_layer"]):
        fn = stage_fn(f"block{i}", lambda hh, PP, i=i: block(hh, PP, i, cfg))
        h = torch.utils.checkpoint.checkpoint(fn, h, *shards, use_reentrant=False)
    fn = stage_fn("final", lambda hh, PP: head(PP, hh, cfg))
    logits = torch.utils.checkpoint.checkpoint(fn, h, *shards, use_reentrant=False)
    loss = F.cross_entropy(logits.reshape(-1, cfg["vocab"]), y.reshape(-1))
    return loss, peak


def check_layerwise():
    full = init_theta()
    S = SHARD
    shards = [full[r * S:(r + 1) * S].clone().requires_grad_(True) for r in range(WORLD)]
    b = BATCHES[0][:MICRO_BS]

    ptrs = {sh.untyped_storage().data_ptr() for sh in shards}
    seen, act = set(), 0

    def pack(t):
        nonlocal act
        d = t.untyped_storage().data_ptr()
        if d not in ptrs and d not in seen:
            seen.add(d)
            act += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss, peak = zero3_layerwise(shards, b[:, :-1], b[:, 1:], CFG, S)
    loss.backward()
    got = torch.cat([s.grad for s in shards])

    th = full.clone().requires_grad_(True)
    ref_loss = loss_fn(views(th), b[:, :-1], b[:, 1:], CFG)
    ref_loss.backward()

    return dict(
        loss=loss.item(), ref_loss=ref_loss.item(),
        grad_err=(got - th.grad).abs().max().item(),
        transient_flat=4 * PSI_PAD, transient_layerwise=peak,
        act_flat=ACT_BYTES, act_layerwise=act,
        steady_layerwise=16 * SHARD,
        peak_layerwise=16 * SHARD + peak + act,
    )


LAYERWISE = check_layerwise()
print(f"layerwise loss {LAYERWISE['loss']:.6f} vs reference {LAYERWISE['ref_loss']:.6f}")
print(f"gradient max|diff| vs unsharded backward : {LAYERWISE['grad_err']:.3e}")
print(f"transient, gather-all  : {fmt(LAYERWISE['transient_flat'])}")
print(f"transient, layer-wise  : {fmt(LAYERWISE['transient_layerwise'])}  "
      f"({LAYERWISE['transient_flat'] / LAYERWISE['transient_layerwise']:.1f}x smaller)")
print(f"activations, flat      : {fmt(LAYERWISE['act_flat'])}")
print(f"activations, layer-wise: {fmt(LAYERWISE['act_layerwise'])}  "
      f"(recomputation, not ZeRO)")
print(f"ZeRO-3 peak/rank, layer-wise : {fmt(LAYERWISE['peak_layerwise'])}")

# %% [markdown]
# ## 8. Is the ledger telling the truth?
#
# One rank's state for each stage, allocated for real, measured against process
# RSS. If the ledger were fantasy this is where it would show.

# %%
CHILD = """
import sys, torch, psutil, gc
n_p, n_g, n_o = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
p = psutil.Process(); gc.collect()
base = p.memory_info().rss
hold = [torch.zeros(n_p), torch.zeros(n_g), torch.zeros(n_o), torch.zeros(n_o)]
for t in hold:
    t.add_(1.0)
print(p.memory_info().rss - base)
"""


def rss_check():
    """Allocate one rank's state in a *fresh* process and read RSS.

    Measuring this in-process gives nonsense: the training runs above have
    already handed the allocator plenty of freed pages, so a new tensor grows
    RSS by zero. A clean child has no such history.
    """
    import subprocess
    import sys
    out = {}
    for st in STAGES:
        n_p = SHARD if st == "zero3" else PSI_PAD
        n_g = PSI_PAD if st in ("ddp", "zero1") else SHARD
        n_o = PSI_PAD if st == "ddp" else SHARD
        try:
            r = subprocess.run([sys.executable, "-c", CHILD, str(n_p), str(n_g), str(n_o)],
                               capture_output=True, text=True, timeout=180)
            measured = int(r.stdout.strip().splitlines()[-1])
        except Exception as e:                       # psutil missing, sandbox, ...
            return None
        ledger = 4 * (n_p + n_g + 2 * n_o)
        out[st] = dict(ledger=ledger, rss=measured, ratio=measured / ledger)
    return out


RSS = rss_check()
if RSS:
    print(f"{'stage':8s} {'ledger':>14s} {'RSS delta':>14s} {'ratio':>7s}")
    for st, d in RSS.items():
        print(f"{st:8s} {fmt(d['ledger'])} {fmt(d['rss'])} {d['ratio']:7.2f}")
else:
    print("RSS cross-check skipped (psutil unavailable)")

# %% [markdown]
# ## 9. Scaling: where ZeRO stops helping
#
# Per-rank bytes as a function of world size, for each stage. The model-state
# term falls like 1/N. The activation term does not fall at all, so every curve
# flattens onto it — and past that point buying more GPUs stops buying you
# headroom and you need activation checkpointing, or a smaller micro-batch, or
# tensor parallelism.


# %%
def model_state_bytes(stage, n, psi=PSI):
    p, g, o = 4 * psi, 4 * psi, 8 * psi
    if stage == "zero1":
        o /= n
    elif stage == "zero2":
        g /= n; o /= n
    elif stage == "zero3":
        p /= n; g /= n; o /= n
    return p + g + o


def comm_units(stage, n):
    """Bytes on the wire per rank per step, in units of 4*Psi."""
    f = (n - 1) / n
    return {"ddp": 2 * f, "zero1": 2 * f, "zero2": 2 * f, "zero3": 3 * f}[stage]


# the counted bytes must equal the closed form, or one of the two is wrong
COMM_CHECK = {st: (RUNS[st]["comm_per_rank"], comm_units(st, WORLD) * 4 * PSI_PAD)
              for st in STAGES}
for _st, (_got, _want) in COMM_CHECK.items():
    assert abs(_got - _want) < 1, f"{_st}: counted {_got} vs formula {_want}"
print("counted communication matches the ring closed form for all four stages")

# and the ledger each rank actually kept must equal the closed form too
MEM_CHECK = {st: (RUNS[st]["steady"], model_state_bytes(st, WORLD, PSI_PAD) + ACT_BYTES)
             for st in STAGES}
for _st, (_got, _want) in MEM_CHECK.items():
    assert abs(_got - _want) < 1, f"{_st}: ledger {_got} vs formula {_want}"
print("ledgered per-rank memory matches the closed form for all four stages")

NS = [1, 2, 4, 8, 16, 32]
SWEEP = {st: [model_state_bytes(st, n) + ACT_BYTES for n in NS] for st in STAGES}

print(f"\nper-rank total bytes (model state + activations)")
print(f"{'N':>4s} " + " ".join(f"{st:>12s}" for st in STAGES))
for i, n in enumerate(NS):
    print(f"{n:>4d} " + " ".join(f"{SWEEP[st][i] / MB:12.2f}" for st in STAGES))

FLOOR = {st: ACT_BYTES / SWEEP[st][-1] for st in STAGES}
print(f"\nat N=32 activations are {FLOOR['zero3'] * 100:.1f}% of ZeRO-3's per-rank total")

# %% [markdown]
# ## 10. Plots

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"ddp": "#444444", "zero1": "#1f77b4", "zero2": "#2ca02c", "zero3": "#d62728"}
SHORT = {"ddp": "ZeRO-0/DDP", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}

# --- memory breakdown -------------------------------------------------------
BARS = STAGES + ["zero3_lw"]
BAR_LABEL = [SHORT[s] for s in STAGES] + ["ZeRO-3\nlayerwise"]


def bar_parts(st):
    if st == "zero3_lw":
        return dict(params=4 * SHARD, grads=4 * SHARD, opt=8 * SHARD,
                    activations=LAYERWISE["act_layerwise"],
                    transient=LAYERWISE["transient_layerwise"])
    n_p = SHARD if st == "zero3" else PSI_PAD
    n_g = PSI_PAD if st in ("ddp", "zero1") else SHARD
    n_o = PSI_PAD if st == "ddp" else SHARD
    return dict(params=4 * n_p, grads=4 * n_g, opt=8 * n_o,
                activations=ACT_BYTES,
                transient=4 * PSI_PAD if st == "zero3" else 0)


fig, ax = plt.subplots(figsize=(9, 4.8))
parts = ["params", "grads", "opt", "activations", "transient"]
pcol = ["#4c72b0", "#dd8452", "#937860", "#c44e52", "#8172b3"]
bottom = [0.0] * len(BARS)
for name, c in zip(parts, pcol):
    vals = [bar_parts(st)[name] / MB for st in BARS]
    ax.bar(BAR_LABEL, vals, bottom=bottom, label=name, color=c)
    bottom = [b + v for b, v in zip(bottom, vals)]
for i, b in enumerate(bottom):
    ax.text(i, b + 1, f"{b:.1f}", ha="center", fontsize=9)
ax.set_ylim(0, max(bottom) * 1.14)
ax.set_ylabel("MiB per virtual GPU")
ax.set_title(f"Per-rank memory, N={WORLD}, Psi={PSI/1e6:.2f}M "
             f"(transient = the gather held during fwd/bwd)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "memory_by_stage.png", dpi=130)
plt.close(fig)

# --- scaling ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for st in STAGES:
    ax.plot(NS, [v / MB for v in SWEEP[st]], "o-", color=COLORS[st], label=SHORT[st])
ax.axhline(ACT_BYTES / MB, ls="--", c="gray", lw=1)
ax.text(1.1, ACT_BYTES / MB * 1.15, "activation floor (never sharded)",
        fontsize=8, color="gray")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(NS); ax.set_xticklabels(NS)
ax.set_xlabel("virtual GPUs"); ax.set_ylabel("MiB per GPU")
ax.set_title("Per-rank memory vs world size")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "memory_vs_ranks.png", dpi=130)
plt.close(fig)

# --- communication ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
meas = [RUNS[st]["comm_per_rank"] / MB for st in STAGES]
ax.bar([SHORT[s] for s in STAGES], meas, color=[COLORS[s] for s in STAGES])
for i, st in enumerate(STAGES):
    ax.text(i, meas[i] + 0.6, f"{meas[i]:.1f} MiB\n{comm_units(st, WORLD):.2f}x 4Psi",
            ha="center", fontsize=9)
ax.set_ylabel("MiB on the wire per rank per step")
ax.set_ylim(0, max(meas) * 1.3)
ax.set_title("Communication volume (counted, not estimated)")
fig.tight_layout()
fig.savefig(HERE / "comm_volume.png", dpi=130)
plt.close(fig)

# --- loss agreement ---------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
xs = range(1, STEPS + 1)
a1.plot(xs, RUNS["single"]["losses"], "k-", lw=3, alpha=.3, label="1 GPU (reference)")
for st in STAGES:
    a1.plot(xs, RUNS[st]["losses"], "--", color=COLORS[st], lw=1.2, label=SHORT[st])
a1.set_xlabel("step"); a1.set_ylabel("loss"); a1.set_title("All five curves coincide")
a1.legend(frameon=False, fontsize=8)
for st in STAGES:
    d = [abs(a - b) for a, b in zip(RUNS[st]["losses"], RUNS["single"]["losses"])]
    a2.semilogy(xs, [max(v, 1e-12) for v in d], "o-", color=COLORS[st],
                ms=3, label=SHORT[st])
a2.set_xlabel("step"); a2.set_ylabel("|loss - reference|")
a2.set_title("Divergence is float32 reduction order")
a2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "loss_agreement.png", dpi=130)
plt.close(fig)
print("wrote 4 figures")

# %% [markdown]
# ## 11. Summary

# %%
rows = []
for st in STAGES:
    r = RUNS[st]
    rows.append(dict(
        stage=SHORT[st],
        steady_mib=r["steady"] / MB,
        peak_mib=r["peak"] / MB,
        model_state_mib=(r["steady"] - ACT_BYTES) / MB,
        vs_ddp=RUNS["ddp"]["peak"] / r["peak"],
        comm_mib=r["comm_per_rank"] / MB,
        comm_x=comm_units(st, WORLD) / comm_units("ddp", WORLD),
        secs=r["secs"],
        final_loss=r["losses"][-1],
        max_dw=AGREE[st],
    ))
rows.append(dict(
    stage="ZeRO-3 layerwise",
    steady_mib=LAYERWISE["steady_layerwise"] / MB,
    peak_mib=LAYERWISE["peak_layerwise"] / MB,
    model_state_mib=LAYERWISE["steady_layerwise"] / MB,
    vs_ddp=RUNS["ddp"]["peak"] / LAYERWISE["peak_layerwise"],
    comm_mib=RUNS["zero3"]["comm_per_rank"] / MB,
    comm_x=comm_units("zero3", WORLD) / comm_units("ddp", WORLD),
    secs=float("nan"), final_loss=LAYERWISE["loss"], max_dw=LAYERWISE["grad_err"],
))

print(f"\n{'stage':17s} {'steady':>10s} {'peak':>10s} {'vs DDP':>8s} "
      f"{'comm/step':>11s} {'vs DDP':>8s} {'secs':>7s} {'max|dw|':>10s}")
for r in rows:
    print(f"{r['stage']:17s} {r['steady_mib']:7.2f}MiB {r['peak_mib']:7.2f}MiB "
          f"{r['vs_ddp']:7.2f}x {r['comm_mib']:8.2f}MiB {r['comm_x']:7.2f}x "
          f"{r['secs']:7.1f} {r['max_dw']:10.2e}")

RESULTS = dict(
    config=CFG, world=WORLD, micro_bs=MICRO_BS, steps=STEPS, lr=LR,
    psi=PSI, psi_padded=PSI_PAD, shard=SHARD,
    activation_bytes=ACT_BYTES,
    collective_identity_err=COLL_ERR,
    stages={st: dict(
        losses=RUNS[st]["losses"],
        peak_bytes=RUNS[st]["peak"],
        steady_bytes=RUNS[st]["steady"],
        breakdown=RUNS[st]["breakdown"],
        comm_bytes_per_rank_per_step=RUNS[st]["comm_per_rank"],
        collectives=RUNS[st]["ops"],
        seconds=RUNS[st]["secs"],
        max_weight_diff_vs_single=AGREE[st],
        max_loss_diff_vs_single=LOSS_DIFF[st],
    ) for st in STAGES},
    single_device=dict(losses=RUNS["single"]["losses"], seconds=RUNS["single"]["secs"]),
    layerwise=LAYERWISE,
    cross_stage_max_weight_diff=CROSS,
    grad_agreement=GRADS,
    adam_amplification=AMP,
    rss_check=RSS,
    comm_counted_vs_formula={k: list(v) for k, v in COMM_CHECK.items()},
    sweep=dict(world_sizes=NS, per_rank_bytes=SWEEP),
    summary=rows,
)
(HERE / "results.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
print("wrote results.json")
