# %% [markdown]
# # Making a training loop tell the truth about itself
#
# A small model, a real loop, and six things it is made to confess.
#
# **Tejaskumar Reddy J** - ERA V5
#
# Runs top to bottom on CPU or GPU. Every number quoted in the README is printed
# by this file; nothing is typed by hand.

# %%
import os
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math, json, time, struct
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device {DEV} | threads {torch.get_num_threads()}")
if DEV.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

TOK = AutoTokenizer.from_pretrained("gpt2")
V = len(TOK)
PAD = TOK.eos_token_id
IGNORE = -100
print(f"vocab {V}")


# %% [markdown]
# ## The model and the data
#
# Deliberately small: the point is the instrumentation, not the model.

# %%
class Cfg:
    d_model, n_layer, n_head, block = 192, 4, 4, 128
C = Cfg()


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


class TinyGPT(nn.Module):
    def __init__(s, vocab=V, cfg=C):
        super().__init__()
        s.cfg = cfg
        s.emb = nn.Embedding(vocab, cfg.d_model)
        s.pos = nn.Embedding(cfg.block, cfg.d_model)
        s.blocks = nn.ModuleList([Blk(cfg.d_model, cfg.n_head) for _ in range(cfg.n_layer)])
        s.norm = nn.LayerNorm(cfg.d_model)
        s.head = nn.Linear(cfg.d_model, vocab, bias=False)
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(s, ids):
        B, T = ids.shape
        x = s.emb(ids) + s.pos(torch.arange(T, device=ids.device))[None]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device,
                                       dtype=x.dtype), 1)
        for b in s.blocks:
            x = b(x, causal)
        return s.head(s.norm(x))


from datasets import load_dataset

_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
_docs = [t.strip() for t in _ds["text"] if len(t.strip()) > 200]
SEQ = 64


def make_data(n_seq=768, seq=SEQ):
    ids = []
    for d in _docs:
        ids.extend(TOK(d)["input_ids"])
        if len(ids) > n_seq * seq + seq:
            break
    return torch.tensor(ids[: n_seq * seq], dtype=torch.long).view(-1, seq).to(DEV)


DATA = make_data()
print(f"data {tuple(DATA.shape)} = {DATA.numel():,} tokens")

model = TinyGPT().to(DEV)
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"params {N_PARAMS:,}")


# %% [markdown]
# ---
# # 1. Every tensor shape in the step, and what each dimension means

# %%
ids = DATA[:8].clone()
logits = model(ids)
inp, tgt = logits[:, :-1], ids[:, 1:]
flat_lg, flat_tg = inp.reshape(-1, V), tgt.reshape(-1)
loss = F.cross_entropy(flat_lg, flat_tg)
loss.backward()

rows = [
    ("ids", ids.shape, "B=batch (independent sequences), T=time (token position within a sequence)"),
    ("emb(ids)", model.emb(ids).shape, "B, T, D. D=d_model, the residual-stream width each token carries"),
    ("logits", logits.shape, "B, T, V. V=vocab. One score per possible next token, at every position"),
    ("logits[:, :-1]", inp.shape, "drop the last position: it would predict a token past the end of the sequence"),
    ("ids[:, 1:]", tgt.shape, "drop the first token: nothing precedes it, so nothing predicts it"),
    ("flat logits", flat_lg.shape, "(B*(T-1), V). cross_entropy wants 2-D: one row per prediction"),
    ("flat targets", flat_tg.shape, "(B*(T-1),). one gold token id per prediction"),
    ("loss", loss.shape, "scalar (). the mean negative log-probability over all contributing positions"),
    ("emb.weight", model.emb.weight.shape, "V, D. one learned vector per vocabulary entry"),
    ("emb.weight.grad", model.emb.weight.grad.shape, "same shape as the weight, always. one gradient per parameter"),
    ("head.weight", model.head.weight.shape, "V, D. projects the residual stream back to vocabulary scores"),
    ("blocks[0].att.in_proj_weight", model.blocks[0].att.in_proj_weight.shape,
     "3D, D. Q, K and V projections stacked into one matrix"),
]
print(f"{'tensor':<30}{'shape':<20}what each dimension means")
print("-" * 118)
for n, sh, why in rows:
    print(f"{n:<30}{str(tuple(sh)):<20}{why}")
print(f"\nloss = {loss.item():.6f} over {flat_tg.numel()} predictions")
model.zero_grad(set_to_none=True)


# %% [markdown]
# ---
# # 2. Verify one gradient by hand
#
# Nudge one weight by `h`, measure how the loss actually changed, and compare
# against what `backward()` claimed.
#
# Two things matter for this to agree to several decimals:
#
# - **float64.** In fp32 the loss itself carries about 7 significant digits, so a
#   difference of two nearby losses keeps only 2-3 of them. The check would look
#   broken when the gradient is fine.
# - **Central difference.** `(L(w+h) - L(w-h)) / 2h` has error O(h^2); the
#   one-sided version has error O(h) and costs a couple of decimals for free.

# %%
gc_model = TinyGPT().to(DEV).double()          # float64 throughout
gc_ids = DATA[:4].clone()


def full_loss(m):
    lg = m(gc_ids)
    return F.cross_entropy(lg[:, :-1].reshape(-1, V), gc_ids[:, 1:].reshape(-1))


L0 = full_loss(gc_model)
gc_model.zero_grad(set_to_none=True)
L0.backward()

# pick a weight that actually participates: a used row of the embedding table
tok_id = int(gc_ids[0, 3])
W = gc_model.emb.weight
i, j = tok_id, 7
analytic = W.grad[i, j].item()

print(f"checking emb.weight[{i}, {j}]  (token {TOK.decode([tok_id])!r})")
print(f"analytic gradient from backward() : {analytic:.16f}\n")
print(f"{'h':<12}{'numeric (central diff)':<26}{'abs err':<14}{'rel err':<14}matching decimals")
print("-" * 92)

best = None
for h in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    with torch.no_grad():
        orig = W[i, j].item()
        W[i, j] = orig + h
        Lp = full_loss(gc_model).item()
        W[i, j] = orig - h
        Lm = full_loss(gc_model).item()
        W[i, j] = orig
    numeric = (Lp - Lm) / (2 * h)
    abs_err = abs(numeric - analytic)
    rel_err = abs_err / max(abs(analytic), 1e-30)
    dec = 0 if rel_err <= 0 else max(0, -math.log10(rel_err))
    print(f"{h:<12.0e}{numeric:<26.16f}{abs_err:<14.2e}{rel_err:<14.2e}{dec:.1f}")
    if best is None or rel_err < best[2]:
        best = (h, numeric, rel_err, dec)

GRAD_H, GRAD_NUM, GRAD_REL, GRAD_DEC = best
print(f"\nbest agreement at h={GRAD_H:.0e}: {GRAD_DEC:.1f} matching decimal digits "
      f"(relative error {GRAD_REL:.2e})")
print("\nThe error curve is a V: too large an h and the O(h^2) truncation term")
print("dominates; too small and catastrophic cancellation in (L+ - L-) does. The")
print("minimum sits near h = eps^(1/3), which for float64 is about 6e-6.")

# same check in float32, to show why the dtype matters
f32 = TinyGPT().to(DEV)
f32.load_state_dict({k: v.float() for k, v in gc_model.state_dict().items()})


def full_loss32(m):
    lg = m(gc_ids)
    return F.cross_entropy(lg[:, :-1].reshape(-1, V), gc_ids[:, 1:].reshape(-1))


f32.zero_grad(set_to_none=True)
full_loss32(f32).backward()
W32 = f32.emb.weight
a32 = W32.grad[i, j].item()
with torch.no_grad():
    o = W32[i, j].item(); h = 1e-4
    W32[i, j] = o + h; Lp = full_loss32(f32).item()
    W32[i, j] = o - h; Lm = full_loss32(f32).item()
    W32[i, j] = o
n32 = (Lp - Lm) / (2 * h)
rel32 = abs(n32 - a32) / max(abs(a32), 1e-30)
print(f"\nsame check in float32 at h=1e-4: rel err {rel32:.2e} "
      f"({max(0, -math.log10(max(rel32, 1e-30))):.1f} decimals) - the gradient is fine,")
print("the measurement is not.")


# %% [markdown]
# ---
# # 3. Break gradient accumulation on purpose
#
# Micro-batches rarely hold the same number of real tokens. When they do not,
# **averaging the per-micro-batch averages is wrong**: it gives every micro-batch
# an equal vote regardless of how many tokens it contains.
#
# ```
# wrong :  L = (1/K) * SUM_k  mean_over_tokens_in_k(loss)
# right :  L = SUM_k sum(loss_k)  /  SUM_k n_tokens_k
# ```
#
# A micro-batch holding 20 tokens moves the update as much as one holding 200.

# %%
MICRO_LENS = [63, 5, 48, 7]           # deliberately unequal, in tokens per row
K = len(MICRO_LENS)


def micro_batches(step, bs=4):
    """K micro-batches whose valid lengths differ a lot. Same data for both arms."""
    g = torch.Generator(device="cpu").manual_seed(1234 + step)
    out = []
    for L in MICRO_LENS:
        idx = torch.randint(0, DATA.shape[0], (bs,), generator=g)
        chunk = DATA[idx.to(DATA.device)][:, : L + 1].clone()
        out.append(chunk)
    return out


def losses_for(m, chunk):
    """Returns (sum_of_token_losses, n_tokens) for one micro-batch."""
    lg = m(chunk)
    per = F.cross_entropy(lg[:, :-1].reshape(-1, V), chunk[:, 1:].reshape(-1),
                          reduction="none")
    return per.sum(), per.numel()


def run_accum(mode, steps=250, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = TinyGPT().to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    hist = []
    for st in range(steps):
        opt.zero_grad(set_to_none=True)
        mb = micro_batches(st)
        if mode == "wrong":
            # average of averages: each micro-batch contributes 1/K
            for chunk in mb:
                s, n = losses_for(m, chunk)
                ((s / n) / K).backward()
        else:
            # token-weighted: one denominator for the whole optimiser step
            total_n = sum(c[:, 1:].numel() for c in mb)
            for chunk in mb:
                s, _ = losses_for(m, chunk)
                (s / total_n).backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9).item()  # measure, do not clip
        opt.step()
        with torch.no_grad():                       # honest yardstick: token-weighted
            ss = nn_ = 0
            for chunk in mb:
                s, n = losses_for(m, chunk)
                ss += s.item(); nn_ += n
        hist.append({"step": st, "true_loss": ss / nn_, "grad_norm": gn})
    return hist


print("micro-batch token counts per step:",
      [4 * L for L in MICRO_LENS], f" (ratio {max(MICRO_LENS) / min(MICRO_LENS):.1f}x)")

# the gap on a single step, before any training
probe = TinyGPT().to(DEV)
mb = micro_batches(0)
with torch.no_grad():
    parts = [losses_for(probe, c) for c in mb]
avg_of_avgs = sum((s / n) for s, n in parts).item() / K
token_weighted = sum(s for s, _ in parts).item() / sum(n for _, n in parts)
print(f"\nsame data, one step, untrained model:")
print(f"  average of averages : {avg_of_avgs:.6f}")
print(f"  token weighted      : {token_weighted:.6f}")
print(f"  difference          : {avg_of_avgs - token_weighted:+.6f}")

print("\ntraining both arms on identical data...")
h_wrong = run_accum("wrong")
h_right = run_accum("right")
GAP = h_wrong[-1]["true_loss"] - h_right[-1]["true_loss"]
print(f"\nfinal token-weighted loss (the honest yardstick for both):")
print(f"  average-of-averages arm : {h_wrong[-1]['true_loss']:.4f}")
print(f"  token-weighted arm      : {h_right[-1]['true_loss']:.4f}")
print(f"  gap                     : {GAP:+.4f}")


# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot([h["step"] for h in h_wrong], [h["true_loss"] for h in h_wrong],
           label="average of averages (wrong)", lw=1.8, color="#d1495b")
ax[0].plot([h["step"] for h in h_right], [h["true_loss"] for h in h_right],
           label="token weighted (right)", lw=1.8, color="#2a9d8f")
ax[0].set_xlabel("step"); ax[0].set_ylabel("token-weighted loss")
ax[0].set_title("Both measured the same honest way"); ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot([h["step"] for h in h_wrong],
           [w["true_loss"] - r["true_loss"] for w, r in zip(h_wrong, h_right)],
           lw=1.8, color="#e07a5f")
ax[1].axhline(0, ls=":", c="gray")
ax[1].set_xlabel("step"); ax[1].set_ylabel("wrong - right")
ax[1].set_title("The gap"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("accumulation_gap.png", dpi=130)
print("saved accumulation_gap.png")


# %% [markdown]
# ---
# # 4. Log the grad norm every step, and find where it moved first
#
# Grad norm is a leading indicator. It reacts to a change in the loss *surface*
# in the step where that change happens; the loss value only reflects it after
# the optimiser has taken a step in response.

# %%
H = h_right
gn = [h["grad_norm"] for h in H]
ls = [h["true_loss"] for h in H]

print(f"{'step':<7}{'grad_norm':<14}{'d grad_norm %':<16}{'loss':<12}{'d loss':<12}")
print("-" * 62)
for k in range(1, min(12, len(H))):
    dg = 100 * (gn[k] - gn[k - 1]) / max(gn[k - 1], 1e-12)
    dl = ls[k] - ls[k - 1]
    print(f"{k:<7}{gn[k]:<14.4f}{dg:<16.1f}{ls[k]:<12.4f}{dl:<12.5f}")

# Step-to-step loss is noisy, so picking a single step off raw values just finds
# noise. Smooth the loss first, then require the grad-norm move to be large in
# absolute terms AND the smoothed loss to be genuinely flat at that step and
# genuinely moving afterwards.
def smooth(x, w=5):
    return [sum(x[max(0, i - w + 1):i + 1]) / len(x[max(0, i - w + 1):i + 1])
            for i in range(len(x))]


ls_s, gn_s = smooth(ls), smooth(gn)

# Aggregate evidence first: at which lag does grad norm best predict the loss?
dgn = [gn_s[i] - gn_s[i - 1] for i in range(1, len(gn_s))]
dls = [ls_s[i] - ls_s[i - 1] for i in range(1, len(ls_s))]


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) ** .5
    vb = sum((x - mb) ** 2 for x in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / max(va * vb, 1e-12)


print("\ncross-correlation of d(grad norm) with d(loss) at several lags:")
lags = {}
for lag in range(0, 6):
    c = corr(dgn[:len(dgn) - lag], dls[lag:])
    lags[lag] = c
    print(f"  loss lagging grad norm by {lag} step(s): r = {c:+.3f}")
BEST_LAG = max(lags, key=lambda k: abs(lags[k]))
print(f"strongest coupling at lag {BEST_LAG}"
      + (" - the grad norm leads" if BEST_LAG > 0 else " - simultaneous"))

W_ = 5
cands = []
for k in range(W_, len(H) - W_ - 1):
    dg = abs(gn_s[k] - gn_s[k - 1]) / max(gn_s[k - 1], 1e-12)
    dl_now = abs(ls_s[k] - ls_s[k - 1])
    dl_after = abs(ls_s[k + W_] - ls_s[k])
    if dg < 0.05 or dl_after < 1e-4:
        continue
    cands.append((dl_after / (dl_now + 1e-6), k, dg, dl_now, dl_after))
cands.sort(reverse=True)
SC, KSTEP, DG, DLNOW, DLAFT = cands[0]

print(f"\nstep {KSTEP}: smoothed grad norm moved {100 * DG:.1f}% while the smoothed "
      f"loss moved only {DLNOW:.6f};")
print(f"over the next {W_} steps the loss then moved {DLAFT:.6f} "
      f"({DLAFT / max(DLNOW, 1e-12):.0f}x larger).")
print("\ncontext:")
print(f"{'step':<7}{'grad_norm(sm)':<14}{'loss(sm)':<12}")
print("-" * 34)
for k in range(max(0, KSTEP - 2), min(len(H), KSTEP + W_ + 2)):
    mark = "  <- grad norm moves here" if k == KSTEP else ""
    print(f"{k:<7}{gn_s[k]:<14.4f}{ls_s[k]:<12.4f}{mark}")

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(ls_s, color="#2a9d8f", lw=1.6, label="loss")
ax1.set_xlabel("step"); ax1.set_ylabel("loss", color="#2a9d8f")
ax2 = ax1.twinx()
ax2.plot(gn_s, color="#7c5cff", lw=1.2, alpha=.85, label="grad norm")
ax2.set_ylabel("grad norm", color="#7c5cff")
ax1.axvline(KSTEP, ls=":", c="#d1495b")
ax1.set_title(f"Grad norm leads the loss (marked step {KSTEP})")
plt.tight_layout(); plt.savefig("gradnorm_vs_loss.png", dpi=130)
print("saved gradnorm_vs_loss.png")


# %% [markdown]
# ---
# # 5. MFU, computed honestly
#
# `MFU = achieved model FLOPs per second / the hardware's peak FLOPs per second`.
#
# Two numbers to be careful about, because both are easy to fudge:
#
# - **Model FLOPs.** The PaLM/nanoGPT convention:
#   `flops_per_token = 6N + 12 * L * H * Q * T`, where `6N` is forward plus
#   backward through the parameters and the second term is attention's score and
#   value matmuls. This counts *useful* FLOPs, not FLOPs actually issued.
# - **Peak.** Vendor peak is a marketing number. Here the device is benchmarked
#   with a large GEMM, so the denominator is what this machine can really do.

# %%
def measure_peak_flops(sizes=(512, 1024, 2048, 3072), reps=3):
    """Best sustained GEMM throughput over several shapes.

    A single matrix size is not a peak: too small and kernel launch and cache
    effects dominate, too large and it goes memory bound. Sweep, take the best.
    An MFU above 100% is the signature of getting this wrong.
    """
    best, table = 0.0, []
    for n in sizes:
        a = torch.randn(n, n, device=DEV)
        b = torch.randn(n, n, device=DEV)
        for _ in range(2):
            a @ b
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ b
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        f = 2 * (n ** 3) * reps / (time.perf_counter() - t0)
        table.append((n, f))
        best = max(best, f)
        del a, b
    for n, f in table:
        print(f"  GEMM {n}x{n}: {f / 1e12:.4f} TFLOP/s")
    return best


print("benchmarking sustained GEMM throughput:")
PEAK = measure_peak_flops()
print(f"measured peak (best of sweep): {PEAK / 1e12:.4f} TFLOP/s")

cfg = model.cfg
Q = cfg.d_model // cfg.n_head
T_ = SEQ
flops_per_token = 6 * N_PARAMS + 12 * cfg.n_layer * cfg.n_head * Q * T_
BS = 8

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)


def one_step():
    b = DATA[torch.randint(0, DATA.shape[0], (BS,), device=DEV)]
    lg = model(b)
    l = F.cross_entropy(lg[:, :-1].reshape(-1, V), b[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()


for _ in range(3):
    one_step()
if DEV.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
ITERS = 20
for _ in range(ITERS):
    one_step()
if DEV.type == "cuda":
    torch.cuda.synchronize()
step_time = (time.perf_counter() - t0) / ITERS

tokens_per_step = BS * T_
achieved = flops_per_token * tokens_per_step / step_time
MFU = achieved / PEAK

print(f"\nparameters N            {N_PARAMS:,}")
print(f"flops per token         {flops_per_token:,}  (6N + 12*L*H*Q*T)")
print(f"tokens per step         {tokens_per_step}")
print(f"step time               {step_time * 1e3:.1f} ms")
print(f"achieved                {achieved / 1e12:.4f} TFLOP/s")
print(f"measured peak           {PEAK / 1e12:.3f} TFLOP/s")
print(f"\nMFU vs measured peak = {100 * MFU:.2f}%")

# Honesty about the denominator. Published MFU numbers are quoted against a
# datacenter accelerator's vendor peak. A 2-thread CPU GEMM peak is a far lower
# bar, so the same model scores much higher against it. Both are reported.
REFERENCE = {"A100 bf16": 312e12, "T4 fp16": 65e12, "L4 bf16": 121e12}
if DEV.type == "cuda":
    print(f"distance to 40%: {40 - 100 * MFU:.2f} percentage points")
    MFU_REPORT = MFU
else:
    print("\nCAVEAT: this is CPU. The denominator is a 2-thread CPU GEMM, which is a")
    print("very low bar, so the ratio flatters the model. It is NOT comparable to the")
    print("MFU numbers people publish, which use accelerator vendor peak. For scale,")
    print("the same achieved throughput against real accelerators would be:")
    for k, v in REFERENCE.items():
        print(f"    vs {k:<12} {100 * achieved / v:.4f}%")
    MFU_REPORT = achieved / REFERENCE["A100 bf16"]
    print(f"\nThe meaningful number comes from running this on a GPU. Against an A100")
    print(f"the distance to 40% would be {40 - 100 * MFU_REPORT:.2f} percentage points.")


# %% [markdown]
# ### What is costing the distance to 40%
#
# Measured rather than guessed, so the diagnosis is not a story:

# %%
with torch.no_grad():
    b = DATA[:BS]
if DEV.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(ITERS):
    lg = model(b)
if DEV.type == "cuda":
    torch.cuda.synchronize()
fwd_t = (time.perf_counter() - t0) / ITERS

t0 = time.perf_counter()
for _ in range(ITERS):
    lg = model(b)
    l = F.cross_entropy(lg[:, :-1].reshape(-1, V), b[:, 1:].reshape(-1))
    model.zero_grad(set_to_none=True)
    l.backward()
if DEV.type == "cuda":
    torch.cuda.synchronize()
fb_t = (time.perf_counter() - t0) / ITERS

t0 = time.perf_counter()
for _ in range(ITERS):
    opt.step()
if DEV.type == "cuda":
    torch.cuda.synchronize()
opt_t = (time.perf_counter() - t0) / ITERS

head_flops = 2 * tokens_per_step * cfg.d_model * V * 3      # fwd + bwd on the vocab proj
head_share = head_flops / (flops_per_token * tokens_per_step)

print(f"forward only          {fwd_t * 1e3:7.1f} ms")
print(f"forward + backward    {fb_t * 1e3:7.1f} ms")
print(f"optimiser step        {opt_t * 1e3:7.1f} ms   ({100 * opt_t / step_time:.0f}% of the step)")
print(f"vocab projection is   {100 * head_share:.0f}% of counted model FLOPs "
      f"(V={V:,} against d_model={cfg.d_model})")
print(f"batch is {tokens_per_step} tokens, GEMMs are about "
      f"{tokens_per_step}x{cfg.d_model}x{cfg.d_model} - far too small to saturate")


# %% [markdown]
# ---
# # 6. The number 0.1 in three formats
#
# 0.1 is not representable in binary at all: it is the repeating fraction
# `0.0001100110011...`, so every format below stores something slightly else.

# %%
def bits_of(x, dtype, nbits, e_bits, m_bits, int_dtype):
    t = torch.tensor([x], dtype=torch.float32).to(dtype)
    raw = t.view(int_dtype).item() & ((1 << nbits) - 1)
    b = format(raw, f"0{nbits}b")
    s, e, m = b[0], b[1:1 + e_bits], b[1 + e_bits:]
    exact = float(t.to(torch.float64).item())
    bias = (1 << (e_bits - 1)) - 1
    unbiased = int(e, 2) - bias
    return b, s, e, m, exact, unbiased


print("0.1 in binary is 0.0001100110011001100... repeating forever.\n")

f32b = bits_of(0.1, torch.float32, 32, 8, 23, torch.int32)
bf16b = bits_of(0.1, torch.bfloat16, 16, 8, 7, torch.int16)
fp8b = bits_of(0.1, torch.float8_e4m3fn, 8, 4, 3, torch.int8)

for name, (b, s, e, m, exact, ub), layout in [
        ("fp32", f32b, "1 sign | 8 exponent | 23 mantissa"),
        ("bf16", bf16b, "1 sign | 8 exponent |  7 mantissa"),
        ("fp8 E4M3", fp8b, "1 sign | 4 exponent |  3 mantissa")]:
    print(f"{name}   ({layout})")
    print(f"  bits      {s} {e} {m}")
    print(f"  exponent  {e} = {int(e, 2)} biased -> 2^{ub}")
    print(f"  mantissa  1.{m}")
    print(f"  value     {exact:.20f}")
    print(f"  error     {abs(exact - 0.1):.3e}   relative {abs(exact - 0.1) / 0.1:.3e}")
    print()

print("By hand, fp8 E4M3: 0.1 = 1.6 x 2^-4. Exponent -4 + bias 7 = 3 = 0011.")
print("The mantissa has 3 bits, so 1.6 must land on a multiple of 1/8:")
print("  1.500 (100) -> 0.09375    error 0.00625")
print("  1.625 (101) -> 0.1015625  error 0.0015625   <- nearer, so this is chosen")
print("giving 0 0011 101, which is what the hardware produced above.")

# Resolution near 0.1: the gap to the next representable number. This is what
# decides whether a small weight update survives being stored in that format.
print("Spacing between representable neighbours at 0.1, and the smallest update")
print("that still changes the stored value:\n")
for name, dt, mbits in [("fp32", torch.float32, 23), ("bf16", torch.bfloat16, 7),
                        ("fp8 E4M3", torch.float8_e4m3fn, 3)]:
    stored = torch.tensor([0.1], dtype=torch.float32).to(dt).to(torch.float64).item()
    ulp = 2.0 ** (-4 - mbits)          # 0.1 lies in the binade [2^-4, 2^-3)
    print(f"  {name:<10} stores {stored:.12f}   ulp {ulp:.3e}   "
          f"updates below {ulp / 2:.1e} vanish")
print("\nThat last column is the one that matters for training: a bf16 weight near")
print("0.1 cannot absorb an update smaller than about 2e-4, and in fp8 anything")
print("below 3e-3 is lost entirely.")


# %% [markdown]
# ### Which one would I train in
#
# **bf16 for the compute, fp32 for the master weights and the optimiser state.**
#
# The reason is the exponent field, not the mantissa. bf16 keeps all 8 exponent
# bits of fp32, so it covers the same dynamic range, roughly 1e-38 to 3e38.
# Gradients that would silently flush to zero in fp16 survive in bf16, which is
# why bf16 training needs no loss scaling and fp16 training does. It pays for
# that with 7 mantissa bits, about 2-3 decimal digits, and that is affordable
# because the thing being computed is a gradient estimate that is already noisy
# from minibatch sampling.
#
# What is *not* affordable is accumulating in it. Optimiser state and the master
# weights stay fp32: a bf16 weight update of size 1e-4 against a weight of size
# 1e-1 is below the 2^-8 relative resolution of bf16 and would round to nothing,
# so the model would simply stop learning while the loss curve looked stable.
#
# **fp8 E4M3 for forward matmuls only, and only with per-tensor scaling.** With 3
# mantissa bits it stores 0.1 as 0.1015625, a 1.6% relative error, and it has 4
# exponent bits so the representable range is small enough that tensors must be
# rescaled into it. That is fine for the inputs of a GEMM whose output is
# accumulated in fp32 or bf16; it is not fine for anything that is summed over
# many terms, and not for weights that are updated.
#
# So: fp8 where the numbers are consumed immediately, bf16 where they flow, fp32
# where they accumulate.


# %% [markdown]
# ---
# ## Summary

# %%
SUMMARY = {
    "1. shapes": f"ids {tuple(ids.shape)} -> logits {tuple(logits.shape)} -> loss (), "
                 f"{len(rows)} tensors annotated",
    "2. gradient check": f"analytic {analytic:.10f} vs numeric {GRAD_NUM:.10f} at h={GRAD_H:.0e}, "
                         f"{GRAD_DEC:.1f} matching decimals (rel err {GRAD_REL:.2e})",
    "2b. float32 check": f"rel err {rel32:.2e} - same gradient, worse measurement",
    "3. accumulation": f"single step: avg-of-avgs {avg_of_avgs:.4f} vs token-weighted "
                       f"{token_weighted:.4f}; after {len(h_right)} steps gap {GAP:+.4f}",
    "4. grad norm leads": f"lag {BEST_LAG} (r={lags[BEST_LAG]:+.3f}); step {KSTEP}: grad norm "
                          f"moved {100 * DG:.1f}%, loss {DLNOW:.6f} then {DLAFT:.6f} over {W_}",
    "5. MFU": f"{100 * MFU:.2f}% vs measured peak {PEAK / 1e12:.3f} TFLOP/s; "
              f"{100 * MFU_REPORT:.4f}% vs A100 reference (achieved {achieved / 1e12:.4f} TFLOP/s)",
    "6. 0.1": f"fp32 {f32b[4]:.10f} | bf16 {bf16b[4]:.10f} | fp8e4m3 {fp8b[4]:.10f}",
}
print(f"{'item':<22}value")
print("-" * 110)
for k, v in SUMMARY.items():
    print(f"{k:<22}{v}")

with open("results.json", "w") as fh:
    json.dump({
        "device": str(DEV), "params": N_PARAMS, "summary": SUMMARY,
        "grad_check": {"analytic": analytic, "numeric": GRAD_NUM, "h": GRAD_H,
                       "rel_err": GRAD_REL, "decimals": GRAD_DEC, "f32_rel_err": rel32},
        "accumulation": {"wrong": h_wrong, "right": h_right,
                         "single_step": {"avg_of_avgs": avg_of_avgs,
                                         "token_weighted": token_weighted}},
        "mfu": {"mfu": MFU, "mfu_reference": MFU_REPORT, "achieved_flops": achieved, "peak_flops": PEAK,
                "flops_per_token": flops_per_token, "step_time_s": step_time,
                "fwd_ms": fwd_t * 1e3, "fwdbwd_ms": fb_t * 1e3, "opt_ms": opt_t * 1e3,
                "head_share": head_share},
        "float_formats": {"fp32": f32b[:5], "bf16": bf16b[:5], "fp8_e4m3": fp8b[:5]},
    }, fh, indent=2, default=str)
print("\nwrote results.json")
