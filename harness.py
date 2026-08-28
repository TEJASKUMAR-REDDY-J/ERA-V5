# %% [markdown]
# # The loss harness
#
# One notebook, one loss harness, and the few lines between the model output and
# the scalar - which is where training bugs live, and where they do not raise
# exceptions.
#
# **Tejaskumar Reddy J** - ERA V5
#
# Runs top to bottom on CPU or GPU. Every number in the write-up is printed below.

# %%
import os
os.environ["USE_TF"] = "0"                    # stop transformers probing TensorFlow
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import math, json, threading
import torch, torch.nn as nn, torch.nn.functional as F
import psutil
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device {DEV}")
if DEV.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

TOK = AutoTokenizer.from_pretrained("gpt2")
V = len(TOK)                                  # 50257
PAD_ID = TOK.eos_token_id                     # gpt2 has no pad token; we mask it anyway
IGNORE = -100                                 # cross_entropy's "skip this position"
print(f"vocab V = {V} | pad id = {PAD_ID}")


# %% [markdown]
# ## Config and a tiny model
#
# Small enough for CPU, big enough that the logits tensor dominates memory - which
# is what makes item 7 measurable. `output_head` is kept separate from the body on
# purpose: every question in this assignment lives between hidden and the scalar.

# %%
class Cfg:
    d_model, n_layer, n_head, block = 256, 4, 4, 512
CFG = Cfg()
print({k: v for k, v in vars(Cfg).items() if not k.startswith("_")})


class Block(nn.Module):
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
    """Returns HIDDEN states. The output head is applied separately."""

    def __init__(s, vocab, cfg):
        super().__init__()
        s.emb = nn.Embedding(vocab, cfg.d_model)
        s.pos = nn.Embedding(cfg.block, cfg.d_model)
        s.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_head) for _ in range(cfg.n_layer)])
        s.norm = nn.LayerNorm(cfg.d_model)
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
        causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        for b in s.blocks:
            x = b(x, causal)
        return s.norm(x)


def new_head():
    h = nn.Linear(CFG.d_model, V, bias=False).to(DEV)
    nn.init.normal_(h.weight, std=0.02)
    return h


model = TinyLM(V, CFG).to(DEV)
head = new_head()
print(f"body params  {sum(p.numel() for p in model.parameters()):,}")
print(f"head params  {sum(p.numel() for p in head.parameters()):,}")


# %% [markdown]
# ## Real text, and a short trainer
#
# Items 2 and 4 cannot be shown on an untrained model: at initialisation every
# position scores about `ln(V)` and nothing is distinguishable from noise. Both
# need a model that has learned something.
#
# wikitext-2 rather than a toy string, because a toy string gets memorised, and a
# memorised corpus collapses every conditional to zero - which hides exactly the
# effects we are trying to see.

# %%
from datasets import load_dataset

_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
_docs = [t.strip() for t in _ds["text"] if len(t.strip()) > 200]
print(f"wikitext-2: {len(_docs):,} documents")

SEQ = 64


def make_batches(docs, n_seq=512, seq=SEQ):
    ids = []
    for d in docs:
        ids.extend(TOK(d)["input_ids"])
        if len(ids) > n_seq * seq + seq:
            break
    return torch.tensor(ids[: n_seq * seq], dtype=torch.long).view(-1, seq).to(DEV)


DATA = make_batches(_docs)
print(f"training block {tuple(DATA.shape)} = {DATA.numel():,} tokens")


def train_briefly(loss_fn, params, steps=150, bs=8, lr=3e-4, tag=""):
    """steps optimiser steps on DATA. loss_fn(batch) -> scalar. Returns history."""
    opt = torch.optim.AdamW(params, lr=lr)
    hist = []
    for st in range(steps):
        idx = torch.randint(0, DATA.shape[0], (bs,), device=DEV)
        l = loss_fn(DATA[idx])
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(l.item())
        if tag and (st % 50 == 0 or st == steps - 1):
            print(f"  {tag} step {st:>4}  loss {l.item():.4f}")
    return hist


# %% [markdown]
# ---
# # Part 1
#
# ## 1. Every tensor shape, and what each dimension means

# %%
DOC_A = ("The cat sat on the mat. It was a warm afternoon and the cat had no "
         "intention of moving anywhere at all.")
DOC_B = ("Cross entropy is the average negative log probability the model assigns "
         "to the token that actually came next.")

tokens = TOK(DOC_A, return_tensors="pt")["input_ids"].to(DEV)
tokens = tokens.repeat(2, 1)                  # B=2 so the batch dim is visible
B, T = tokens.shape

hidden = model(tokens)
logits = head(hidden)

SHAPES = [
    ("tokens", tokens.shape, "B=batch, T=time. Integer token ids, one row per sequence."),
    ("hidden", hidden.shape, "B, T, D. One D-dim vector per token position."),
    ("logits", logits.shape, "B, T, V. One score per vocabulary entry, per position."),
    ("logits[:, :-1]", logits[:, :-1].shape, "drop the LAST position: it predicts a token we do not have."),
    ("tokens[:, 1:]", tokens[:, 1:].shape, "drop the FIRST token: nothing predicts it."),
    ("flat logits", logits[:, :-1].reshape(-1, V).shape, "(B*(T-1), V). cross_entropy wants 2-D."),
    ("flat targets", tokens[:, 1:].reshape(-1).shape, "(B*(T-1),). one gold id per prediction."),
]
print(f"{'tensor':<16}{'shape':<22}what each dimension is")
print("-" * 100)
for n, sh, why in SHAPES:
    print(f"{n:<16}{str(tuple(sh)):<22}{why}")

loss_correct = F.cross_entropy(logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1))
print(f"\nbaseline loss = {loss_correct.item():.4f}   over {tokens[:, 1:].numel()} predictions")


# %% [markdown]
# ## 2. Verify the shift by printing STRINGS, not ids
#
# A shift in the wrong direction still trains and still produces a pretty curve,
# because predicting the token you were just given is easy, learnable and useless.

# %%
def show_pairs(inp_ids, tgt_ids, n=12, title=""):
    print(f"\n{title}")
    print(f"{'pos':<5}{'input (model sees)':<28}{'target (must predict)':<28}verdict")
    print("-" * 92)
    for i in range(min(n, inp_ids.numel())):
        a = repr(TOK.decode([inp_ids[i].item()]))
        b = repr(TOK.decode([tgt_ids[i].item()]))
        ok = "next token" if i + 1 < inp_ids.numel() and tgt_ids[i] == inp_ids[i + 1] else ""
        print(f"{i:<5}{a:<28}{b:<28}{ok}")


row = 0
show_pairs(tokens[row, :-1], tokens[row, 1:],
           title="CORRECT: logits[:, :-1] vs tokens[:, 1:]")
print("\nFirst line: the model sees 'The' and must predict ' cat'. Correct.")

show_pairs(tokens[row, 1:], tokens[row, :-1],
           title="WRONG: logits[:, 1:] vs tokens[:, :-1]  (off by one, backwards)")

bad_loss = F.cross_entropy(logits[:, 1:].reshape(-1, V), tokens[:, :-1].reshape(-1))
print(f"\ncorrect-shift loss = {loss_correct.item():.4f}")
print(f"wrong-shift  loss  = {bad_loss.item():.4f}")
print("At initialisation both are about ln(V). Nothing here reveals the bug.")


# %% [markdown]
# The two are indistinguishable before training, which is exactly why this bug
# survives review. So train both and watch.
#
# The wrong shift asks the model to predict the token it was **just handed**.
# Under a causal mask that token is already in the residual stream, so the task is
# a copy - and copying is easy. The curve looks wonderful.

# %%
SHIFT_STEPS = 150
m_ok, h_ok = TinyLM(V, CFG).to(DEV), new_head()
m_bad, h_bad = TinyLM(V, CFG).to(DEV), new_head()


def loss_ok(b):        # predict t+1 from <=t  -- correct
    lg = h_ok(m_ok(b))
    return F.cross_entropy(lg[:, :-1].reshape(-1, V), b[:, 1:].reshape(-1))


def loss_bad(b):       # predict t-1 from <=t  -- off by one, backwards
    lg = h_bad(m_bad(b))
    return F.cross_entropy(lg[:, 1:].reshape(-1, V), b[:, :-1].reshape(-1))


print("training the CORRECT shift:")
hist_ok = train_briefly(loss_ok, list(m_ok.parameters()) + list(h_ok.parameters()),
                        steps=SHIFT_STEPS, tag="ok ")
print("training the WRONG shift:")
hist_bad = train_briefly(loss_bad, list(m_bad.parameters()) + list(h_bad.parameters()),
                         steps=SHIFT_STEPS, tag="bad")

c_final, w_final = hist_ok[-1], hist_bad[-1]
print(f"\nafter {SHIFT_STEPS} steps on real text:")
print(f"  correct shift (predict t+1) : {c_final:.4f}")
print(f"  wrong   shift (predict t-1) : {w_final:.4f}")
lower = "lower" if w_final < c_final else "higher"
print(f"  the WRONG one is {lower} by {abs(c_final - w_final):.4f}")
print("\nThis is the trap. The buggy harness produces the better-looking number,")
print("because copying the previous token is trivial while predicting the next one")
print("is the actual task. A loss curve cannot tell you which you trained.")
print("Only printing the strings can.")


# %% [markdown]
# ## 3. Mask padding, and confirm the contributing-token count changes
#
# `ignore_index` skips a position entirely: it contributes to neither the
# numerator nor the denominator of the mean.

# %%
short = TOK("The cat sat on the mat.", return_tensors="pt")["input_ids"][0]
long_ = TOK(DOC_A, return_tensors="pt")["input_ids"][0][:24]
L = max(short.numel(), long_.numel())

pad_batch = torch.full((2, L), PAD_ID, dtype=torch.long)
pad_batch[0, :short.numel()] = short
pad_batch[1, :long_.numel()] = long_
attn_mask = torch.zeros(2, L, dtype=torch.bool)
attn_mask[0, :short.numel()] = True
attn_mask[1, :long_.numel()] = True
pad_batch, attn_mask = pad_batch.to(DEV), attn_mask.to(DEV)

print(f"padded batch {tuple(pad_batch.shape)}  "
      f"(row 0 has {short.numel()} real tokens, row 1 has {long_.numel()})")
print("row 0 tail decoded:", repr(TOK.decode(pad_batch[0, -6:])))

with torch.no_grad():
    lg = h_ok(m_ok(pad_batch))
lg_flat = lg[:, :-1].reshape(-1, V)
tgt_all = pad_batch[:, 1:].reshape(-1)

loss_unmasked = F.cross_entropy(lg_flat, tgt_all)
n_unmasked = tgt_all.numel()

valid = attn_mask[:, 1:].reshape(-1)
tgt_masked = tgt_all.clone()
tgt_masked[~valid] = IGNORE
loss_masked = F.cross_entropy(lg_flat, tgt_masked, ignore_index=IGNORE)
n_masked = int(valid.sum())

print(f"\nunmasked : loss {loss_unmasked.item():.4f}  over {n_unmasked} positions")
print(f"masked   : loss {loss_masked.item():.4f}  over {n_masked} positions")
print(f"padding positions removed from the mean: {n_unmasked - n_masked}")
print("\nThe unmasked number is not comparable to the masked one: predicting a run")
print("of identical pad tokens is trivial, so those positions move the average")
print("without teaching the model anything.")


# %% [markdown]
# ## 4. Pack two documents, and mask the boundary
#
# Packing wastes no space but creates one genuinely impossible prediction: the
# last token of doc A must predict the first token of doc B, and nothing in A
# implies it.

# %%
a = TOK(DOC_A, return_tensors="pt")["input_ids"][0][:20]
b = TOK(DOC_B, return_tensors="pt")["input_ids"][0][:20]
packed = torch.cat([a, b])[None].to(DEV)
doc_id = torch.cat([torch.zeros_like(a), torch.ones_like(b)])[None].to(DEV)
n_a = a.numel()

print(f"packed {tuple(packed.shape)} = doc A ({n_a} tokens) + doc B ({b.numel()} tokens)")
print(f"boundary prediction sits at position {n_a - 1} -> {n_a}")
print(f"  input : {TOK.decode([packed[0, n_a - 1].item()])!r}   (last token of A)")
print(f"  target: {TOK.decode([packed[0, n_a].item()])!r}   (first token of B)")

# Measured on the correctly-trained model: an untrained one scores every position
# at ln(V), so the effect does not exist yet.
with torch.no_grad():
    lp = h_ok(m_ok(packed))[:, :-1].reshape(-1, V)
tp = packed[:, 1:].reshape(-1)

loss_with_boundary = F.cross_entropy(lp, tp)

same_doc = (doc_id[:, :-1] == doc_id[:, 1:]).reshape(-1)
tp_masked = tp.clone()
tp_masked[~same_doc] = IGNORE
loss_no_boundary = F.cross_entropy(lp, tp_masked, ignore_index=IGNORE)

per_tok = F.cross_entropy(lp, tp, reduction="none")
boundary_loss = per_tok[n_a - 1].item()
elsewhere = per_tok[same_doc].mean().item()

print(f"\nwith boundary   : {loss_with_boundary.item():.4f}  over {tp.numel()} predictions")
print(f"boundary masked : {loss_no_boundary.item():.4f}  over {int(same_doc.sum())} predictions")
print(f"difference      : {loss_with_boundary.item() - loss_no_boundary.item():+.4f}")
print(f"\nboundary position alone : {boundary_loss:.4f}")
print(f"mean elsewhere          : {elsewhere:.4f}")
print(f"ratio                   : {boundary_loss / max(elsewhere, 1e-9):.2f}x")
print("\nWhy: nothing in document A implies the first token of document B, so that")
print("prediction is unlearnable and its loss never falls. Leaving it in the mean")
print("adds a penalty the model cannot optimise away, and worse, its gradient")
print("pushes the model to guess document openings from unrelated context.")


# %% [markdown]
# ## 5. Perplexity, and the untrained-model sanity check
#
# An untrained model has no reason to prefer any token, so it spreads mass roughly
# uniformly over V. Uniform over V gives loss `ln(V)` and perplexity `V`.
#
# **If perplexity is not near V, stop and find the bug.** Cheapest correctness
# check in the harness.

# %%
fresh, fresh_head = TinyLM(V, CFG).to(DEV), new_head()
with torch.no_grad():
    lf = fresh_head(fresh(tokens))
    loss_fresh = F.cross_entropy(lf[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1))
ppl_fresh = math.exp(loss_fresh.item())

print(f"untrained loss        {loss_fresh.item():.4f}")
print(f"untrained perplexity  {ppl_fresh:,.1f}")
print(f"vocabulary size V     {V:,}")
print(f"ln(V)                 {math.log(V):.4f}")
print(f"ratio ppl / V         {ppl_fresh / V:.3f}   (want approximately 1.0)")
ok_ppl = 0.5 < ppl_fresh / V < 2.0
print(f"\nSANITY CHECK: {'PASS' if ok_ppl else 'FAIL - find the bug before continuing'}")


# %% [markdown]
# ## 6. Tied vs untied output head
#
# Tying means the output head reuses the input embedding matrix instead of
# learning its own.

# %%
emb_params = V * CFG.d_model
body_params = sum(p.numel() for p in model.parameters())
untied_total = body_params + emb_params        # body (incl. embedding) + separate head
tied_total = body_params                       # head IS the embedding

print(f"embedding / head matrix   V x D = {V:,} x {CFG.d_model} = {emb_params:,}")
print(f"body (incl. embedding)    {body_params:,}")
print()
print(f"UNTIED total parameters   {untied_total:,}")
print(f"TIED   total parameters   {tied_total:,}")
print(f"saved by tying            {untied_total - tied_total:,}  "
      f"({100 * (untied_total - tied_total) / untied_total:.1f}% of the untied model)")

tied_head_out = F.linear(hidden, model.emb.weight)
print(f"\ntied head output {tuple(tied_head_out.shape)} == untied {tuple(logits.shape)}")


# %% [markdown]
# ## 7. Peak memory: ordinary cross-entropy vs a chunked version
#
# The `[B, T, V]` logits tensor is usually the largest thing in a step. Chunking
# never materialises all of it: project a slice, take its loss, accumulate, free.
#
# One honest caveat, which is the part you have to get right by reading: **under
# autograd, naive chunking saves nothing**, because every chunk's logits stay
# alive for backward. It saves memory under `no_grad`, and to get the saving
# during training you must recompute the logits in backward. Both are measured.

# %%
def peak_bytes(fn):
    """Peak allocation during fn(). Exact on CUDA; RSS-sampled on CPU."""
    if DEV.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out = fn(); torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated()
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak, stop = [base], [False]

    def watch():
        while not stop[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)

    t = threading.Thread(target=watch); t.start()
    out = fn()
    stop[0] = True; t.join()
    return out, max(0, peak[0] - base)


def plain_ce(hid, w, tgt):
    return F.cross_entropy(F.linear(hid, w).reshape(-1, V), tgt.reshape(-1))


def chunked_ce(hid, w, tgt, chunk=256):
    """Written out rather than imported: flatten to rows, project one slice at a
    time, accumulate a SUM (not a mean of means), divide once at the end."""
    h = hid.reshape(-1, hid.shape[-1])
    t = tgt.reshape(-1)
    total, n = h.new_zeros(()), 0
    for i in range(0, h.shape[0], chunk):
        hs, ts = h[i:i + chunk], t[i:i + chunk]
        total = total + F.cross_entropy(F.linear(hs, w), ts, reduction="sum")
        n += ts.numel()
    return total / n


MB, TB_ = 4, 256
mem_ids = torch.randint(0, V, (MB, TB_), device=DEV)
with torch.no_grad():
    mem_hidden = model(mem_ids)
mem_tgt = torch.randint(0, V, (MB, TB_), device=DEV)

CHUNK = 256
rows = MB * TB_

# Analytic peak: the logits tensor is what dominates, and its size is exact
# arithmetic. Plain materialises all rows at once; chunked holds one slice.
an_plain = rows * V * 4
an_chunk = CHUNK * V * 4
an_ratio = an_plain / an_chunk

print(f"ANALYTIC (exact arithmetic, the quantity that actually moves)")
print(f"  plain   : {rows} rows x {V:,} x 4B = {an_plain / 2**20:,.1f} MiB")
print(f"  chunked : {CHUNK} rows x {V:,} x 4B = {an_chunk / 2**20:,.1f} MiB")
print(f"  ratio   : {an_ratio:.2f}x  (= rows / chunk = {rows} / {CHUNK})\n")

with torch.no_grad():
    l1, m1 = peak_bytes(lambda: plain_ce(mem_hidden, head.weight, mem_tgt))
    l2, m2 = peak_bytes(lambda: chunked_ce(mem_hidden, head.weight, mem_tgt, chunk=CHUNK))

print(f"MEASURED ({'CUDA allocator, exact' if DEV.type == 'cuda' else 'CPU RSS sampling'})")
print(f"  plain   cross-entropy : loss {l1.item():.4f}   peak {m1 / 2**20:,.1f} MiB")
print(f"  chunked cross-entropy : loss {l2.item():.4f}   peak {m2 / 2**20:,.1f} MiB")
print(f"  agreement             : {abs(l1.item() - l2.item()):.2e}  (same maths)")

# On CPU the measured chunked figure is often ~0: the allocator has already
# reserved the pages from the plain run, so RSS never grows again. That is a
# property of the measurement, not a 1000x saving. Do not report it as one.
measured_usable = DEV.type == "cuda" or m2 > 4 * 2**20
if measured_usable:
    ratio = m1 / max(m2, 1)
    print(f"  RATIO plain/chunked   : {ratio:.2f}x")
else:
    ratio = an_ratio
    print(f"  RATIO plain/chunked   : not measurable on CPU "
          f"(chunked read {m2 / 2**20:.1f} MiB)")
    print("\n  Why: the allocator already reserved these pages during the plain run,")
    print("  so RSS never grows again and the chunked peak reads as zero. That is an")
    print("  artefact of RSS sampling, not a real saving. The analytic ratio above is")
    print("  the honest number on CPU; run this on a GPU for an exact measurement.")

REPORTED_RATIO = ratio
RATIO_SOURCE = "measured" if measured_usable else "analytic"


# %% [markdown]
# ### The training case, where naive chunking does not help

# %%
from torch.utils.checkpoint import checkpoint


def chunked_ce_ckpt(hid, w, tgt, chunk=256):
    h = hid.reshape(-1, hid.shape[-1])
    t = tgt.reshape(-1)
    total, n = h.new_zeros(()), 0
    for i in range(0, h.shape[0], chunk):
        hs, ts = h[i:i + chunk], t[i:i + chunk]
        f = lambda x, y: F.cross_entropy(F.linear(x, w), y, reduction="sum")
        total = total + checkpoint(f, hs, ts, use_reentrant=False)
        n += ts.numel()
    return total / n


gh = mem_hidden.clone().requires_grad_(True)
_, mg1 = peak_bytes(lambda: plain_ce(gh, head.weight, mem_tgt).backward())
gh.grad = None
_, mg2 = peak_bytes(lambda: chunked_ce(gh, head.weight, mem_tgt).backward())
gh.grad = None
_, mg3 = peak_bytes(lambda: chunked_ce_ckpt(gh, head.weight, mem_tgt).backward())

print(f"with autograd, plain          peak {mg1 / 2**20:,.1f} MiB")
print(f"with autograd, naive chunked  peak {mg2 / 2**20:,.1f} MiB")
print(f"with autograd, checkpointed   peak {mg3 / 2**20:,.1f} MiB  <- recomputes in backward")


# %% [markdown]
# ---
# # Part 2: a second head predicting t+2
#
# Head 1 predicts the next token; head 2 predicts the one after, from the same
# hidden state. The alignment is the whole exercise again:
#
# - head 1: `hidden[:, :-1]` vs `tokens[:, 1:]`
# - head 2: `hidden[:, :-2]` vs `tokens[:, 2:]`

# %%
class TwoHead(nn.Module):
    def __init__(s, vocab, cfg):
        super().__init__()
        s.body = TinyLM(vocab, cfg)
        s.h1 = nn.Linear(cfg.d_model, vocab, bias=False)   # t+1
        s.h2 = nn.Linear(cfg.d_model, vocab, bias=False)   # t+2
        for h in (s.h1, s.h2):
            nn.init.normal_(h.weight, std=0.02)

    def losses(s, ids):
        hid = s.body(ids)
        l1 = F.cross_entropy(s.h1(hid[:, :-1]).reshape(-1, V), ids[:, 1:].reshape(-1))
        l2 = F.cross_entropy(s.h2(hid[:, :-2]).reshape(-1, V), ids[:, 2:].reshape(-1))
        return l1, l2


two = TwoHead(V, CFG).to(DEV)

r = tokens[0]
print(f"{'pos':<5}{'input':<22}{'head1 target (t+1)':<24}head2 target (t+2)")
print("-" * 82)
for i in range(8):
    print(f"{i:<5}{TOK.decode([r[i].item()])!r:<22}"
          f"{TOK.decode([r[i+1].item()])!r:<24}{TOK.decode([r[i+2].item()])!r}")

with torch.no_grad():
    a1, a2 = two.losses(tokens)
print(f"\nat init: head1 {a1.item():.4f} | head2 {a2.item():.4f} | sum {(a1 + a2).item():.4f}")
print(f"both near ln(V) = {math.log(V):.4f}, as they should be before training")


# %%
TWO_STEPS = 300
hist2 = []


def loss_two(batch):
    l1, l2 = two.losses(batch)
    hist2.append((l1.item(), l2.item()))
    return l1 + l2


_ = train_briefly(loss_two, list(two.parameters()), steps=TWO_STEPS, tag="two")

f1 = sum(h[0] for h in hist2[-10:]) / 10        # average last 10 to damp noise
f2 = sum(h[1] for h in hist2[-10:]) / 10
s1 = sum(h[0] for h in hist2[:10]) / 10
s2 = sum(h[1] for h in hist2[:10]) / 10
print(f"\nFINAL (mean of last 10 steps)")
print(f"  head1 (t+1) {f1:.4f}")
print(f"  head2 (t+2) {f2:.4f}")
print(f"  sum         {f1 + f2:.4f}")
print(f"\ngap head2 - head1 : start {s2 - s1:+.4f}  ->  end {f2 - f1:+.4f}")


# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
plt.plot([h[0] for h in hist2], label="head 1  (t+1)", lw=1.6)
plt.plot([h[1] for h in hist2], label="head 2  (t+2)", lw=1.6)
plt.axhline(math.log(V), ls=":", c="gray", label=f"ln(V) = {math.log(V):.2f}")
plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.legend(); plt.grid(alpha=.3)
plt.title("Two heads, one body: t+1 vs t+2")
plt.tight_layout(); plt.savefig("two_heads.png", dpi=130)
print("saved two_heads.png")


# %% [markdown]
# ### What happens to head 2's loss, and why
#
# Both heads start at `ln(V)`. Head 2 then falls more slowly and settles above
# head 1, and the gap **opens during training** rather than being present at the
# start.
#
# The reason is not that head 2 is worse at its job - it is that its job is
# harder, and irreducibly so. Head 1 models `p(x_t+1 | x_<=t)`. Head 2 models
# `p(x_t+2 | x_<=t)`, which marginalises over the token in between:
#
# ```
# p(x_t+2 | x_<=t) = SUM over x_t+1 of  p(x_t+2 | x_<=t, x_t+1) * p(x_t+1 | x_<=t)
# ```
#
# That marginalisation destroys information. The single most useful clue for
# predicting a token is the token immediately before it, and head 2 is denied it.
# So `H(x_t+2 | x_<=t) >= H(x_t+1 | x_<=t)`: the gap is a property of the data,
# not a training failure, and a perfectly trained model still shows it.
#
# The size of the gap depends on the corpus. On text the model can memorise, both
# conditionals collapse toward zero and the gap nearly vanishes - which is why
# this notebook trains on wikitext rather than a repeated toy string.
#
# This is the idea behind multi-token prediction (Gloeckle et al., 2024; used in
# DeepSeek-V3): the extra heads are a *training signal*, not a better predictor.
# They force the hidden state to carry information beyond the immediate next
# token, and are usually discarded at inference or reused for speculative
# decoding.


# %% [markdown]
# ---
# ## Summary: the seven numbers, plus Part 2

# %%
SUMMARY = {
    "1. shapes": f"tokens {tuple(tokens.shape)} -> hidden {tuple(hidden.shape)} -> logits {tuple(logits.shape)}",
    "2. shift check": f"correct {c_final:.4f} vs wrong-direction {w_final:.4f} "
                      f"after {SHIFT_STEPS} steps (wrong looks better)",
    "3. padding": f"{n_unmasked} -> {n_masked} contributing positions "
                  f"({n_unmasked - n_masked} removed)",
    "4. packing": f"boundary in {loss_with_boundary.item():.4f} / masked {loss_no_boundary.item():.4f} "
                  f"(delta {loss_with_boundary.item() - loss_no_boundary.item():+.4f}, "
                  f"boundary is {boundary_loss / max(elsewhere, 1e-9):.2f}x mean)",
    "5. perplexity": f"{ppl_fresh:,.1f} vs V={V:,} (ratio {ppl_fresh / V:.3f})",
    "6. tied vs untied": f"{tied_total:,} tied vs {untied_total:,} untied "
                         f"({untied_total - tied_total:,} saved)",
    "7. peak memory": f"plain {an_plain / 2**20:,.1f} MiB vs chunked {an_chunk / 2**20:,.1f} MiB "
                      f"analytic (ratio {an_ratio:.2f}x); measured plain {m1 / 2**20:,.1f} MiB "
                      f"[{RATIO_SOURCE} ratio {REPORTED_RATIO:.2f}x]",
    "P2. head1 (t+1)": f"{f1:.4f}",
    "P2. head2 (t+2)": f"{f2:.4f}",
    "P2. sum": f"{f1 + f2:.4f}",
}
print(f"{'item':<20}value")
print("-" * 104)
for k, v in SUMMARY.items():
    print(f"{k:<20}{v}")

with open("results.json", "w") as fh:
    json.dump({"device": str(DEV), "vocab": V, "summary": SUMMARY,
               "two_head_history": hist2,
               "shift_history": {"correct": hist_ok, "wrong": hist_bad}}, fh, indent=2)
print("\nwrote results.json")
