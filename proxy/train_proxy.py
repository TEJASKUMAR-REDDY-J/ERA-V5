#!/usr/bin/env python3
"""
Proxy mixture ablation. Trains one small decoder-only LM per mixture arm at a FIXED
token budget and reports held-out bits-per-character (BPC) per domain.

Why BPC and not loss/token: token counts differ per language (Session 2 fertility), so
nats/token is not comparable across English / Indic / code. BPC divides by characters,
which is tokenizer-invariant and therefore comparable across lanes.

The arms sweep the Indic share at a fixed code share. The resulting marginal-return
curve is what sets the Indic floor in the mixture plan, instead of a round number.

Run:  python train_proxy.py            (all arms)
      ARMS=indic16 python train_proxy.py   (single arm)
"""
from __future__ import annotations
import json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.manual_seed(1337)
ROOT = Path(__file__).resolve().parent
POOLS = ROOT / "pools"
OUT = ROOT / "proxy_results.json"
TOKENIZER = (ROOT.parent.parent / "Session 2_Tokenization and vocabulary design"
             / "tokenizer-assignment" / "tokenizer.json")

# ---- config (small on purpose: this ranks recipes, it does not predict final scores) ----
BLOCK, BATCH = 256, 24
N_LAYER, N_HEAD, N_EMBD = 4, 4, 192
STEPS = int(os.environ.get("STEPS", 900))
LR, WARMUP = 3e-3, 60
EVAL_TOKENS = 240_000          # held-out tokens per domain
DOMAINS = ["web", "indic", "code"]

# arms: Indic share swept, code held fixed, web absorbs the remainder
CODE_SHARE = 0.25
ARMS = {
    "indic06": 0.06,   # naive web-heavy preset
    "indic16": 0.16,   # session pretrain preset
    "indic24": 0.24,
    "indic34": 0.34,
}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
                                 nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab, bias=False)
        self.head.weight = self.tok.weight          # tied
        self.register_buffer("mask", torch.triu(torch.full((BLOCK, BLOCK), float("-inf")), 1))
        # GPT-style init: without this, tied N(0,1) embeddings make logits ~sqrt(d) and
        # the step-0 loss lands near 127 instead of ln(vocab)=9.2.
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


def load_pools():
    data = {}
    for d in DOMAINS:
        p = POOLS / f"{d}.npy"
        if not p.exists():
            raise SystemExit(f"missing pool {p} — run build_corpus.py first")
        a = np.load(p)
        split = len(a) - EVAL_TOKENS
        data[d] = {"train": a[:split], "eval": a[split:]}
    return data


def batch_from(pool, shares, rng):
    """Sample each sequence's domain by the mixture shares — the mixture is enforced
    in expectation at the sequence level, which is how a real dataloader mixes."""
    xs, ys = [], []
    picks = rng.choice(DOMAINS, size=BATCH, p=[shares[d] for d in DOMAINS])
    for d in picks:
        a = pool[d]["train"]
        i = rng.integers(0, len(a) - BLOCK - 1)
        chunk = a[i:i + BLOCK + 1].astype(np.int64)
        xs.append(chunk[:-1]); ys.append(chunk[1:])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


@torch.no_grad()
def eval_bpc(model, pool, chars_per_domain):
    """Held-out cross-entropy converted to bits per character."""
    model.eval(); out = {}
    for d in DOMAINS:
        a = pool[d]["eval"]
        n = (len(a) - 1) // BLOCK
        tot_nats, tot_tok = 0.0, 0
        for i in range(0, min(n, 60)):
            chunk = a[i * BLOCK:(i + 1) * BLOCK + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1])[None, :]
            y = torch.from_numpy(chunk[1:])[None, :]
            _, loss = model(x, y)
            tot_nats += loss.item() * y.numel(); tot_tok += y.numel()
        nats_per_tok = tot_nats / tot_tok
        # bits per char = (nats/token / ln2) * (tokens/char)
        toks_per_char = 1.0 / chars_per_domain[d]     # chars per token -> inverse
        out[d] = {
            "loss_nats_per_token": round(nats_per_tok, 4),
            "bits_per_char": round(nats_per_tok / math.log(2) * toks_per_char, 4),
            "ppl": round(math.exp(nats_per_tok), 2),
        }
    model.train(); return out


def chars_per_token(tok, pool):
    """Decode the eval slice once to get chars/token per domain (fertility, Session 2)."""
    cpt = {}
    for d in DOMAINS:
        ids = pool[d]["eval"][:60 * BLOCK].astype(int).tolist()
        txt = tok.decode(ids)
        cpt[d] = max(1e-9, len(txt) / max(1, len(ids)))
    return cpt


def run_arm(name, indic_share, pool, cpt, vocab):
    web = 1.0 - indic_share - CODE_SHARE
    shares = {"web": web, "indic": indic_share, "code": CODE_SHARE}
    seed = int(os.environ.get("SEED", 0))    # vary to estimate the noise floor
    rng = np.random.default_rng(seed)
    torch.manual_seed(1337 + seed)
    model = TinyLM(vocab)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    sched = lambda s: (s + 1) / WARMUP if s < WARMUP else 0.5 * (1 + math.cos(
        math.pi * (s - WARMUP) / max(1, STEPS - WARMUP)))
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * sched(step)
        x, y = batch_from(pool, shares, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 150 == 0:
            print(f"  {name} step {step}/{STEPS} loss {loss.item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    res = eval_bpc(model, pool, cpt)
    return {"arm": name, "shares": {k: round(v, 4) for k, v in shares.items()},
            "params": nparam, "steps": STEPS,
            "tokens_seen": STEPS * BATCH * BLOCK,
            "train_s": round(time.time() - t0, 1), "eval": res}


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER))
    vocab = tok.get_vocab_size()
    pool = load_pools()
    cpt = chars_per_token(tok, pool)
    print("chars/token (fertility):", {k: round(v, 3) for k, v in cpt.items()})
    only = os.environ.get("ARMS")
    arms = {k: v for k, v in ARMS.items() if not only or k in only.split(",")}
    prev = json.loads(OUT.read_text()) if OUT.exists() else {"arms": []}
    done = {a["arm"] for a in prev["arms"]}
    seed = int(os.environ.get("SEED", 0))
    for name, share in arms.items():
        tag = name if seed == 0 else f"{name}_s{seed}"
        if tag in done:
            print(f"{tag}: done, skip"); continue
        print(f"== arm {tag} (indic={share:.0%}) ==", flush=True)
        r = run_arm(name, share, pool, cpt, vocab); r["arm"] = tag
        prev["arms"].append(r)
        prev["config"] = {"block": BLOCK, "batch": BATCH, "layers": N_LAYER,
                          "heads": N_HEAD, "d_model": N_EMBD, "steps": STEPS,
                          "code_share": CODE_SHARE, "chars_per_token": {k: round(v, 3) for k, v in cpt.items()},
                          "tokenizer": "Session 2 wiki-faithful 10k BPE"}
        OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print(f"   -> {prev['arms'][-1]['eval']}", flush=True)
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
