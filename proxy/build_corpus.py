#!/usr/bin/env python3
"""
Build the proxy corpus: three domain pools (web-EN, Indic, code) tokenized with the
Session 2 wiki-faithful 10k BPE tokenizer, saved as uint16 token arrays.

The Session 4 cleaning normalizer is reused so the proxy trains on cleaned text —
the same discipline the mixture plan claims to stand on.
"""
from __future__ import annotations
import html, json, sys, unicodedata
from pathlib import Path

import numpy as np
import regex
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pools"
OUT.mkdir(exist_ok=True)
TOKENIZER = (ROOT.parent.parent / "Session 2_Tokenization and vocabulary design"
             / "tokenizer-assignment" / "tokenizer.json")

# tokens to collect per pool (uint16 ids). Enough to feed every arm without reuse.
POOL_TOKENS = {"web": 14_000_000, "indic": 14_000_000, "code": 8_000_000}

# ---- Session 4 normalization (noise invisibles out, Brahmic joiners kept) ----
NOISE = set("​﻿�­‪‫‬‭‮⁦⁧⁨⁩")
CTRL = regex.compile(r"[\p{Cc}\p{Cf}]")
KEEP = {"‌", "‍", "\n", "\t"}

def clean(t: str) -> str:
    t = unicodedata.normalize("NFC", html.unescape(t))
    out = []
    for ch in t:
        if ch in NOISE:
            continue
        if ch in KEEP or not CTRL.match(ch):
            out.append(ch)
    t = "".join(out)
    t = regex.sub(r"[ \t]+", " ", t)
    return regex.sub(r"\n{3,}", "\n\n", t).strip()


def stream(repo, **kw):
    return load_dataset(repo, split="train", streaming=True, **kw)


def collect(pool: str, tok: Tokenizer) -> None:
    dst = OUT / f"{pool}.npy"
    if dst.exists():
        print(f"{pool}: exists, skip"); return
    target = POOL_TOKENS[pool]
    ids: list[int] = []
    eos = tok.token_to_id("[UNK]") or 0   # separator; content-agnostic

    def feed(it, key):
        nonlocal ids
        for r in it:
            txt = r.get(key) or ""
            if len(txt) < 200:
                continue
            ids.extend(tok.encode(clean(txt)).ids); ids.append(eos)
            if len(ids) >= target:
                return True
        return False

    if pool == "web":
        feed(stream("HuggingFaceFW/fineweb-edu", name="sample-10BT"), "text")
    elif pool == "indic":
        # half Telugu, half Hindi — the two Indic languages the S2 tokenizer covers
        half = target // 2
        for cfg in ("verified/tel", "verified/hin"):
            sub = []
            for r in stream("ai4bharat/sangraha", data_dir=cfg):
                txt = r.get("text") or ""
                if len(txt) < 200:
                    continue
                sub.extend(tok.encode(clean(txt)).ids); sub.append(eos)
                if len(sub) >= half:
                    break
            ids.extend(sub)
    elif pool == "code":
        # ungated python sources, first that works wins
        for repo, key, kw in (("codeparrot/codeparrot-clean-valid", "content", {}),
                              ("codeparrot/xlcost-text-to-code", "code", {"name": "Python-program-level"}),
                              ("nampdn-ai/tiny-codes", "response", {})):
            try:
                if feed(stream(repo, **kw), key) or ids:
                    print(f"  code source: {repo}")
                    break
            except Exception as e:
                print(f"  code source {repo} failed: {type(e).__name__}")

    arr = np.array(ids[:target], dtype=np.uint16)
    np.save(dst, arr)
    print(f"{pool}: {len(arr):,} tokens -> {dst.name}")


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER))
    print(f"tokenizer vocab={tok.get_vocab_size()}")
    for pool in ("indic", "web", "code"):
        try:
            collect(pool, tok)
        except Exception as e:
            print(f"{pool}: FAILED {type(e).__name__}: {str(e)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
