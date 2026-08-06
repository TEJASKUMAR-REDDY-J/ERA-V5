"""S4 admission contract, executable. The hash of this module's source becomes
the cleaning_pipeline_hash stamped on every shard manifest, so a shard can
always name the exact code that produced it."""
from __future__ import annotations
import html
import unicodedata
from pathlib import Path

import regex

from .hashing import sha256_bytes

# Noise invisibles. NOTE: ZWNJ (200C) and ZWJ (200D) are deliberately NOT here —
# they carry meaning in Brahmic scripts and stripping them corrupts Indic text.
NOISE = set("​﻿�­‪‫‬‭‮"
            "⁦⁧⁨⁩")
CTRL = regex.compile(r"[\p{Cc}\p{Cf}]")
KEEP = {"‌", "‍", "\n", "\t"}
# Ghost markers: conversation/control tokens written into text as ordinary
# characters by upstream sources. Pretraining on them teaches a fake chat
# structure that then collides with the tokenizer's real special tokens at SFT.
# <|endoftext|> is included because the real glaive corpus embeds it literally.
GHOST = regex.compile(r"<\|im_(?:start|end)\|>|<\|(?:user|assistant|system|endoftext)\|>"
                      r"|\[/?INST\]|<<SYS>>|</?s>", regex.IGNORECASE)
PII = [("EMAIL", regex.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
       ("PHONE", regex.compile(r"(?:\+91[\-\s]?)?\b[6-9]\d{9}\b")),
       ("IP", regex.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"))]


def clean(text: str) -> tuple[str, dict]:
    """Return (cleaned_text, counters). Counters feed the shard manifest."""
    c = {"garbage_chars": 0, "indic_joiners_kept": 0, "ghost_markers": 0, "pii_masked": 0}
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch in NOISE:
            c["garbage_chars"] += 1
            continue
        if ch in KEEP:
            if ch in ("‌", "‍"):
                c["indic_joiners_kept"] += 1
            out.append(ch)
            continue
        if CTRL.match(ch):
            c["garbage_chars"] += 1
            continue
        out.append(ch)
    text = "".join(out)
    text, n = GHOST.subn(" ", text)
    c["ghost_markers"] += n
    for label, rx in PII:
        text, n = rx.subn(f"[{label}]", text)
        c["pii_masked"] += n
    text = regex.sub(r"[ \t]+", " ", text)
    text = regex.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), c


def pipeline_hash() -> str:
    """Content hash of this cleaning implementation."""
    return sha256_bytes(Path(__file__).read_bytes())
