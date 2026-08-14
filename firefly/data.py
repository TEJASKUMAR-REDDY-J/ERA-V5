"""Real data for every experiment. Nothing synthetic where real exists.

Sources (all public, all downloaded once and cached):
  text    Salesforce/wikitext, wikitext-2-raw-v1   CC BY-SA 3.0
  images  ylecun/mnist                             CC BY-SA 3.0
  audio   google/speech_commands v0.02             CC BY 4.0

Arithmetic is generated, because "3417 + 882" has no corpus and needs none --
the point there is length generalisation, which requires controlling the digit
count exactly.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CACHE = Path(__file__).resolve().parent.parent / "data_cache"
CACHE.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def wikitext(split: str = "train", max_docs: int | None = None) -> list[str]:
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    txt = [t for t in d["text"] if len(t.strip()) > 32]
    return txt[:max_docs] if max_docs else txt


def train_bpe(vocab_size: int = 8192):
    """A real BPE over real text, so token byte-lengths have a realistic
    distribution rather than one we invented."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    path = CACHE / f"bpe_{vocab_size}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tok.train_from_iterator(wikitext("train"),
                            BpeTrainer(vocab_size=vocab_size,
                                       special_tokens=["[UNK]", "[EOS]"]))
    tok.save(str(path))
    return tok


def token_vocab_bytes(vocab_size: int = 8192) -> list[bytes]:
    """The vocabulary as raw byte strings -- the NCA decoder's training set."""
    from tokenizers.pre_tokenizers import ByteLevel
    tok = train_bpe(vocab_size)
    inv = {v: k for k, v in tok.get_vocab().items()}
    alphabet = {c: i for i, c in enumerate(ByteLevel.alphabet())}
    out = []
    for i in range(tok.get_vocab_size()):
        s = inv.get(i, "")
        try:
            out.append(bytes(alphabet[c] for c in s))
        except KeyError:
            out.append(s.encode("utf-8", "replace"))
    return out


def token_stream(split: str = "train", vocab_size: int = 8192,
                 max_tokens: int | None = None) -> np.ndarray:
    cache = CACHE / f"ids_{split}_{vocab_size}.npy"
    if cache.exists():
        ids = np.load(cache)
    else:
        tok = train_bpe(vocab_size)
        chunks = [np.array(e.ids, dtype=np.int32)
                  for e in tok.encode_batch(wikitext(split))]
        ids = np.concatenate(chunks) if chunks else np.zeros(0, np.int32)
        np.save(cache, ids)
    return ids[:max_tokens] if max_tokens else ids


LANGS = ["en", "hi", "ar", "ru", "zh", "el", "th", "ja", "es", "de"]


def language_id(n: int = 6000, split: str = "train"):
    """Real multilingual text, 10 scripts. A genuinely byte-level task: the
    scripts differ, so the byte field carries the answer."""
    from datasets import load_dataset
    d = load_dataset("papluca/language-identification", split=split)
    want = {l: i for i, l in enumerate(LANGS)}
    xs, ys = [], []
    for t, lab in zip(d["text"], d["labels"]):
        if lab in want and len(t.strip()) > 24:
            xs.append(t.strip()[:256])
            ys.append(want[lab])
            if len(xs) >= n:
                break
    return xs, np.array(ys, dtype=np.int64)


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------
def mnist(n: int = 4000, split: str = "train", side: int = 32):
    """MNIST padded 28 -> 32 so the 16x16 spatial mode budget clears Nyquist."""
    from datasets import load_dataset
    d = load_dataset("ylecun/mnist", split=f"{split}[:{n}]")
    pad = (side - 28) // 2
    imgs = np.stack([np.pad(np.array(r["image"], dtype=np.uint8),
                            pad, mode="constant") for r in d])
    return imgs, np.array(d["label"], dtype=np.int64)


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
def spoken_digits(n: int = 2700, length: int = 4096):
    """Free Spoken Digit Dataset -- 2700 real recordings of digits 0-9 at 8 kHz.

    Decoded with soundfile from the raw bytes rather than through the datasets
    audio feature, which in datasets>=4 requires torchcodec. Installing that
    would drag numpy>=2 back in and break matplotlib and scipy, so we decode the
    wav ourselves and leave the pinned stack alone.
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    d = load_dataset("mteb/free-spoken-digit-dataset", split="train")
    d = d.cast_column("audio", Audio(decode=False))
    xs, ys = [], []
    for row in d.select(range(min(n, len(d)))):
        a, _ = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float64")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if a.size < length:
            a = np.pad(a, (0, length - a.size))
        mid = a.size // 2
        xs.append(a[max(0, mid - length // 2):][:length])
        ys.append(int(row["label"]))
    return xs, np.array(ys, dtype=np.int64)


# --------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------
def arithmetic(n: int, digits_lo: int, digits_hi: int, op: str = "+",
               seed: int = 0):
    """(a, b, result) triples with operand digit counts UNIFORM over [lo, hi].

    Sampling `integers(10**(lo-1), 10**hi)` instead would put 90% of the mass on
    the longest length, so a model trained on "1 to 3 digits" would barely see a
    1-digit operand and would look broken on the short end of its own training
    range. Digit count is drawn first, then the value within that count.
    """
    rng = np.random.default_rng(seed)

    def draw():
        d = rng.integers(digits_lo, digits_hi + 1, size=n)
        lo = np.where(d == 1, 0, 10 ** (d - 1))
        return rng.integers(lo, 10 ** d)

    a, b = draw(), draw()
    r = a + b if op == "+" else a * b
    return a.astype(np.int64), b.astype(np.int64), r.astype(np.int64)


# --------------------------------------------------------------------------
def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
