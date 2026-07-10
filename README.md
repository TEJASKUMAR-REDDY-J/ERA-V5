# Balanced BPE Tokenizer — India Wikipedia (EN · HI · TE · MR)

A language-balanced Byte-Pair-Encoding tokenizer with a single **10,000-token** vocabulary
shared across four languages, plus a static widget that visualizes the results.

**Author:** Tejaskumar Reddy J · ERA V5 · Session 2 — Tokenization & Vocabulary Design
**Repo:** https://github.com/TEJASKUMAR-REDDY-J/ERA-V5
**Final balance score: `6807.35`** · English fertility **1.24**

---

## The assignment

Pick the **India** Wikipedia article in English, Hindi, Telugu and one more language (Marathi),
and design a BPE tokenizer such that:

- there are **10,000 tokens total** (one shared vocab for all languages);
- for each language, **fertility = tokens ÷ words** (English target ≈ 1.6 or less), call them `X1…X4`;
- sort the ratios; the score is `1000 / (X_max − X_min)`.

A high score means the fertilities are close together — i.e. the tokenizer is **fair** across
languages rather than efficient for one and wasteful for the others.

## Results

Corpora are capped to **3,000 words per language** (equal-sized, comparable — the assignment's
"say 5000 words" framing) so English can reach a low fertility inside the shared 10k vocab.

| Language | Words | Tokens | Fertility (t/w) |
|----------|------:|-------:|----------------:|
| English  | 2,880 | 3,578  | 1.2424  ← X_min |
| Marathi  | 2,904 | 3,936  | 1.3554 |
| Hindi    | 2,930 | 4,031  | 1.3758 |
| Telugu   | 2,646 | 3,676  | 1.3893  ← X_max |

```
delta = X_max − X_min = 1.3893 − 1.2424 = 0.1469
score = 1000 / 0.1469 = 6807.35
```

English lands at **1.24** (well under the ~1.6 guideline); the other three cluster at 1.36–1.39.

## Method — why "balanced"

Naive joint BPE training lets English (the highest byte volume) dominate the early, most
valuable merges and inflates the Indic languages' fertility. Instead:

1. **Fetch & clean** — download the India article in all 4 languages; strip HTML, infoboxes,
   tables, navigation and citation markers down to plain prose.
2. **4 monolingual tokenizers** — train a separate BPE tokenizer per language with *identical*
   preprocessing: Unicode **NFC** normalization, **WhitespaceSplit** pre-tokenization, no
   case-folding, and the full per-language alphabet seeded (high character coverage) so rare
   Hindi/Telugu/Marathi conjuncts are never dropped as `<unk>`. WhitespaceSplit splits on
   whitespace *only*, so punctuation stays attached to its word (`India,` `(the`) and frequent
   word+punctuation pairs fold into single tokens instead of every comma/period becoming its own
   token — this alone drops English fertility by ~0.13 (punctuation was 18.6% of its tokens).
3. **Weighted merge & interleave** — interleave the four languages' merge rules in a **weighted
   round-robin** and replay them into one unified 10,000-token vocabulary. Weight = turns per
   round = how much of the vocab each language gets.
4. **Tune (English ~1.2 target)** — search the merge-budget weights; keep English fertility
   ≤1.25 and then minimise the gap so the other three cluster as close to English as possible.
   Chosen weights **`en:5 · hi:3 · te:7 · mr:6`** give English **1.24** with the rest at
   1.36–1.39, score **6,807** (equal-split gap was 0.4758). No held-out data is touched —
   only the vocab budget is balanced.
5. **Evaluate & score** — tokenize each full article with the one merged tokenizer, measure
   fertility per language, sort, and score `1000 / (X_max − X_min)`.

## Project structure

```
tokenizer-assignment/
├── data/
│   ├── raw/                 raw Wikipedia HTML per language
│   └── clean/               cleaned plain-text article per language (en/hi/te/mr .txt)
├── scripts/
│   ├── fetch_data.py        fetch + clean the 4 Wikipedia pages
│   ├── train_tokenizer.py   train 4 monolingual BPEs, weighted-merge to 10k, tune allocation
│   ├── evaluate.py          fertility ratios + score  ->  results.json (+ copies into frontend/)
│   └── utils.py             shared config, NFC normalize, consistent word split
├── tokenizer/
│   ├── merged_vocab.json    final 10,000-token tokenizer (HuggingFace format, inspectable)
│   ├── mono_*.json          the 4 monolingual tokenizers (used for own-vs-borrowed attribution)
│   └── vocab_stats.json     overlap, per-language allocation, tuning weights, gap before/after
├── results/
│   ├── results.json         structured output the frontend reads (schema below)
│   └── tokens_*.json        full ordered token stream per language
├── frontend/                static site (HTML/CSS/JS, no build step) — Netlify-ready
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── results.json, tokens_*.json   auto-copied by evaluate.py so this folder deploys alone
└── README.md
```

## Run it

Requires Python 3.10+ with `tokenizers`, `requests`, `beautifulsoup4`.

```bash
pip install tokenizers requests beautifulsoup4

python scripts/fetch_data.py        # -> data/raw/*.html, data/clean/*.txt
python scripts/train_tokenizer.py   # -> tokenizer/merged_vocab.json (+ mono, stats)
python scripts/evaluate.py          # -> results/results.json (+ copies into frontend/)
```

Preview the widget locally (any static server):

```bash
python -m http.server 8000 --directory frontend
# open http://localhost:8000/
```

## The widget

A dark, Mac-terminal-styled single-page site with top-level tabs:

- **Overview** — the big balance score, the sorted fertility bar chart with X_max/X_min, and caveats.
- **How it works** — the 5-step method, the pipeline flow, and the project layout / core code shown
  in mac-terminal windows.
- **Languages** — a tab per language with word/token counts, fertility, own-vs-borrowed tokens, a
  tokenized sample sentence, top-30 tokens, and the **entire tokenized article in order** (searchable).
- **Vocabulary** — all **10,000** tokens of the merged vocab, searchable across every script.

## Deploy to Netlify

The `frontend/` folder is self-contained (`evaluate.py` copies `results.json` and `tokens_*.json`
into it), so its `index.html` sits at the site root.

1. Go to **https://app.netlify.com/drop**
2. Drag **only the `frontend` folder** onto the page.
3. Open the resulting root URL (`https://<your-site>.netlify.app/`) — done.

> Drag `frontend` itself, not its parent — otherwise `index.html` won't be at the root and you get a 404.
> Re-running `evaluate.py` refreshes the copies inside `frontend/`, so a re-drag always ships current data.

## results.json schema (abridged)

```jsonc
{
  "languages": [
    {
      "name": "English", "wiki_url": "...",
      "word_count": 14019, "token_count": 21858, "ratio": 1.5592,
      "vocab_allocated": 2500, "final_vocab_allocated": 3887,
      "own_tokens": 21833, "borrowed_tokens": 25,
      "top_tokens": [{ "token": "the", "count": 183 }, ...],
      "sample_tokenized_sentence": { "original": "...", "tokens": ["...", ...] }
    }
    // Hindi, Telugu, Marathi ...
  ],
  "vocab_stats": {
    "total_vocab_size": 10000, "overlap_tokens": 54,
    "per_language_final_allocation": { "en": 3887, "hi": 1750, "te": 3142, "mr": 2739 },
    "allocation_weights": { "en": 8, "hi": 2, "te": 6, "mr": 5 },
    "equal_weight_gap": 0.4282, "tuned_gap": 0.0538,
    "all_tokens": ["<unk>", " ", ...]   // full 10k vocab for the viewer
  },
  "scoring": {
    "sorted_ratios": [{ "lang": "Telugu", "ratio": 1.5212 }, ...],
    "max_ratio": 1.575, "min_ratio": 1.5212, "delta": 0.0538, "score": 18587.36
  },
  "notes": ["Ratio = fertility = tokens per word ...", ...]
}
```

## Caveats

- Trained and evaluated on the **same corpus** (the India Wikipedia pages) per the assignment
  spec — fertility is optimistic versus a held-out set.
- Word counts use one whitespace + punctuation rule for all four languages; Indic scripts have
  no intra-word spaces, so this is a consistent cross-language proxy rather than a linguistic
  word count.
