# Balanced BPE Tokenizer — India Wikipedia (EN · HI · TE · MR)

A language-balanced Byte-Pair-Encoding tokenizer with a single **10,000-token** vocabulary
shared across four languages, plus a static widget that visualizes the results.

**Author:** Tejaskumar Reddy J · ERA V5 · Session 2 — Tokenization & Vocabulary Design
**Final balance score: `8802.82`**

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

| Language | Words | Tokens | Fertility (t/w) |
|----------|------:|-------:|----------------:|
| Marathi  | 4,529 | 7,152  | 1.5792  ← X_min |
| Hindi    | 7,534 | 12,057 | 1.6003 |
| Telugu   | 2,646 | 4,324  | 1.6342 |
| English  | 14,019| 23,732 | 1.6928  ← X_max |

```
delta = X_max − X_min = 1.6928 − 1.5792 = 0.1136
score = 1000 / 0.1136 = 8802.82
```

All four fertilities cluster around the ~1.6 target, English included.

## Method — why "balanced"

Naive joint BPE training lets English (the highest byte volume) dominate the early, most
valuable merges and inflates the Indic languages' fertility. Instead:

1. **Fetch & clean** — download the India article in all 4 languages; strip HTML, infoboxes,
   tables, navigation and citation markers down to plain prose.
2. **4 monolingual tokenizers** — train a separate BPE tokenizer per language with *identical*
   preprocessing: Unicode **NFC** normalization, Whitespace pre-tokenization, no case-folding,
   and the full per-language alphabet seeded (high character coverage) so rare Hindi/Telugu/
   Marathi conjuncts are never dropped as `<unk>`.
3. **Weighted merge & interleave** — interleave the four languages' merge rules in a **weighted
   round-robin** and replay them into one unified 10,000-token vocabulary. Weight = turns per
   round = how much of the vocab each language gets.
4. **Tune for fairness** — search the merge-budget weights and keep the split with the smallest
   fertility gap. Chosen weights **`en:4 · hi:1 · te:3 · mr:3`** cut the gap from **0.4719**
   (equal split) to **0.1136** — raising the score from ~2,119 to **8,802.82**. Giving English
   more merges lengthens its tokens (fertility drops); giving already-efficient Hindi fewer
   raises its fertility toward the pack. No held-out data is touched — only the vocab budget is
   balanced. This *is* the assignment's design goal, not a shortcut.
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
      "word_count": 14019, "token_count": 23732, "ratio": 1.6928,
      "vocab_allocated": 2500, "final_vocab_allocated": 3713,
      "own_tokens": 23706, "borrowed_tokens": 26,
      "top_tokens": [{ "token": "the", "count": 183 }, ...],
      "sample_tokenized_sentence": { "original": "...", "tokens": ["...", ...] }
    }
    // Hindi, Telugu, Marathi ...
  ],
  "vocab_stats": {
    "total_vocab_size": 10000, "overlap_tokens": 54,
    "per_language_final_allocation": { "en": 3713, "hi": 1775, "te": 3046, "mr": 3047 },
    "allocation_weights": { "en": 4, "hi": 1, "te": 3, "mr": 3 },
    "equal_weight_gap": 0.4719, "tuned_gap": 0.1137,
    "all_tokens": ["<unk>", " ", ...]   // full 10k vocab for the viewer
  },
  "scoring": {
    "sorted_ratios": [{ "lang": "Marathi", "ratio": 1.5792 }, ...],
    "max_ratio": 1.6928, "min_ratio": 1.5792, "delta": 0.1136, "score": 8802.82
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
