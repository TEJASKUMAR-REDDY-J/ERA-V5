# Wiki-Faithful BPE Tokenizer — India Wikipedia (EN · HI · TE · MR)

One shared **10,000-token** BPE tokenizer for the India Wikipedia page in **English, Hindi,
Telugu and Marathi**, trained on a *faithful Markdown* corpus, plus a static widget that
visualizes and lets a grader download/inspect the tokenizer.

**Author:** Tejaskumar Reddy J · ERA V5 · Session 2 — Tokenization & Vocabulary Design
**Repo:** https://github.com/TEJASKUMAR-REDDY-J/ERA-V5
**Score: `4481.90`** · Hindi penalty **1.0×** · `decode(encode(text))` is exactly faithful (0 `[UNK]`, no normalizer)

> This follows the ERA V5 Assignment 2 **reference solution**: faithful-Markdown corpus,
> Metaspace BPE, and fertility measured against *faithful units* (not clipped word prose).

## Results

Fertility = `token_count / faithful_units`, where a **faithful unit** is one contiguous
Unicode letter/mark/number run **or** one visible punctuation/symbol character.

| Language | Tokens | Faithful units | Fertility |
|----------|-------:|---------------:|----------:|
| Hindi    | 51,270 | 88,359  | 0.580246  ← X_min |
| English  | 114,772| 186,367 | 0.615839 |
| Telugu   | 24,751 | 36,292  | 0.681996 |
| Marathi  | 23,913 | 29,766  | 0.803366  ← X_max |

```
spread = X_max − X_min = 0.803366 − 0.580246 = 0.223120
raw score = 1000 / 0.223120 = 4481.90
hindi_penalty = exp(max(0, 0.580246/1.2 − 1)) = 1.0   →  adjusted score = 4481.90
```

All four ratios are under the 1.2 threshold. (Marathi's larger corpus makes it the most
efficient / highest-fertility language, which widens the spread vs. the tiny Maithili page.)

## Method

1. **Faithful Markdown corpus** — `build_wiki_faithful_markdown.py` fetches each page via the
   Wikipedia REST HTML API and converts it to Markdown with `markdownify`. Links, URLs, tables,
   references, image links, navboxes and categories are **kept**; only `script`/`style`/`meta`
   are stripped. This is a real document, not a clipped word list.
2. **One shared BPE** — `train_tokenizer.py`: HuggingFace `BPE`, vocab **10,000**,
   `min_frequency=1`, **Metaspace** pre-tokenizer + decoder (`▁` space marker), and **no
   normalizer**. Metaspace preserves punctuation, brackets, URL characters, apostrophes and
   number separators; it beats ByteLevel because ByteLevel wastes tokens on UTF-8 bytes for
   Indic scripts. The NFKC normalizer from the reference is **dropped on purpose** — NFKC is
   lossy (rewrites ligatures, superscripts, full-width forms) and would make
   `decode(encode(text))` equal `NFKC(text)` instead of the raw text, failing the faithful gate.
3. **Language weighting** — each corpus file is **duplicated** during training by its weight
   `{en:3, hi:4, te:4, mr:2}`, giving the smaller Indic pages fair influence on the shared merges.
4. **Faithful-unit fertility & score** — `evaluate_tokenizer.py` counts faithful units with
   `regex` `[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]`, computes `tokens / units` per language, and
   scores `1000 / (max − min)`. A Hindi penalty `exp(max(0, hindi/1.2 − 1))` only applies if
   Hindi fertility exceeds 1.2 (here it is 1.0).

**Faithfulness requirement:** `decode(encode(text))` preserves every non-whitespace character.
Example: `India's population is 1,428,627,663.` round-trips exactly.

## Files

```
tokenizer-assignment/
├── build_wiki_faithful_markdown.py   fetch + convert the 4 Wikipedia pages
├── train_tokenizer.py                train the shared 10k BPE
├── evaluate_tokenizer.py             faithful-unit fertility + score
├── make_results.py                   build results.json (+ tokens_*.json) for the widget
├── tokenizer.json                    trained tokenizer (inspectable / downloadable)
├── metrics.json                      saved metrics
├── SOLUTION.md                       reference write-up
├── corpus/
│   ├── {en,hi,te,mr}.faithful.md    faithful Markdown snapshots
│   ├── {en,hi,te,mr}.faithful.txt   tokenizer input (same content, plain text)
│   ├── {en,hi,te,mr}.meta.json      corpus metadata
│   └── {en,hi,te,mr}.raw.html       raw REST HTML
├── results/  results.json + tokens_*.json
└── frontend/ static widget (HTML/CSS/JS, Netlify-ready)
```

## Run it

```bash
pip install tokenizers regex requests beautifulsoup4 lxml markdownify

python build_wiki_faithful_markdown.py   # -> corpus/*.faithful.{md,txt}
python train_tokenizer.py                # -> tokenizer.json, metrics.json
python evaluate_tokenizer.py             # -> prints ratios + score
python make_results.py                   # -> results/results.json (+ copies into frontend/)
```

The included corpus snapshots are the exact ones used for the metrics above; counts may drift
if the live Wikipedia pages change.

## Widget

Dark, Mac-terminal-styled single page with tabs — **Overview** (score, sorted fertility chart,
round-trip faithfulness check), **How it works** (method + code), **Languages** (per-language
faithful units, tokens, fertility, weight, sample tokenization, full token stream), and
**Vocabulary** (all 10,000 tokens, searchable).

## Deploy to Netlify

`make_results.py` copies `results.json` + `tokens_*.json` into `frontend/`, so it deploys alone.

1. https://app.netlify.com/drop
2. Drag **only the `frontend` folder** onto the page.
3. Open the root URL (`https://<your-site>.netlify.app/`).
