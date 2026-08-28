<h1 align="center">The Loss Harness</h1>
<p align="center">Making the few lines between model output and the scalar correct, and observable.</p>
<p align="center">Tejaskumar Reddy J · ERA V5</p>

**Notebook:** [`loss_harness.ipynb`](loss_harness.ipynb) — runs top to bottom, CPU or GPU.

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

Four lines. Every bug below lives inside them, and not one of them raises an
exception.

---

## The seven numbers

Run on CPU, GPT-2 tokenizer (`V = 50,257`), 4-layer / `d_model=256` model,
wikitext-2. Regenerate with `python harness.py`; all values also land in
`results.json`.

| # | What | Result |
|---|---|---|
| 1 | Shapes | `tokens (2,24)` → `hidden (2,24,256)` → `logits (2,24,50257)` |
| 2 | Shift check | correct **6.3961** vs wrong-direction **4.2610** after 150 steps |
| 3 | Padding mask | **46 → 29** contributing positions (17 removed) |
| 4 | Packing boundary | with **7.8616** / masked **7.6936** (Δ **+0.1680**; boundary is **1.83×** the mean position) |
| 5 | Perplexity | **58,685.7** vs `V = 50,257` → ratio **1.168** ✅ |
| 6 | Tied vs untied | **16,156,416** vs **29,022,208** (**12,865,792** saved, 44.3%) |
| 7 | Peak memory | plain **196.3 MiB** vs chunked **49.1 MiB** → **4.00×** |

## Part 2 — the second head

| Head | Predicts | Loss after 300 steps |
|---|---|---|
| head 1 | `t+1` | **5.4421** |
| head 2 | `t+2` | **5.8464** |
| | **sum** | **11.2885** |

Gap `head2 − head1`: **+0.0285 at the start → +0.4043 at the end.**

---

## Item 2 is the one that matters

The assignment's warning is that a target shift in the wrong direction produces a
beautiful loss curve. It does, and the numbers above understate how bad it is,
because **the buggy version looks better**:

```
correct shift (predict t+1) : 10.883  ->  6.3961
wrong   shift (predict t-1) : 10.889  ->  4.2610
```

Both start at `ln(V)`. Both fall smoothly. The wrong one ends **2.13 nats lower**,
so if you were comparing runs on a dashboard you would ship the broken one.

The reason is mechanical. The wrong shift asks the model to predict the token it
was *just handed*. Under a causal mask that token is already in the residual
stream, so the task is a copy, and copying is far easier than prediction. Nothing
in the loss can tell you which task you trained.

The only thing that catches it is printing the pairs as strings:

```
pos  input (model sees)        target (must predict)     verdict
0    'The'                     ' cat'                    next token
1    ' cat'                    ' sat'                    next token
2    ' sat'                    ' on'                      next token
```

versus the same batch shifted the wrong way:

```
pos  input (model sees)        target (must predict)
0    ' cat'                    'The'
1    ' sat'                    ' cat'
2    ' on'                     ' sat'
```

The second table is obviously wrong at a glance and invisible in a wall of ids.

## Item 4: why masking the boundary matters

Packing two documents into one sequence creates exactly one impossible
prediction — the last token of doc A must predict the first token of doc B:

```
input : ' moving'   (last token of A)
target: 'Cross'     (first token of B)
```

Measured on a **trained** model (on an untrained one every position sits at
`ln(V)` and the effect does not exist yet):

- boundary position alone: **7.86** — 1.83× the mean position
- masking it moves the reported loss from 7.8616 to 7.6936

Nothing in A implies the first token of B, so that prediction is unlearnable and
its loss never falls. Leaving it in adds a constant the model cannot optimise
away, and its gradient actively teaches the model to guess document openings from
unrelated context.

## Item 7: the honest version

Naive chunking **does not save memory under autograd**. Every chunk's logits stay
alive for the backward pass, so peak is unchanged. The saving is real under
`no_grad`, and during training only if you recompute the logits in backward:

| | peak |
|---|---|
| plain, with autograd | 196.4 MiB |
| naive chunked, with autograd | 149.9 MiB |
| checkpointed chunked | **0.7 MiB** |

The headline 4.00× is **analytic** (`rows/chunk = 1024/256`), because that is
exact arithmetic. On CPU the measured chunked peak reads 0.0 MiB — the allocator
already reserved those pages during the plain run, so RSS never grows again. That
is a measurement artefact, not a 6516× saving, and the notebook says so rather
than reporting the flattering number. On CUDA the measurement is exact and the
notebook uses it.

## Part 2: what happens to head 2, and why

Both heads start at `ln(V)`. Head 2 falls more slowly and settles above head 1,
and **the gap opens during training** (+0.03 → +0.40) rather than being there at
the start.

Head 2 is not worse at its job; its job is harder, irreducibly. Head 1 models
`p(x_t+1 | x_≤t)`. Head 2 models `p(x_t+2 | x_≤t)`, which marginalises over the
token in between:

```
p(x_t+2 | x_≤t) = Σ over x_t+1 of  p(x_t+2 | x_≤t, x_t+1) · p(x_t+1 | x_≤t)
```

That marginalisation destroys information. The most useful clue for predicting a
token is the token immediately before it, and head 2 is denied it. So
`H(x_t+2 | x_≤t) ≥ H(x_t+1 | x_≤t)` — the gap is a property of the data, not a
training failure, and a perfectly trained model still shows it.

The gap size depends on the corpus. On text small enough to memorise, both
conditionals collapse toward zero and the gap nearly vanishes — which is why this
notebook trains on wikitext-2 rather than a repeated toy string. An earlier
version used a toy string and measured a gap of **−0.0013**, i.e. the wrong sign,
purely because the model had memorised the corpus.

This is the mechanism behind multi-token prediction (Gloeckle et al., 2024; used
in DeepSeek-V3): the extra heads are a *training signal*, not a better predictor.
They force the hidden state to carry information beyond the immediate next token,
and are usually discarded at inference or reused for speculative decoding.

## Running it

```bash
pip install torch transformers datasets psutil matplotlib
python harness.py          # prints everything, writes results.json + two_heads.png
python to_notebook.py      # regenerates loss_harness.ipynb from harness.py
```

`harness.py` is the source of truth: it is what gets run and verified, and the
notebook is generated from it, so the two cannot drift. On Colab, open
`loss_harness.ipynb` and Run All — the first cell installs what is missing.

## Honest limits

- **The model is tiny and trained for a few hundred steps.** The losses are not
  competitive with anything; the point is the harness, not the model.
- **Item 7's CPU measurement is unusable** and reported as such. The GPU path is
  exact.
- **Item 5's ratio is 1.168, not 1.000.** A randomly initialised model is not
  exactly uniform — the embedding and head are drawn from `N(0, 0.02²)`, which
  gives slightly non-uniform logits. Anything in roughly 0.5–2.0× V passes; far
  outside that range means a real bug.
- **The boundary effect is measured on one packed pair**, so the 1.83× is
  illustrative rather than an average over a corpus.
