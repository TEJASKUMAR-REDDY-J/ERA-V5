<h1 align="center">Reversible 20M LLM</h1>
<p align="center">Three runs: baseline, reversible, reversible at maximum batch size.</p>
<p align="center">Tejaskumar Reddy J · ERA V5</p>

> **Status: placeholder.** The plan and the budget are settled; the runs and the
> notebooks land on this same branch. Numbers below marked `TBD` are the ones the
> runs produce.

---

## The assignment

Train a 20M-parameter LLM on 50M tokens, three times:

1. **Baseline** — largest batch size that fits.
2. **Reversible** — same budget, reversible layers, reporting which coupling
   variant was used (additive / Euler / midpoint / Verlet).
3. **Reversible, max batch** — push the batch size as far as the freed activation
   memory allows.

Report final loss, throughput (tokens/s), peak memory, and findings.

## Why reversibility is the whole point

A standard transformer stores every layer's activations for the backward pass, so
activation memory grows linearly with depth. A reversible layer does not: given
the outputs, the inputs are **recomputed exactly** by running the coupling
backwards, so activations are freed on the way forward and reconstructed on the
way back.

The additive coupling, splitting the stream into `x1, x2`:

```
forward                        inverse
y1 = x1 + F(x2)                x1 = y1 - F(x2)
y2 = x2 + G(y1)                x2 = y2 - G(y1)
```

Nothing is saved but the final output. Depth becomes free in memory and costs
roughly one extra forward pass in compute — the same trade as gradient
checkpointing, except exact rather than segment-wise, and with no stored
boundaries at all.

That is why run 3 exists. Reversibility is not a loss-quality technique; its
payoff is the batch size it unlocks, and the honest comparison is run 2 against
run 3, not run 1 against run 2.

## Budget — this costs nothing

| | |
|---|---|
| hardware | free Colab T4 (fallback: local CPU for correctness checks only) |
| training FLOPs | `6 * 20e6 * 50e6 = 6.0e15` per run |
| T4 realistic fp16 throughput | ~8 TFLOP/s sustained |
| estimated wall-clock | **~12–15 min per run, ~45 min for all three** |
| cost | **₹0** — inside the free tier, no paid compute, no API calls |

50M tokens at 20M parameters is roughly 2.5 tokens/parameter. That is well under
Chinchilla, so the model will be undertrained and the absolute loss will not be
impressive. It is the correct budget for this assignment anyway, because the
question is about memory and throughput, not about a good model.

## Plan

- **Data:** a small pre-tokenised corpus streamed from disk, ~50M GPT-2 tokens.
  No download-heavy dataset, no preprocessing that has to run twice.
- **Model:** ~20M non-embedding parameters, tied embeddings, fp16/bf16 autocast.
- **Fixed across all three runs:** tokens seen, learning-rate schedule, sequence
  length, seed. Only batch size and layer type change, and run 3 changes batch
  size alone.
- **Measured:** final loss, tokens/s, `torch.cuda.max_memory_allocated()`, and
  the activation-memory breakdown that explains the difference.
- **Correctness check before any training:** reconstruct the inputs from the
  outputs and assert the reversible layer's gradients match an ordinary backward
  pass. Reversibility that is not exact is just a slow bug.

## Results

| run | layers | batch | final loss | tokens/s | peak memory |
|---|---|---|---|---|---|
| 1. baseline | standard | TBD | TBD | TBD | TBD |
| 2. reversible | TBD variant | same | TBD | TBD | TBD |
| 3. reversible, max batch | TBD variant | TBD | TBD | TBD | TBD |

Coupling variant chosen: **TBD** — with the reason it beat the others.

## Files

| file | what |
|---|---|
| `README.md` | this |
| `reversible.ipynb` | _to come_ — the three runs, end to end on Colab |
| `train.py` | _to come_ — source of truth; the notebook is generated from it |
| `results.json` | _to come_ — every number in the tables above |

## Running it

_to come_
