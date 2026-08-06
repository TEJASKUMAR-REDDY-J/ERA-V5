"""Static configuration for the V5 training-data execution system.

Everything here is a *contract* inherited from an earlier session:
  S1 loss contract      -> LOSS_POLICY per lane
  S2 tokenizer contract -> TOKENIZER_PATH + frozen hash check
  S3 source contract    -> provenance/license fields required on every manifest
  S4 admission contract -> cleaning hash, dedup/PII/contamination status
  S5 mixture contract   -> LANES, STAGES, PROTECTED_FLOORS, anneal reserve
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "submission_artifacts"
MANIFESTS = ARTIFACTS / "manifests"
LEDGERS = ARTIFACTS / "ledgers"
CHECKPOINTS = ARTIFACTS / "checkpoints"
SHARD_DIR = ARTIFACTS / "shards"
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"

# ---- S2: the tokenizer is frozen. Token ids are meaningless without this hash.
TOKENIZER_ID = "s2-wiki-faithful-bpe-10k"

# ---- S5: capability lanes ---------------------------------------------------
LANES = ["web", "code", "indic", "reasoning", "agentic"]

# Loss policy per lane (S1 contract made operational).
#   "all"       -> every real token is loss-bearing (plain pretraining)
#   "response"  -> prompt is context, response bears loss (SFT shape)
#   "model"     -> only the model's own tokens bear loss; tool observations
#                  are context. Training on observations teaches the model to
#                  invent tool results instead of calling tools.
LOSS_POLICY = {
    "web": "all", "code": "all", "indic": "all",
    "reasoning": "response", "agentic": "model",
}

# Packing policy per lane (S6 section 5).
#   concat_chop         -> join with EOS, cut fixed windows (plain text)
#   structure_preserving-> one sample per sequence, never merge (masks stay clean)
PACKING_POLICY = {
    "web": "concat_chop", "code": "concat_chop", "indic": "concat_chop",
    "reasoning": "structure_preserving", "agentic": "structure_preserving",
}

# ---- S5: curriculum stages, scaled down. Shares sum to 1.0 per stage. -------
# Real plan was 1.85T + 0.15T anneal; here the same shape over a tiny budget so
# the compiler logic (stages, warmup, floors, anneal reserve) is exercised.
STAGES = [
    {
        "stage": "s1-general", "step_start": 0, "step_end": 80,
        "sequence_length": 256, "warmup_steps": 0,
        "mixture": {"web": 0.46, "code": 0.24, "indic": 0.16, "reasoning": 0.10, "agentic": 0.04},
    },
    {
        "stage": "s2-capability", "step_start": 80, "step_end": 135,
        "sequence_length": 256, "warmup_steps": 12,
        "mixture": {"web": 0.30, "code": 0.28, "indic": 0.18, "reasoning": 0.16, "agentic": 0.08},
    },
    {
        "stage": "s4-anneal", "step_start": 135, "step_end": 160,
        "sequence_length": 256, "warmup_steps": 8, "anneal": True,
        "mixture": {"web": 0.10, "code": 0.20, "indic": 0.28, "reasoning": 0.30, "agentic": 0.12},
    },
]
TOTAL_STEPS = STAGES[-1]["step_end"]

# S5 protected floors: the selector may never push a lane below these.
PROTECTED_FLOORS = {"indic": 0.12, "agentic": 0.02, "reasoning": 0.04}

# Shards flagged anneal_reserve are invisible to the sampler until the anneal
# stage. If the selector eats the best data early, the anneal has nothing left.
ANNEAL_RESERVE_FRACTION = 0.20

# ---- training ---------------------------------------------------------------
SEQ_LEN = 256
MICRO_BATCH = 8
GRAD_ACCUM = 2                     # global batch = MICRO_BATCH * GRAD_ACCUM
CHECKPOINT_EVERY = 30
CRASH_AT_STEP = 90                 # run_demo hard-kills the trainer here
REPLAY_RANGE = (40, 55)            # step interval replayed and hash-compared
KEEP_CHECKPOINT_WEIGHTS = 2        # older .pt weights pruned; their .json metadata is kept

# ---- model (deliberately tiny: the data plane is the deliverable) -----------
MODEL = {"n_layer": 2, "n_head": 4, "d_model": 96, "block": SEQ_LEN}
LEARNING_RATE = 3e-3
SEED = 20260706

# ---- OPUS -------------------------------------------------------------------
OPUS_ACCEPT_QUANTILE = 0.45        # reject the weakest fraction of candidates
OPUS_DEFER_MARGIN = 0.10           # near-miss candidates are deferred, not dropped
