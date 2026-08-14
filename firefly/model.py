"""A small decoder-only transformer, shared by the arithmetic and LM experiments.

Deliberately plain. Every arm of every experiment uses the same body, so any
difference in the results is attributable to the embedding under test and not to
an architectural change smuggled in alongside it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class TinyTransformer(nn.Module):
    """ids -> logits, with an optional per-position continuous feature vector.

    `feats` is how a frozen FIREFLY code enters the model: it is projected once
    and added to the token embedding, exactly as the Kronecker paper projects its
    codec through a single learned matrix.
    """

    def __init__(self, vocab: int, d_model: int = 128, n_layer: int = 4,
                 n_head: int = 4, max_len: int = 256, feat_dim: int = 0,
                 n_positions: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(n_positions or max_len, d_model)
        self.feat = nn.Linear(feat_dim, d_model, bias=False) if feat_dim else None
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor, feats: torch.Tensor | None = None,
                pos_ids: torch.Tensor | None = None,
                return_hidden: bool = False) -> torch.Tensor:
        b, t = ids.shape
        if pos_ids is None:
            pos_ids = torch.arange(t, device=ids.device).expand(b, t)
        x = self.tok(ids) + self.pos(pos_ids)
        if self.feat is not None and feats is not None:
            x = x + self.feat(feats)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=ids.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        return x if return_hidden else self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ResidueReadout(nn.Module):
    """Same body, but the answer is read out as residues instead of digits.

    E2 found that exact structure in the embedding buys nothing while the
    readout is still digit tokens -- the bottleneck moves to the output, where
    carries and digit counts reappear. A residue head keeps the representation
    consistent at both ends: residue(a+b) is a fixed function of residue(a) and
    residue(b), and carries no dependence on how many digits anything has.
    The integer is reassembled by CRT, which has no parameters.
    """

    def __init__(self, body: TinyTransformer, primes, d_model: int):
        super().__init__()
        self.body = body
        self.primes = tuple(primes)
        self.heads = nn.ModuleList([nn.Linear(d_model, p) for p in self.primes])

    def forward(self, ids, feats=None, pos_ids=None, read_at: int = -1):
        h = self.body(ids, feats, pos_ids, return_hidden=True)[:, read_at]
        return [head(h) for head in self.heads]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FrozenCodeEmbedding(nn.Module):
    """A frozen codebook (FIREFLY or Kronecker) plus one learned projection.

    The codes are computed once, stored as a non-trainable buffer, and the only
    parameters are the D -> d_model matrix. This is the arrangement whose
    parameter count the whole approach is arguing about.
    """

    def __init__(self, codes, d_model: int):
        super().__init__()
        self.register_buffer("codes", torch.as_tensor(codes, dtype=torch.float32))
        self.proj = nn.Linear(self.codes.shape[1], d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(self.codes.shape[1]))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes[ids])

    def n_trainable(self) -> int:
        return self.proj.weight.numel()


class LMWithFrozenEmbedding(nn.Module):
    """Language model whose input side is a frozen codebook.

    Used for E3, where the arms are (learned table | Kronecker | FIREFLY) and the
    body is byte-for-byte identical across them.
    """

    def __init__(self, vocab: int, codes=None, d_model: int = 192, n_layer: int = 4,
                 n_head: int = 4, max_len: int = 256):
        super().__init__()
        self.embed = FrozenCodeEmbedding(codes, d_model) if codes is not None else None
        self.tok = nn.Embedding(vocab, d_model) if codes is None else None
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(TinyTransformer._init)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        x = (self.embed(ids) if self.embed is not None else self.tok(ids))
        x = x + self.pos(torch.arange(t, device=ids.device))[None]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=ids.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm(x))

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self(ids[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1))

    def n_params(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)
