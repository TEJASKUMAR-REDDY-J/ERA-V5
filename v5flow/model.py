"""Deliberately tiny decoder-only LM. The data plane is the deliverable; this
model exists only to turn batches into a real loss signal.

It takes the packed batch's own masks rather than assuming a dense causal
window: attention is restricted to (causal AND same-document), and position ids
come from the packer instead of arange, so packed documents behave as if each
had been trained alone.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .packing import LOSS_IGNORE


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h, self.dk = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, allow):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, cfg: dict | None = None):
        super().__init__()
        c = cfg or config.MODEL
        d, h, L, blk = c["d_model"], c["n_head"], c["n_layer"], c["block"]
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(blk, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, input_ids, labels, segment_ids, position_ids):
        B, T = input_ids.shape
        x = self.tok(input_ids) + self.pos(position_ids.clamp(max=self.pos.num_embeddings - 1))
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=input_ids.device))
        same = segment_ids[:, :, None] == segment_ids[:, None, :]
        allow = (causal[None, :, :] & same)[:, None, :, :]        # (B,1,T,T)
        for b in self.blocks:
            x = b(x, allow)
        logits = self.head(self.lnf(x))
        per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  labels.reshape(-1), ignore_index=LOSS_IGNORE,
                                  reduction="none").view(B, T)
        mask = labels != LOSS_IGNORE
        loss = per_tok[mask].mean() if mask.any() else per_tok.sum() * 0.0
        return loss, per_tok.detach(), mask
