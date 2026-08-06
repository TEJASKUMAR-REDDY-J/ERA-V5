"""Documents -> immutable tokenized shards + manifests.

Two shard kinds, because the packing policy differs by data type:

  stream  (web/code/indic)      flat token array; documents joined with EOS and
                                their spans recorded. Safe to concat-and-chop.
  records (reasoning/agentic)   one record per sample, each carrying segment
                                spans tagged context/model, so the packer can
                                place loss only on the model's own tokens.

A shard is sealed once written. Any edit produces a new content hash, hence a
new shard id and a new lineage.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict

import numpy as np

from . import config
from .cleaning import clean, pipeline_hash
from .hashing import canonical_json, hash_obj, hash_tokens, short
from .tokenizer_gate import TokenizerGate

EOS_TOKEN = "[UNK]"          # frozen vocab has no dedicated EOS; reuse a fixed id
DOCS_PER_SHARD = {"web": 30, "code": 30, "indic": 30, "reasoning": 24, "agentic": 20, "eval": 12}


@dataclass
class Manifest:
    shard_id: str
    kind: str
    lane: str
    source_ids: list[str]
    document_ids: list[str]
    tokenizer_id: str
    tokenizer_hash: str
    token_count: int
    language: str
    script: str
    license: str
    provenance_tier: str
    cleaning_pipeline_hash: str
    dedup_status: str
    pii_status: str
    contamination_status: str
    eval_overlap: bool
    never_train: bool
    anneal_reserve: bool
    content_hash: str
    parent_shard_ids: list[str]
    cleaning_counters: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=True)


LANG = {"web": ("en", "Latin"), "code": ("en", "Latin"), "indic": ("te+hi", "Telugu+Devanagari"),
        "reasoning": ("en", "Latin"), "agentic": ("en", "Latin"), "eval": ("en", "Latin")}


def _read_lane(lane: str) -> list[dict]:
    p = config.DATA_RAW / f"{lane}.jsonl"
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _build_stream_shard(gate: TokenizerGate, lane: str, docs: list[dict], idx: int,
                        eos_id: int, reserve: bool) -> tuple[np.ndarray, Manifest, list[dict]]:
    ids: list[int] = []
    spans, counters = [], {}
    for d in docs:
        text, c = clean(d["text"])
        for k, v in c.items():
            counters[k] = counters.get(k, 0) + v
        tokens = gate.encode(text)
        if not tokens:
            continue
        start = len(ids)
        ids.extend(tokens)
        ids.append(eos_id)
        spans.append({"doc_id": d["doc_id"], "start": start, "end": len(ids) - 1,
                      "eos_at": len(ids) - 1})
    arr = np.array(ids, dtype=np.uint16)
    content_hash = hash_tokens(arr)
    m = Manifest(
        shard_id=f"{lane}-stream-{idx:03d}-{short(content_hash)}", kind="stream", lane=lane,
        source_ids=sorted({d["source"] for d in docs}),
        document_ids=[d["doc_id"] for d in docs],
        tokenizer_id=config.TOKENIZER_ID, tokenizer_hash=gate.hash,
        token_count=int(arr.size), language=LANG[lane][0], script=LANG[lane][1],
        license=docs[0]["license"], provenance_tier=docs[0]["provenance_tier"],
        cleaning_pipeline_hash=pipeline_hash(), dedup_status="passed",
        pii_status="masked", contamination_status="clean", eval_overlap=False,
        never_train=False, anneal_reserve=reserve, content_hash=content_hash,
        parent_shard_ids=[], cleaning_counters=counters)
    return arr, m, spans


def _build_record_shard(gate: TokenizerGate, lane: str, docs: list[dict], idx: int,
                        reserve: bool) -> tuple[dict, Manifest, list[dict]]:
    records = []
    counters: dict = {}
    total = 0
    for d in docs:
        seg_ids, segs = [], []
        for s in d["segments"]:
            text, c = clean(s["text"])
            for k, v in c.items():
                counters[k] = counters.get(k, 0) + v
            t = gate.encode(text)
            if not t:
                continue
            segs.append({"role": s["role"], "start": len(seg_ids), "end": len(seg_ids) + len(t)})
            seg_ids.extend(t)
        if not seg_ids:
            continue
        total += len(seg_ids)
        records.append({"doc_id": d["doc_id"], "tokens": seg_ids, "segments": segs,
                        "band": d.get("band")})
    payload = {"records": records}
    content_hash = hash_obj(payload)
    m = Manifest(
        shard_id=f"{lane}-records-{idx:03d}-{short(content_hash)}", kind="records", lane=lane,
        source_ids=sorted({d["source"] for d in docs}),
        document_ids=[d["doc_id"] for d in docs],
        tokenizer_id=config.TOKENIZER_ID, tokenizer_hash=gate.hash,
        token_count=total, language=LANG[lane][0], script=LANG[lane][1],
        license=docs[0]["license"], provenance_tier=docs[0]["provenance_tier"],
        cleaning_pipeline_hash=pipeline_hash(), dedup_status="passed",
        pii_status="masked", contamination_status="clean", eval_overlap=False,
        never_train=False, anneal_reserve=reserve, content_hash=content_hash,
        parent_shard_ids=[], cleaning_counters=counters)
    return payload, m, [{"doc_id": r["doc_id"], "tokens": len(r["tokens"])} for r in records]


def _build_eval_shard(gate: TokenizerGate, docs: list[dict]) -> tuple[np.ndarray, Manifest]:
    ids: list[int] = []
    for d in docs:
        ids.extend(gate.encode(d["text"]))
    arr = np.array(ids, dtype=np.uint16)
    content_hash = hash_tokens(arr)
    m = Manifest(
        shard_id=f"eval-holdout-000-{short(content_hash)}", kind="stream", lane="eval",
        source_ids=[docs[0]["source"]], document_ids=[d["doc_id"] for d in docs],
        tokenizer_id=config.TOKENIZER_ID, tokenizer_hash=gate.hash,
        token_count=int(arr.size), language="en", script="Latin",
        license="eval-only", provenance_tier="E",
        cleaning_pipeline_hash=pipeline_hash(), dedup_status="n/a", pii_status="n/a",
        contamination_status="is_benchmark", eval_overlap=True,
        never_train=True, anneal_reserve=False, content_hash=content_hash,
        parent_shard_ids=[], cleaning_counters={})
    return arr, m


def build_all(gate: TokenizerGate, log) -> list[Manifest]:
    """Tokenize every lane into shards, write payloads + manifests, return manifests."""
    config.SHARD_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFESTS.mkdir(parents=True, exist_ok=True)
    eos_id = gate.tok.token_to_id(EOS_TOKEN) or 0
    manifests: list[Manifest] = []

    for lane in config.LANES:
        docs = _read_lane(lane)
        per = DOCS_PER_SHARD[lane]
        chunks = [docs[i:i + per] for i in range(0, len(docs), per)]
        for idx, chunk in enumerate(chunks):
            # last shard of each lane is held back for the anneal (S5 reserve)
            reserve = (idx == len(chunks) - 1) and len(chunks) > 2
            if config.PACKING_POLICY[lane] == "concat_chop":
                arr, m, spans = _build_stream_shard(gate, lane, chunk, idx, eos_id, reserve)
                np.save(config.SHARD_DIR / f"{m.shard_id}.npy", arr)
                (config.SHARD_DIR / f"{m.shard_id}.spans.json").write_text(
                    canonical_json(spans), encoding="utf-8")
            else:
                payload, m, spans = _build_record_shard(gate, lane, chunk, idx, reserve)
                (config.SHARD_DIR / f"{m.shard_id}.json").write_text(
                    canonical_json(payload), encoding="utf-8")
            (config.MANIFESTS / f"{m.shard_id}.json").write_text(m.to_json(), encoding="utf-8")
            manifests.append(m)
            log.event("shard_created", shard_id=m.shard_id, lane=lane, kind=m.kind,
                      tokens=m.token_count, anneal_reserve=m.anneal_reserve)

    arr, m = _build_eval_shard(gate, _read_lane("eval"))
    np.save(config.SHARD_DIR / f"{m.shard_id}.npy", arr)
    (config.MANIFESTS / f"{m.shard_id}.json").write_text(m.to_json(), encoding="utf-8")
    manifests.append(m)
    log.event("shard_created", shard_id=m.shard_id, lane="eval", kind="stream",
              tokens=m.token_count, never_train=True)
    return manifests


def load_stream(shard_id: str) -> np.ndarray:
    return np.load(config.SHARD_DIR / f"{shard_id}.npy")


def load_records(shard_id: str) -> list[dict]:
    return json.loads((config.SHARD_DIR / f"{shard_id}.json").read_text(encoding="utf-8"))["records"]
