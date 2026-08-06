#!/usr/bin/env python3
"""ONE-TIME author-side script: download real corpora from the Hugging Face Hub
and vendor them into data/raw/*.jsonl so that run_demo.py runs fully offline.

Every lane is real, published data with real provenance:

  web        HuggingFaceFW/fineweb-edu            (ODC-By-1.0)   url per document
  code       codeparrot/codeparrot-clean-valid    (Apache-2.0)   repo + path
  indic      ai4bharat/sangraha verified tel+hin  (CC-BY-4.0)    verified native
  agentic    glaiveai/glaive-function-calling-v2  (Apache-2.0)   real tool calls
  reasoning  openai/gsm8k train + open-r1/OpenR1-Math-220k       short + long traces
  eval       openai/gsm8k TEST split              held out, never_train=True

The agentic and reasoning lanes carry segment roles so the packer can put loss
on the model's own tokens only. In glaive, `FUNCTION RESPONSE:` is a tool
observation: training on it would teach the model to invent tool results instead
of calling the tool, so it is tagged `context`.

The eval lane is the *test* split of the same benchmark whose *train* split
feeds the reasoning lane -- a real train/test boundary for the firewall to
enforce, not a synthetic placeholder.

Not part of the graded demo path; its output is committed to the repo.
"""
from __future__ import annotations
import itertools
import json
import re
import shutil
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ERA = ROOT.parent
S2_TOK = ERA / "Session 2_Tokenization and vocabulary design" / "tokenizer-assignment" / "tokenizer.json"
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

N = {"web": 120, "code": 90, "indic_per_lang": 60,
     "agentic": 100, "gsm8k": 60, "openr1": 40, "eval": 16}
MAX_CHARS = 4000                      # keep documents small; this is a demo corpus


def emit(lane: str, docs: list[dict]) -> None:
    p = OUT / f"{lane}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    kb = p.stat().st_size / 1024
    print(f"  {lane:10s} {len(docs):4d} docs  {kb:8.1f} KB -> {p.name}")


def stream(repo: str, **kw):
    return load_dataset(repo, split=kw.pop("split", "train"), streaming=True, **kw)


# ---------------------------------------------------------------- plain lanes
def build_web() -> list[dict]:
    docs = []
    for i, r in enumerate(itertools.islice(stream("HuggingFaceFW/fineweb-edu",
                                                  name="sample-10BT"), N["web"] * 3)):
        t = (r.get("text") or "").strip()
        if len(t) < 400:
            continue
        docs.append({"doc_id": f"web-{len(docs):04d}", "lane": "web", "text": t[:MAX_CHARS],
                     "source": "HuggingFaceFW/fineweb-edu", "license": "ODC-By-1.0",
                     "provenance_tier": "B",
                     "origin": {"url": r.get("url"), "dump": r.get("dump")}})
        if len(docs) >= N["web"]:
            break
    return docs


def build_code() -> list[dict]:
    docs = []
    for r in itertools.islice(stream("codeparrot/codeparrot-clean-valid"), N["code"] * 3):
        t = (r.get("content") or "").strip()
        if len(t) < 300:
            continue
        docs.append({"doc_id": f"code-{len(docs):04d}", "lane": "code", "text": t[:MAX_CHARS],
                     "source": "codeparrot/codeparrot-clean-valid", "license": "Apache-2.0",
                     "provenance_tier": "B",
                     "origin": {"repo_name": r.get("repo_name"), "path": r.get("path")}})
        if len(docs) >= N["code"]:
            break
    return docs


def build_indic() -> list[dict]:
    docs = []
    for cfg, lang in (("verified/tel", "te"), ("verified/hin", "hi")):
        got = 0
        for r in itertools.islice(stream("ai4bharat/sangraha", data_dir=cfg), N["indic_per_lang"] * 4):
            t = (r.get("text") or "").strip()
            if len(t) < 400:
                continue
            docs.append({"doc_id": f"indic-{lang}-{got:04d}", "lane": "indic",
                         "text": t[:MAX_CHARS], "source": f"ai4bharat/sangraha:{cfg}",
                         "license": "CC-BY-4.0", "provenance_tier": "A",
                         "origin": {"language": lang, "sangraha_doc_id": r.get("doc_id"),
                                    "verified_native": True}})
            got += 1
            if got >= N["indic_per_lang"]:
                break
    return docs


# ------------------------------------------------------------- agentic lane
TURN = re.compile(r"^(USER|ASSISTANT|FUNCTION RESPONSE):\s*", re.MULTILINE)


def parse_glaive(chat: str) -> list[dict]:
    """Split a glaive conversation into role-tagged segments.

    USER and FUNCTION RESPONSE are context; ASSISTANT (including its
    <functioncall> payload) is what the model must learn to produce.
    """
    parts, segs = TURN.split(chat), []
    # TURN.split yields ['', ROLE, body, ROLE, body, ...]
    for i in range(1, len(parts) - 1, 2):
        role_raw, body = parts[i], parts[i + 1].strip()
        if not body:
            continue
        role = "model" if role_raw == "ASSISTANT" else "context"
        segs.append({"role": role, "text": f"{role_raw}: {body}\n"})
    return segs


def build_agentic() -> list[dict]:
    docs = []
    for r in itertools.islice(stream("glaiveai/glaive-function-calling-v2"), N["agentic"] * 4):
        chat = r.get("chat") or ""
        if "FUNCTION RESPONSE" not in chat:        # keep genuine tool-use trajectories
            continue
        segs = parse_glaive(chat)
        if not any(s["role"] == "model" for s in segs) or len(segs) < 4:
            continue
        sys_txt = (r.get("system") or "").strip()
        if sys_txt:
            segs.insert(0, {"role": "context", "text": sys_txt[:1200] + "\n"})
        docs.append({"doc_id": f"agentic-{len(docs):04d}", "lane": "agentic",
                     "segments": segs, "source": "glaiveai/glave-function-calling-v2",
                     "license": "Apache-2.0", "provenance_tier": "B",
                     "origin": {"has_tool_response": True,
                                "turns": len(segs)}})
        if len(docs) >= N["agentic"]:
            break
    # fix source typo in a single place
    for d in docs:
        d["source"] = "glaiveai/glaive-function-calling-v2"
    return docs


# ----------------------------------------------------------- reasoning lane
def build_reasoning() -> list[dict]:
    docs = []
    for r in itertools.islice(stream("openai/gsm8k", name="main"), N["gsm8k"]):
        docs.append({"doc_id": f"reasoning-gsm-{len(docs):04d}", "lane": "reasoning",
                     "band": "L1",
                     "segments": [{"role": "context", "text": f"Question: {r['question']}\n"},
                                  {"role": "model", "text": f"{r['answer']}\n"}],
                     "source": "openai/gsm8k:main:train", "license": "MIT",
                     "provenance_tier": "A", "origin": {"benchmark_family": "gsm8k",
                                                        "split": "train"}})
    for r in itertools.islice(stream("open-r1/OpenR1-Math-220k", name="default"), N["openr1"] * 3):
        sol = (r.get("solution") or "").strip()
        prob = (r.get("problem") or "").strip()
        if len(sol) < 400 or not prob:
            continue
        docs.append({"doc_id": f"reasoning-r1-{len(docs):04d}", "lane": "reasoning",
                     "band": "L2" if len(sol) < 2500 else "L3",
                     "segments": [{"role": "context", "text": f"Problem: {prob[:1200]}\n"},
                                  {"role": "model", "text": sol[:MAX_CHARS] + "\n"}],
                     "source": "open-r1/OpenR1-Math-220k", "license": "Apache-2.0",
                     "provenance_tier": "A",
                     "origin": {"problem_type": r.get("problem_type"), "uuid": r.get("uuid")}})
        if len(docs) >= N["gsm8k"] + N["openr1"]:
            break
    return docs


# ---------------------------------------------------------------- eval lane
def build_eval() -> list[dict]:
    """GSM8K *test* split. The reasoning lane trains on the *train* split of the
    same benchmark, so this is a real held-out boundary the firewall must hold."""
    docs = []
    for r in itertools.islice(stream("openai/gsm8k", name="main", split="test"), N["eval"]):
        docs.append({"doc_id": f"eval-gsm-{len(docs):04d}", "lane": "eval",
                     "text": f"Question: {r['question']}\nAnswer: {r['answer']}",
                     "source": "openai/gsm8k:main:test", "license": "eval-only",
                     "provenance_tier": "E", "never_train": True,
                     "origin": {"benchmark_family": "gsm8k", "split": "test"}})
    return docs


def main() -> int:
    shutil.copyfile(S2_TOK, ROOT / "data" / "tokenizer.json")
    print(f"tokenizer vendored <- {S2_TOK}")
    print("downloading real corpora from the Hugging Face Hub:")
    for lane, fn in (("web", build_web), ("code", build_code), ("indic", build_indic),
                     ("agentic", build_agentic), ("reasoning", build_reasoning),
                     ("eval", build_eval)):
        try:
            emit(lane, fn())
        except Exception as e:
            print(f"  {lane:10s} FAILED {type(e).__name__}: {str(e)[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
