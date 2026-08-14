"""One command. Runs the invariant suite and every experiment, in order.

    python run_all.py            everything
    python run_all.py e1 e4      just those

Writes results/*.json, figures/*.png and results/run.log. Every claim in the
README is produced here; nothing is typed in by hand.
"""
from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
LOG = RESULTS / "run.log"

STAGES = [
    ("e0", "tests", "invariants: algebra, shift, inversion, modality widths"),
    ("e1", "experiments.e1_geometry", "geometry vs Kronecker under edits"),
    ("e2", "experiments.e2_arithmetic", "arithmetic: input x output-space grid"),
    ("e2b", "experiments.e2b_residue_diagnostic", "why both E2 fixes failed"),
    ("e3", "experiments.e3_textlm", "text LM control (a tie is predicted)"),
    ("e4", "experiments.e4_multimodal", "one projection, three modalities"),
    ("e5", "experiments.e5_decoding", "zero-parameter decoding, no output vocab"),
]


class Tee:
    def __init__(self, path: Path):
        self.f = io.open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def run_tests(log) -> dict:
    log("  running pytest on tests/")
    p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-1:]
    ok = p.returncode == 0
    log(f"  {tail[0] if tail else ''}")
    log(f"[{'PASS' if ok else 'FAIL'}] e0_invariants  {tail[0] if tail else ''}")
    return {"passed": ok, "output": tail[0] if tail else ""}


def main(which: list[str]) -> int:
    LOG.write_text("", encoding="utf-8")
    log = Tee(LOG)
    log("=" * 78)
    log("FIREFLY -- Fourier Interference Representations for Embedding Fields of any Length")
    log("Tejaskumar Reddy J  |  ERA V5  |  Kronecker Embeddings V2")
    log(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 78)

    # Merge rather than overwrite, so running a subset of stages leaves the
    # others' results in place instead of silently truncating the bundle.
    sp = RESULTS / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    t0, failed = time.time(), []
    for key, mod, desc in STAGES:
        if which and key not in which:
            continue
        log("")
        log(f"### {key.upper()}  {desc}")
        t = time.time()
        try:
            if key == "e0":
                summary[key] = run_tests(log)
            else:
                summary[key] = importlib.import_module(mod).run(log=log)
            summary[key]["_seconds"] = round(time.time() - t, 1)
        except Exception as exc:                      # keep going, report honestly
            log(f"[FAIL] {key}  {type(exc).__name__}: {exc}")
            summary[key] = {"error": f"{type(exc).__name__}: {exc}"}
            failed.append(key)
        log(f"    ({time.time() - t:.0f}s)")

    log("")
    log("=" * 78)
    log(f"total {time.time() - t0:.0f}s"
        + (f"   FAILED: {', '.join(failed)}" if failed else "   all stages completed"))
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log(f"wrote {RESULTS/'summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main([a.lower() for a in sys.argv[1:]]))
