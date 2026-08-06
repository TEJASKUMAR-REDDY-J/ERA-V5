"""run.log writer. Every significant event is one structured line, so the log is
both human-readable and machine-checkable. [PASS]/[FAIL] markers are emitted by
`check()`, which takes a boolean the caller computed — never a literal."""
from __future__ import annotations
import time
from pathlib import Path

from . import config
from .hashing import canonical_json


class RunLog:
    def __init__(self, path: Path | None = None, echo: bool = True):
        self.path = Path(path or (config.ARTIFACTS / "run.log"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.checks: list[dict] = []
        self.t0 = time.time()

    def _write(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    def event(self, name: str, **fields) -> None:
        t = f"{time.time() - self.t0:8.2f}s"
        detail = canonical_json(fields) if fields else ""
        self._write(f"[{t}] EVENT {name} {detail}".rstrip())

    def section(self, title: str) -> None:
        self._write("")
        self._write("=" * 78)
        self._write(f"== {title}")
        self._write("=" * 78)

    def check(self, name: str, passed: bool, **evidence) -> bool:
        """Record a PASS/FAIL. `passed` must be computed by the caller from real
        state; this function only reports it."""
        tag = "[PASS]" if passed else "[FAIL]"
        detail = canonical_json(evidence) if evidence else ""
        self._write(f"{tag} {name} {detail}".rstrip())
        self.checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        return passed

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()
