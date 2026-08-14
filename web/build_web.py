"""Bundle results + figures into web/ so the folder is Netlify-droppable.

    python run_all.py          # produces results/ and figures/
    python web/build_web.py    # copies them next to index.html

Then drag the `web/` folder onto Netlify. No build step, no dependencies, no
server -- index.html is self-contained apart from data.json and the PNGs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

KEYS = {"e1": "e1_geometry", "e2": "e2_arithmetic",
        "e2b": "e2b_residue_diagnostic", "e3": "e3_textlm",
        "e4": "e4_multimodal", "e5": "e5_decoding"}


def main() -> None:
    data = {}
    for short, stem in KEYS.items():
        p = RESULTS / f"{stem}.json"
        if p.exists():
            data[short] = json.loads(p.read_text(encoding="utf-8"))
            print(f"  + {stem}.json")
        else:
            print(f"  - {stem}.json missing (section will be skipped)")

    (WEB / "data.json").write_text(json.dumps(data), encoding="utf-8")

    n = 0
    for png in sorted(FIGURES.glob("*.png")):
        shutil.copy2(png, WEB / png.name)
        n += 1
    print(f"  + {n} figures")
    print(f"\nwrote {WEB/'data.json'}  ->  drag `web/` onto Netlify")


if __name__ == "__main__":
    main()
