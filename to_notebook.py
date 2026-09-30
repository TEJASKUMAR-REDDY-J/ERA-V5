"""Convert train.py (jupytext percent format) into reversible.ipynb.

One source of truth: the .py is what gets run and verified locally, and the
notebook is generated from it, so the two cannot drift.

    python to_notebook.py
"""
import re
from pathlib import Path

import nbformat as nbf

SRC = Path(__file__).parent / "train.py"
OUT = Path(__file__).parent / "reversible.ipynb"

COLAB_SETUP = '''# Colab: a T4 is enough. Runtime > Change runtime type > T4 GPU.
!pip -q install datasets

import os
# "full"     = the assignment: 20M parameters, 50M tokens (~45 min on a T4)
# "local20m" = same model, 1.5M tokens, for a CPU box
# "rehearsal"= small model, for a quick end-to-end check
os.environ["SCALE"] = "full"
os.environ["MEM_BUDGET_GIB"] = "12"     # only used when there is no GPU
'''


def split_cells(text):
    """Split on '# %%' markers into (kind, source) pairs."""
    parts = re.split(r"^# %%", text, flags=re.M)
    cells = []
    for p in parts:
        if not p.strip():
            continue
        is_md = p.lstrip().startswith("[markdown]")
        body = p.lstrip()[len("[markdown]"):] if is_md else p
        if is_md:
            lines = [re.sub(r"^# ?", "", ln) for ln in body.strip("\n").split("\n")]
            src = "\n".join(lines).strip("\n")
        else:
            src = body.strip("\n")
        if src.strip():
            cells.append(("markdown" if is_md else "code", src))
    return cells


def main():
    text = SRC.read_text(encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_code_cell(COLAB_SETUP.strip()))
    for kind, src in split_cells(text):
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "markdown"
                        else nbf.v4.new_code_cell(src))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    }
    nbf.write(nb, OUT)
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"wrote {OUT.name}: {n_code} code cells, {n_md} markdown cells")


if __name__ == "__main__":
    main()
