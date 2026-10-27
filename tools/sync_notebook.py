"""Generate xnids.ipynb from xnids.py so both files always contain identical code.

The script uses the "percent" cell format (the same one VS Code, Spyder and Jupytext understand):

    # %% [markdown]      -> markdown cell (each following line is a "# " comment)
    # %% optional title  -> code cell

Usage:
    python tools/sync_notebook.py            # (re)build xnids.ipynb from xnids.py
    python tools/sync_notebook.py --check    # exit 1 if the notebook's code differs from the script
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "xnids.py"
NOTEBOOK = ROOT / "xnids.ipynb"


def parse_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = list(buf)
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        cells.append((kind, "\n".join(lines)))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if line.startswith("# %% [markdown]") else "code"
            buf = [] if kind == "markdown" else ([f"# {line[4:].strip()}"] if line[4:].strip() else [])
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def build_notebook(cells):
    nb_cells = []
    for i, (kind, source) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"cell-{i:03d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def code_of(cells):
    return [src for kind, src in cells if kind == "code"]


def notebook_code(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]).rstrip() for c in nb["cells"] if c["cell_type"] == "code"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the notebook matches the script")
    args = ap.parse_args()
    cells = parse_cells(SCRIPT.read_text(encoding="utf-8"))
    if args.check:
        if not NOTEBOOK.exists() or notebook_code(NOTEBOOK) != code_of(cells):
            print("OUT OF SYNC: run `python tools/sync_notebook.py`")
            sys.exit(1)
        print(f"OK: {len(code_of(cells))} code cells identical in xnids.py and xnids.ipynb")
        return
    NOTEBOOK.write_text(json.dumps(build_notebook(cells), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK.name}: {len(cells)} cells ({len(code_of(cells))} code)")


if __name__ == "__main__":
    main()
