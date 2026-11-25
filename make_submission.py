#!/usr/bin/env python3
"""Check HW6.ipynb and package it with HW6.pdf into a zip you can upload to Gradescope.

    python3 make_submission.py

Before zipping, it runs every `# GRADED:` cell the same way the autograder does (on
its own, without the rest of the notebook) and calls each function once on a tiny
input. Finding a missing function here takes a second; finding it from a Gradescope
score takes a submission. Passing these checks does NOT mean the hidden tests pass.
"""

import argparse
import json
import os
import re
import sys
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))

# graded cell name -> functions it must define
CELLS = {
    "clip_contrastive_loss": ["clip_contrastive_loss"],
    "zero_shot": ["build_zero_shot_weights", "zero_shot_logits"],
    "execute": ["execute"],
    "rsa": ["rsa_L0", "rsa_S1", "rsa_L1"],
    "yes_no_margin": ["yes_no_margin"],
}
PART = {
    "clip_contrastive_loss": "1.1", "build_zero_shot_weights": "1.2", "zero_shot_logits": "1.2",
    "execute": "2.2", "rsa_L0": "3.1", "rsa_S1": "3.1", "rsa_L1": "3.1", "yes_no_margin": "4.1",
}
EXTRA_CREDIT = {"yes_no_margin"}   # Part 4 is optional


def smoke_tests():
    """One call per function, on inputs small enough to check by hand."""
    import numpy as np
    import torch

    with np.errstate(divide="ignore"):
        fg_phi = np.log(np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]]))   # Frank & Goodman (2012)
    red_above = [{"color": "red", "shape": "circle", "box": (10, 10, 30, 30)},
                 {"color": "blue", "shape": "square", "box": (100, 10, 120, 30)}]

    def returns_scalar(ns):
        out = ns["clip_contrastive_loss"](torch.randn(4, 8), torch.randn(4, 8), 10.0)
        if not (torch.is_tensor(out) and out.dim() == 0):
            return "should return a 0-dim tensor"

    def zs_weights(ns):
        out = ns["build_zero_shot_weights"](["cat", "dog"], ["a {}", "the {}"], lambda t: torch.randn(len(t), 8))
        if tuple(out.shape) != (2, 8):
            return f"shape {tuple(out.shape)}, expected (2, 8)"

    def executes(ns):
        prog = {"op": "relation", "rel": "above", "a": {"color": "red"}, "b": {"shape": "square"}}
        if not ns["execute"](prog, red_above):
            return "the red circle is above the blue square, but execute returned False"

    def fg_listener(ns):
        L1 = np.exp(np.asarray(ns["rsa_L1"](ns["rsa_S1"](ns["rsa_L0"](fg_phi)))))
        if not np.allclose(L1[0], [0.6, 0.4, 0.0], atol=1e-6):
            return f"L1('blue') = {np.round(L1[0], 3)}, expected [0.6, 0.4, 0.0]"

    def shape_of(fn, args, shape):
        def check(ns):
            out = np.asarray(ns[fn](*args))
            if out.shape != shape:
                return f"shape {out.shape}, expected {shape}"
        return check

    return {
        "clip_contrastive_loss": returns_scalar,
        "build_zero_shot_weights": zs_weights,
        "zero_shot_logits": lambda ns: None if tuple(ns["zero_shot_logits"](torch.randn(5, 8), torch.randn(2, 8), 10.0).shape) == (5, 2) else "shape should be [N, K]",
        "execute": executes,
        "rsa_L0": shape_of("rsa_L0", (fg_phi,), (4, 3)),
        "rsa_S1": shape_of("rsa_S1", (np.zeros((4, 3)) - np.log(3),), (3, 4)),
        "rsa_L1": fg_listener,
        "yes_no_margin": lambda ns: None if abs(float(ns["yes_no_margin"](torch.zeros(10), [1, 2], [3])) - np.log(2)) < 1e-5 else "uniform logits with 2 yes ids and 1 no id should give log 2",
    }


def graded_cells(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = {}
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        lines = [l for l in src.splitlines() if l.strip()]
        m = re.match(r"#\s*GRADED\s*:?\s*(\w+)", lines[0].strip()) if lines else None
        if m:
            cells[m.group(1)] = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", default=os.path.join(REPO, "HW6.ipynb"))
    parser.add_argument("--pdf", default=os.path.join(REPO, "HW6.pdf"),
                        help="PDF export of the notebook; the written questions are graded from it")
    parser.add_argument("--out", default=os.path.join(REPO, "submission.zip"))
    args = parser.parse_args()

    if not os.path.exists(args.notebook):
        sys.exit(f"no notebook at {args.notebook} -- save your work as HW6.ipynb next to this script")

    cells = graded_cells(args.notebook)
    tests = smoke_tests()
    problems = []
    print(f"{'part':>5}  {'function':<32} status")
    print("-" * 78)
    for cell, fns in CELLS.items():
        ns = {"__name__": "submission_check"}
        cell_error = None
        if cell not in cells:
            cell_error = f"cell '# GRADED: {cell}' not found -- do not edit the header line"
        else:
            try:
                exec(compile(cells[cell], f"<GRADED: {cell}>", "exec"), ns)
            except Exception as e:  # noqa: BLE001
                hint = " (graded cells must be self-contained)" if isinstance(e, NameError) else ""
                cell_error = f"the cell itself fails to run: {type(e).__name__}: {e}{hint}"
        for fn in fns:
            optional = fn in EXTRA_CREDIT
            label = f"{fn}{' (extra credit)' if optional else ''}"
            if cell_error:
                status = cell_error
            elif fn not in ns:
                status = "not defined in the graded cell"
            else:
                try:
                    complaint = tests[fn](ns)
                    status = f"runs, but {complaint}" if complaint else "ok"
                except NotImplementedError:
                    status = "not implemented yet"
                except Exception as e:  # noqa: BLE001
                    status = f"raises {type(e).__name__}: {e}"
            print(f"{PART[fn]:>5}  {label:<32} {status}")
            if status != "ok" and not optional:
                problems.append(fn)

    pdf = args.pdf if os.path.exists(args.pdf) else None
    print("-" * 78)
    if pdf:
        print(f"{'':>5}  {os.path.basename(pdf):<32} ok ({os.path.getsize(pdf) / 1024:.0f} KB)")
    else:
        print(f"{'':>5}  {'HW6.pdf':<32} not found -- export the notebook to PDF; every ★ question is graded from it")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.notebook, "HW6.ipynb")
        if pdf:
            z.write(pdf, "HW6.pdf")

    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB, {1 + bool(pdf)} files)")
    if problems:
        print("\nSome functions above still need work. The zip was written anyway, so you can submit "
              "partial progress, but those parts will not score.")
    elif not pdf:
        print("\nThe code checks out, but HW6.pdf is missing: the written (★) questions are graded from it.")
    else:
        print("\nEverything checks out. Upload submission.zip to Gradescope (A6).")


if __name__ == "__main__":
    main()
