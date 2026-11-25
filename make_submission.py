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
    "rsa": ["literal_listener", "pragmatic_speaker", "pragmatic_listener"],
    "neural_rsa": ["context_log_L0", "literal_speaker", "neural_log_S1", "neural_log_L1"],
    "population_speaker": ["population_speaker"],
}
PART = {
    "clip_contrastive_loss": "1.1", "build_zero_shot_weights": "1.2", "zero_shot_logits": "1.2",
    "literal_listener": "2.1", "pragmatic_speaker": "2.1", "pragmatic_listener": "2.1",
    "context_log_L0": "3.1", "literal_speaker": "3.1", "neural_log_S1": "3.1", "neural_log_L1": "3.1",
    "population_speaker": "4.1",
}
ONLY_283A = {"population_speaker"}


def smoke_tests():
    """One call per function, on inputs small enough to check by hand."""
    import numpy as np
    import torch

    lex = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]])    # Frank & Goodman (2012)
    prior = np.ones(3) / 3
    scores = np.array([[30.0, 21.0, 25.0], [22.0, 29.0, 25.0], [24.0, 23.0, 28.0]])  # [C, U]

    def fg_listener(ns):
        L1 = np.asarray(ns["pragmatic_listener"](ns["pragmatic_speaker"](ns["literal_listener"](lex, prior), 1.0), prior))
        if not np.allclose(L1[0], [0.6, 0.4, 0.0], atol=1e-6):
            return f"L1('blue') = {np.round(L1[0], 3)}, expected [0.6, 0.4, 0.0]"

    def returns_scalar(ns):
        out = ns["clip_contrastive_loss"](torch.randn(4, 8), torch.randn(4, 8), 10.0)
        if not (torch.is_tensor(out) and out.dim() == 0):
            return "should return a 0-dim tensor"

    def zs_weights(ns):
        out = ns["build_zero_shot_weights"](["cat", "dog"], ["a {}", "the {}"], lambda t: torch.randn(len(t), 8))
        if tuple(out.shape) != (2, 8):
            return f"shape {tuple(out.shape)}, expected (2, 8)"

    def shape_of(fn, args, shape):
        def check(ns):
            out = np.asarray(ns[fn](*args))
            if out.shape != shape:
                return f"shape {out.shape}, expected {shape}"
        return check

    def log_softmax_rows(x):
        x = x - x.max(1, keepdims=True)
        return x - np.log(np.exp(x).sum(1, keepdims=True))

    log_L0 = log_softmax_rows(scores.T)                                  # [U, C]
    return {
        "clip_contrastive_loss": returns_scalar,
        "build_zero_shot_weights": zs_weights,
        "zero_shot_logits": lambda ns: None if tuple(ns["zero_shot_logits"](torch.randn(5, 8), torch.randn(2, 8), 10.0).shape) == (5, 2) else "shape should be [N, K]",
        "literal_listener": shape_of("literal_listener", (lex, prior), (4, 3)),
        "pragmatic_speaker": shape_of("pragmatic_speaker", (lex / lex.sum(1, keepdims=True), 1.0), (3, 4)),
        "pragmatic_listener": fg_listener,
        "context_log_L0": shape_of("context_log_L0", (scores,), (3, 3)),
        "literal_speaker": lambda ns: None if int(ns["literal_speaker"](scores, 0)) == 0 else "S0 should pick argmax_u scores[target, u]",
        "neural_log_S1": shape_of("neural_log_S1", (log_L0,), (3, 3)),
        "neural_log_L1": shape_of("neural_log_L1", (log_softmax_rows(scores),), (3, 3)),
        "population_speaker": lambda ns: None if 0 <= int(ns["population_speaker"]([log_L0, log_L0], 0, np.zeros(3), "min")) < 3 else "index out of range",
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
    print(f"{'part':>5}  {'function':<26} status")
    print("-" * 72)
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
            optional = fn in ONLY_283A
            label = f"{fn}{' (283A)' if optional else ''}"
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
            print(f"{PART[fn]:>5}  {label:<26} {status}")
            if status != "ok" and not optional:
                problems.append(fn)

    pdf = args.pdf if os.path.exists(args.pdf) else None
    print("-" * 72)
    if pdf:
        print(f"{'':>5}  {os.path.basename(pdf):<26} ok ({os.path.getsize(pdf) / 1024:.0f} KB)")
    else:
        print(f"{'':>5}  {'HW6.pdf':<26} not found -- export the notebook to PDF; every ★ question is graded from it")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.notebook, "HW6.ipynb")
        if pdf:
            z.write(pdf, "HW6.pdf")

    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB, {1 + bool(pdf)} files)")
    if problems or not pdf:
        print("\nSome items above still need work. The zip was written anyway, so you can submit "
              "partial progress, but those parts will not score.")
    else:
        print("\nEverything checks out. Upload submission.zip to Gradescope (A6).")


if __name__ == "__main__":
    main()
