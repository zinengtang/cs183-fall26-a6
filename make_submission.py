#!/usr/bin/env python3
"""Check HW6.ipynb and hw6_results.json, and package them with HW6.pdf into a zip for Gradescope.

    python3 make_submission.py

The autograder grades hw6_results.json, which the last cell of the notebook writes after
calling your graded functions on fixed inputs. This script checks that the file exists (or
recovers it from the notebook's last cell), lists any function that raised an error, and zips
the notebook, the results and the PDF. Passing these checks does NOT mean the values are correct.
"""

import argparse
import json
import os
import sys
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))

PART = {
    "clip_loss": "1.1", "clip_loss_scaled_inputs": "1.1", "clip_loss_swapped": "1.1", "clip_loss_grad_norm": "1.1",
    "zs_weights_single": "1.2", "zs_weights_ensemble": "1.2", "zs_logits": "1.2",
    "execute": "2.2", "parse_program": "3.1", "check_caption": "3.2", "vote": "3.3",
    "rsa_L0": "4.1", "rsa_S1": "4.1", "rsa_L1": "4.1", "rsa_fg_L1": "4.1",
    "yes_no_margin": "5.1", "yes_no_margin_big": "5.1", "yes_no_margin_uniform": "5.1",
}
EXTRA_CREDIT = {"yes_no_margin", "yes_no_margin_big", "yes_no_margin_uniform"}   # Part 5 is optional


def is_results(d):
    return isinstance(d, dict) and isinstance(d.get("functions"), dict) and "version" in d


def from_notebook(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in reversed(nb.get("cells", [])):
        for out in cell.get("outputs", []) if cell.get("cell_type") == "code" else []:
            d = out.get("data", {}).get("application/json")
            if isinstance(d, str):
                try:
                    d = json.loads(d)
                except json.JSONDecodeError:
                    continue
            if is_results(d):
                return d
    return None


def errors(v):
    """The error messages inside one recorded value (a value, an error dict, or a list of them)."""
    if isinstance(v, dict) and set(v) == {"error"}:
        return [v["error"]]
    if isinstance(v, list):
        return [x["error"] for x in v if isinstance(x, dict) and set(x) == {"error"}]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", default=os.path.join(REPO, "HW6.ipynb"))
    parser.add_argument("--results", default=os.path.join(REPO, "hw6_results.json"))
    parser.add_argument("--pdf", default=os.path.join(REPO, "HW6.pdf"),
                        help="PDF export of the notebook; the written questions are graded from it")
    parser.add_argument("--out", default=os.path.join(REPO, "submission.zip"))
    args = parser.parse_args()

    if not os.path.exists(args.notebook):
        sys.exit(f"no notebook at {args.notebook} -- save your work as HW6.ipynb next to this script")

    results, source = None, None
    if os.path.exists(args.results):
        with open(args.results, encoding="utf-8") as f:
            results, source = json.load(f), args.results
    if not is_results(results):
        results = from_notebook(args.notebook)
        if results is not None:
            source = "the notebook's last cell"
            with open(args.results, "w") as f:
                json.dump(results, f)
    if results is None:
        sys.exit("no results found: run the notebook to the end (the last cell writes hw6_results.json), then rerun this script")

    print(f"results from {source if source == args.results else source + ' (saved to ' + os.path.basename(args.results) + ')'}\n")
    print(f"{'part':>5}  {'check':<26} status")
    print("-" * 78)
    problems = []
    for key, part in PART.items():
        errs = errors(results["functions"].get(key, {"error": "missing"}))
        status = "ran" if not errs else f"error: {errs[0][:60]}" + (f" (+{len(errs) - 1} more)" if len(errs) > 1 else "")
        label = key + (" (extra credit)" if key in EXTRA_CREDIT else "")
        print(f"{part:>5}  {label:<26} {status}")
        if errs and key not in EXTRA_CREDIT:
            problems.append(key)
    if not isinstance(results.get("pipeline", {}).get("perceived_exactly"), (int, float)):
        print(f"{'2.2':>5}  {'perceive (pipeline)':<26} not recorded: run Part 2 before the last cell")
        problems.append("perceive")

    pdf = args.pdf if os.path.exists(args.pdf) else None
    print("-" * 78)
    if pdf:
        print(f"{'':>5}  {os.path.basename(pdf):<26} ok ({os.path.getsize(pdf) / 1024:.0f} KB)")
    else:
        print(f"{'':>5}  {'HW6.pdf':<26} not found -- export the notebook to PDF; every ★ question is graded from it")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.notebook, "HW6.ipynb")
        z.write(args.results, "hw6_results.json")
        if pdf:
            z.write(pdf, "HW6.pdf")

    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB, {2 + bool(pdf)} files)")
    if problems:
        print("\nSome checks above raised errors. The zip was written anyway, so you can submit "
              "partial progress, but those parts will not score.")
    elif not pdf:
        print("\nThe results look complete, but HW6.pdf is missing: the written (★) questions are graded from it.")
    else:
        print("\nEverything ran. Upload submission.zip to Gradescope (A6).")


if __name__ == "__main__":
    main()
