# Assignment 6: Vision and Language

**Due Fri Dec 4, 11:59 PM PT.** All the work is in `HW6.ipynb`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zinengtang/cs183-fall26-a6/blob/main/HW6.ipynb)

## Setup

**Colab:** click the badge, then *File → Save a copy in Drive*, then run the notebook from the top. The default CPU runtime is enough; there is no training and no GPU is needed.

**Local:**

```bash
conda create -y -n ugrad-nlp python=3.11 jupyterlab && conda activate ugrad-nlp
pip install -r requirements.txt && jupyter lab
```

You only download one CLIP model (about 600 MB) and the CIFAR-10 test split (24 MB).

## Submit

Upload `HW6.ipynb` and `HW6.pdf` (*File → Print → Save as PDF*) to Gradescope **A6**.

If you work locally, `python3 make_submission.py` checks your graded functions and builds `submission.zip`.

## If something breaks

- **`NotImplementedError`:** a function you still need to write.
- **A Hugging Face download is slow or fails (HTTP 429):** wait a minute and re-run the cell, or set an `HF_TOKEN`.
- **The autograder can't find a cell, or reports a `NameError`:** keep each `# GRADED:` header line as it is, put imports inside the cell, and don't use variables from other cells.
