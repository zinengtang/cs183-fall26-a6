# Assignment 6: Grounded Language with CLIP and Pragmatic Reference Games

**Due Friday, December 4, 2026 at 11:59 PM PT.**

All the work happens in `HW6.ipynb`. Set up with one of the two options below, then run the notebook from the top.

## Colab setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zinengtang/cs183-fall26-a6/blob/main/HW6.ipynb)

1. Click the badge.
2. *File → Save a copy in Drive*, so your work is saved.
3. Run the notebook from the top. The default CPU runtime is enough. A GPU (*Runtime → Change runtime type → T4 GPU*) makes it faster but is not needed.

The notebook needs nothing else from this repo: the models and CIFAR-10 download from Hugging Face.

## Prefer your own machine?

Clone this repo, then from it:

```bash
conda create -y -n ugrad-nlp python=3.11 jupyter jupyterlab && conda activate ugrad-nlp
pip install -r requirements.txt
jupyter lab
```

The three CLIP checkpoints take about 1.8 GB of disk in total (the third is only needed for Part 4). They are cached under `~/.cache/huggingface`, so they download only once.

**That is the whole setup.** Everything below is reference. Read it when something breaks.

---

## What is in it

| Part | Topic | Points |
| --- | --- | --- |
| 1 | CLIP: contrastive loss and zero-shot classification | 35 |
| 2 | Rational Speech Acts (RSA) with a hand-written lexicon | 30 |
| 3 | Neural RSA: CLIP as the literal listener in a reference game | 35 |
| 4 | **283A only:** speaking to an unknown partner | 20 |

EECS 183 students do Parts 1–3 (100 points). EECS 283A students do Parts 1–4 (120 points).

## How long it takes

| | |
| --- | --- |
| Part 1 | about 1 minute to encode 1,000 CIFAR-10 images on a CPU, plus the first model download |
| Part 2 | seconds; it is all NumPy |
| Part 3 | about 1 minute, plus downloading a second CLIP model |
| Part 4 | about 1 minute, plus downloading a third CLIP model |

On a fresh Colab CPU runtime, the whole notebook takes roughly 10–15 minutes, mostly downloads.

## If something breaks

**`NotImplementedError`**. That is the starter code. It means you reached a function you still have to write.

**Downloads from Hugging Face are slow, or fail with HTTP 429**. Anonymous downloads are rate-limited. Wait a minute and re-run the cell, or set an `HF_TOKEN` (*Colab: the key icon in the left sidebar*).

**`AttributeError` mentioning `pooler_output`, or `TypeError` in `torch.cat`**. Different `transformers` versions return different types from `get_image_features`. The notebook's `_as_tensor` helper handles both, so make sure you did not delete it, and always go through `encode_images` / `encode_texts`.

**Colab ran out of memory**. *Runtime → Restart session*, then run from the top. The notebook frees each extra CLIP model after it computes that model's scores, so this should be rare.

**The autograder says a graded cell was not found, or reports a `NameError`**. The autograder runs only the cells that start with `# GRADED:`, each on its own. Keep the header line exactly as given, and keep imports inside the cell. A graded cell cannot use variables such as `clip_b32` or `CLASSES` that are defined elsewhere in the notebook.

## Submitting

Upload two files to the Gradescope assignment **A6**:

* `HW6.ipynb`, your notebook. The functions in the `# GRADED:` cells are autograded. On Colab: *File → Download → Download .ipynb*.
* `HW6.pdf`, the notebook exported with all outputs. Every starred (★) question is graded from this, so check that your plots actually rendered before you export. Use *File → Print → Save as PDF*.

Working locally? Put `HW6.pdf` next to the notebook and run

```bash
python3 make_submission.py
```

It runs each graded cell on its own, as the autograder does, calls every function once on a tiny input, and lists anything still unimplemented with the part it belongs to. Then it packs both files into `submission.zip`. Passing it does not mean you pass the hidden tests. It means nothing is obviously missing.
