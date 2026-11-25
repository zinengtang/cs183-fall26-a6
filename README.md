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

Parts 1, 2 and 4 download one CLIP model (about 600 MB) and the CIFAR-10 test split (24 MB). Part 3 downloads a 0.5B LLM (about 1 GB) and takes about 10 minutes on a CPU. Optional Part 5 (+10 extra credit) downloads a small vision-language model (about 1 GB) and takes about 10 minutes on a CPU. A free Colab GPU (*Runtime → Change runtime type → T4 GPU*) makes Parts 3 and 5 take seconds.

## Submit

Upload `HW6.ipynb`, `HW6.pdf` (*File → Print → Save as PDF*) and `hw6_results.json` to Gradescope **A6**. The last cell of the notebook writes `hw6_results.json`; run it after *Restart and run all*, then download it from Colab's *Files* panel.

If you work locally, `python3 make_submission.py` checks `hw6_results.json` and builds `submission.zip`.

## If something breaks

- **`NotImplementedError`:** a function you still need to write.
- **A Hugging Face download is slow or fails (HTTP 429):** wait a minute and re-run the cell, or set an `HF_TOKEN`.
- **The autograder says no results were found:** run the last cell (*Save your results*) and upload `hw6_results.json`.
- **A check reports an error in `hw6_results.json`:** that function raised an exception on the fixed inputs; the message names the error.
