# English → Hindi NMT with OpenNMT-py: FastText vs BPEmb Pretrained Embeddings

A Jupyter notebook (`assignment-9-8-10-illm.ipynb`) that trains a small **Transformer** translation model with **OpenNMT-py** twice, once per pretrained embedding type, and compares them with **sacreBLEU / chrF** overall and by sentence-length bin.

| Experiment | Embeddings | Tokenisation |
|---|---|---|
| 1. **FastText** | Pretrained wiki word vectors (300-d) | Word-level (lower-cased, punctuation split) |
| 2. **BPEmb** | Pretrained subword vectors (300-d, vocab 10k) | BPEmb SentencePiece subwords |

Both runs use the same data, splits, model size and training steps, so the only difference is the embedding type.

## Dataset

- **ILCI subset of [`ai4bharat/BPCC`](https://huggingface.co/datasets/ai4bharat/BPCC)** (Hugging Face), English → Hindi (`eng_Latn` → `hin_Deva`).
- Sentences are split into **4 length bins** by the number of words in the English sentence:

| Bin | English length |
|---|---|
| bin1 | ≤ 10 words |
| bin2 | 11–20 words |
| bin3 | 21–29 words |
| bin4 | ≥ 30 words |

- Each bin is split separately (seed 42): up to **5000 train**, **100 valid**, and **500 test** sentences per bin, giving 20,000 train / 400 valid / 2,000 test in total.

## Requirements

- **Python 3.9+**. The notebook handles OpenNMT-py's version limits itself: OpenNMT-py 3.x needs `torch<2.3` and `pyonmttok`, which only support Python ≤ 3.11/3.12. If your Python is newer (e.g. 3.13 on Kaggle), the first cell automatically creates a separate **Python 3.11 virtual environment** (`onmt_py311`) using `uv` and runs the OpenNMT commands there.
- A **Hugging Face account and access token** (the notebook calls `login()` to download the dataset).
- **Internet access** to download:
  - the BPCC dataset from Hugging Face,
  - FastText wiki vectors from `dl.fbaipublicfiles.com` (the notebook streams the files and keeps only the 200k most frequent vectors, so it does not download several GB),
  - BPEmb models from `nlp.h-its.org`.
- A GPU is **optional**. The notebook detects CUDA automatically and falls back to CPU.
- The packages below are installed by the first cell; no manual install is needed.

| Installed in the notebook's Python | Installed in the OpenNMT Python |
|---|---|
| `sentencepiece`, `sacrebleu`, `datasets`, `huggingface_hub`, `pandas`, `bpemb`, `requests`, `tqdm` | `numpy<2`, `OpenNMT-py` |

## How to Run

### Option A: Kaggle / Colab (what the notebook was run on)

1. Upload `assignment-9-8-10-illm.ipynb` and open it.
2. Turn on internet access (and a GPU if available).
3. Run all cells from top to bottom.
4. When prompted by `login()`, paste your Hugging Face token.

### Option B: Locally

```bash
pip install jupyter
jupyter notebook assignment-9-8-10-illm.ipynb
```

Then run all cells in order and enter your Hugging Face token when asked.

**Run time (from the saved run, CPU):** FastText about 14 min, BPEmb about 7 min.

## Notebook Walkthrough

| Section | What it does |
|---|---|
| 0. Environment check | Installs packages; creates a Python 3.11 env for OpenNMT-py if needed |
| 1. Configuration | Language pair, data sizes, embedding and Transformer settings |
| 2. Load data | Downloads and cleans the ILCI English–Hindi pairs |
| 3. Length bins | Builds the length-binned train / valid / test splits |
| 4. Text helpers | `word_tok` (Devanagari-safe tokeniser) and `norm_eval` (same normalisation for references and hypotheses before BLEU) |
| 5–6. Embeddings | Builds the FastText `.vec` files and exports BPEmb vectors to word2vec format |
| 7. `run_experiment` | Builds vocab, trains, translates with beam search, and computes BLEU/chrF overall and per bin |
| 8–9. Experiments | Runs FastText, then BPEmb |
| 10–11. Comparison | Bin-wise summary table, analysis, and example translations |
| Last cell | Packages the outputs into `illm_assignment_outputs.zip` |

## Key Configuration (Section 1)

| Setting | Value | Meaning |
|---|---|---|
| `SOURCE_LANG` / `TARGET_LANG` | `eng_Latn` / `hin_Deva` | Change the target to use another Indian language (see `LANG2ISO`) |
| `EXPERIMENTS` | `["fasttext", "bpemb"]` | Which embedding types to run |
| `EMB_DIM` | 300 | Embedding and hidden size (`HEADS` must divide it) |
| `FT_VOCAB_LIMIT` | 200000 | Number of FastText vectors kept |
| `WORD_VOCAB_SIZE` | 30000 | Max word vocabulary for the FastText model |
| `BPEMB_VS` | 10000 | BPEmb subword vocabulary size |
| `FREEZE_EMB` | `False` | `False` fine-tunes the embeddings, `True` freezes them |
| `TRAIN_STEPS` | 5000 | Training steps |
| `LAYERS, HEADS, FF_SIZE` | 3, 6, 1024 | Transformer shape |
| `BEAM_SIZE` | 5 | Beam search width |
| `MAX_TRAIN_PER_BIN` / `MAX_TEST_PER_BIN` / `VALID_PER_BIN` | 5000 / 500 / 100 | Data per bin (`None` = use all) |
| `FORCE_RERUN` | `False` | If `False`, training and translation are skipped when their outputs already exist |

## Results (from the saved run)

Test set: 2,000 sentences (500 per bin). BLEU and chrF use `norm_eval` normalisation.

| Bin | FastText BLEU | FastText chrF | BPEmb BLEU | BPEmb chrF |
|---|---|---|---|---|
| bin1 (short) | 13.80 | 26.52 | 12.13 | 25.79 |
| bin2 | 9.93 | 25.04 | 9.27 | 23.39 |
| bin3 | 7.55 | 23.99 | 7.20 | 22.41 |
| bin4 (long) | 8.83 | 25.20 | 6.59 | 21.61 |
| **Overall** | **9.11** | **24.94** | **7.75** | **22.56** |

| | FastText | BPEmb |
|---|---|---|
| Vocabulary (src / tgt) | 28,497 / 29,996 | 8,918 / 8,996 |
| Pretrained-embedding coverage (src / tgt) | 76.6% / 68.5% | 99.6% / 99.4% |
| Source OOV rate | about 2.6–3.8% | about 0.1–0.2% |
| Total time | 14 min | 7 min |

**Observations**

- FastText scored a higher overall BLEU (9.11 vs 7.75) and chrF (24.94 vs 22.56) in this run.
- BPEmb BLEU falls steadily as sentences get longer. FastText does not (bin4 is above bin3).
- BPEmb has almost no out-of-vocabulary tokens and much better pretrained coverage, but it produces longer sequences, which can hurt long sentences.
- Absolute scores are low because this is a small, short CPU-scale run (3 layers, 5,000 steps, 20k training pairs).

## Outputs

Running the last cell creates `illm_assignment_outputs.zip` containing:

- `summary_binwise_comparison.csv`: the comparison table above
- `<exp>_bin_results.csv`: per-bin BLEU, chrF and OOV for each experiment
- `<exp>_test_translations.csv`: source, reference and hypothesis for every test sentence
- `results_report.txt`: plain-text report
- `<exp>/`: configs, train/valid/test files, vocab files and raw hypotheses (no model checkpoints)

Working files go to `mt_emb_work/` (embeddings, tokenised data, checkpoints).

## Troubleshooting

| Problem | Fix |
|---|---|
| `OpenNMT-py import failed` | Re-run the first cell. On Python 3.12+ it builds a separate 3.11 environment, which takes a few minutes and needs internet. |
| Hugging Face login or dataset error | Check that your token is valid and that you can access `ai4bharat/BPCC`. |
| `No ILCI file found for this language` | The target language code does not match a file in BPCC; check `TARGET_LANG`. |
| Training or translation is skipped | Outputs already exist in `mt_emb_work/`. Set `FORCE_RERUN = True` or delete that folder. |
| OpenMP crash on macOS | Already handled in the notebook via `KMP_DUPLICATE_LIB_OK`. |
| Very slow training | Use a GPU, or reduce `TRAIN_STEPS` and `MAX_TRAIN_PER_BIN`. |

## Tech Stack

Python · OpenNMT-py · PyTorch · FastText wiki vectors · BPEmb (SentencePiece) · sacreBLEU · Hugging Face `datasets` / `huggingface_hub` · pandas
