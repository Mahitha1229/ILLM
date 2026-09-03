
# Indic Language Identification (from scratch)

A from-scratch language identification system for 23 Indic languages, built without any ML libraries — the TF-IDF vectorizer, softmax logistic regression classifier, and evaluation metrics are all implemented in pure Python.

## Overview

Given a sentence, the model predicts which of 23 Indian languages it's written in. The pipeline:

1. **Data collection** — streams text from [`ai4bharat/IndicCorpV2`](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) on Hugging Face, one language at a time.
2. **Sentence tokenization** — a lightweight regex-based sentence splitter that handles Indic sentence-ending punctuation (`। ॥ ۔ ؟` alongside `. ! ?`).
3. **Filtering & sampling** — keeps sentences with at least 15 characters and 3 words, deduplicates, and collects up to 1,000 sentences per language.
4. **Stratified train/val/test split** — 80/10/10 per language.
5. **Feature extraction** — word unigrams/bigrams + character 2/3/4-grams.
6. **TF-IDF vectorization** — custom implementation (min document frequency filtering, vocabulary capped at 20,000 features, L2-normalized vectors).
7. **Classification** — a multi-class softmax logistic regression trained with mini-batch gradient descent and L2 regularization, all implemented from scratch (no scikit-learn/PyTorch/TensorFlow).
8. **Evaluation** — per-class precision, recall, F1, macro-F1, and accuracy, also implemented from scratch.

## Languages covered

23 languages/scripts from IndicCorpV2:

`asm_Beng, ben_Beng, brx_Deva, doi_Deva, gom_Deva, guj_Gujr, hin_Deva, kan_Knda, kas_Arab, mai_Deva, mal_Mlym, mar_Deva, mni_Mtei, npi_Deva, ory_Orya, pan_Guru, san_Deva, snd_Deva, tam_Taml, tel_Telu, urd_Arab, khasi, santhali`

(Assamese, Bengali, Bodo, Dogri, Konkani, Gujarati, Hindi, Kannada, Kashmiri, Maithili, Malayalam, Marathi, Manipuri, Nepali, Odia, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu, Khasi, Santali)

## Notebook structure

| Cell | Purpose |
|---|---|
| 1 | Install `datasets` |
| 2 | Mount Google Drive, set project directory |
| 3 | Sentence tokenizer |
| 4 | Data collection from IndicCorpV2 + stratified train/val/test split, saved as JSONL |
| 5 | Custom TF-IDF vectorizer (word + char n-gram features) |
| 6 | Custom softmax logistic regression classifier |
| 7 | Custom evaluation metrics (precision, recall, F1, macro-F1, accuracy) |
| 8 | Training script: loads data, fits vectorizer + classifier, evaluates on the test set, prints a per-language metrics table |

## Requirements

- Python 3
- `datasets` (Hugging Face) — for streaming IndicCorpV2
- Google Colab (notebook mounts Google Drive for persistent storage), or adapt the `PROJECT_DIR` path to run locally

No other ML/NLP libraries are used — TF-IDF, the classifier, and all metrics are implemented from scratch in pure Python.

## Usage

1. Open the notebook in Google Colab (or a local Jupyter environment, after adjusting the Drive-mount cell).
2. Run cells in order. The data-collection cell streams and caches per-language raw data, so re-running is resumable — it skips languages that already have a cached `_raw_<lang>.jsonl` file.
3. The final cell trains the classifier and prints:
   - A worked TF-IDF example for one training sentence
   - Training progress (loss and validation accuracy per epoch)
   - Test accuracy, macro-F1, and a per-language precision/recall/F1 table

## Output

Running the pipeline produces, under `data/`:
- `train.jsonl`, `val.jsonl`, `test.jsonl` — the stratified splits
- `_raw_<lang>.jsonl` — cached raw sentences per language (used to resume interrupted collection)
