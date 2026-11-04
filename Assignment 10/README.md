# Indian Language Identification: Classical ML vs Deep Learning vs MuRIL

A Jupyter notebook (`assignment-10-15-10.ipynb`) that identifies which of **6 Indian languages** a sentence is written in. It builds a synthetic labelled dataset from scratch, then trains and compares six different models on it.

**Languages:** Hindi, Gujarati, Bengali, Marathi, Tamil, Telugu

## Models Compared

| # | Model | Input features |
|---|---|---|
| 1 | Logistic Regression | Character n-gram TF-IDF (1–5) |
| 2 | MLP (256-128, ReLU) | Character n-gram TF-IDF (1–5) |
| 3 | FastText-style classifier | Word + character-subword n-gram embeddings, averaged (pure PyTorch `EmbeddingBag`) |
| 4 | BiLSTM | Character-level embeddings |
| 5 | BiGRU | Character-level embeddings |
| 6 | MuRIL (`google/muril-base-cased`) | Fine-tuned BERT-family model for Indian languages |

The FastText model is re-implemented directly in PyTorch instead of using the `fasttext` package, because that package can fail to build on newer Kaggle Python versions.

## Dataset

No external dataset is needed. The notebook generates it:

- Each language has a **word bank (28 words)** and **7 sentence templates** (e.g. `{a} {b} {c} है।`).
- Random words are slotted into random templates to create sentences.
- **1,200 sentences per language, 7,200 in total**, shuffled with seed 42.
- Split **80 / 20** (stratified): **5,760 train / 1,440 test** (240 test sentences per language).
- The generated dataset is saved as `indian_language_identification_dataset.csv`.

## Requirements

- **Python 3.9+**
- A **GPU is recommended** (the saved run used a Tesla T4) but not required for the first five models. Fine-tuning MuRIL on CPU will be slow.
- **Internet access** to download the MuRIL checkpoint from Hugging Face.
- Libraries (the first cell installs `transformers` and `accelerate`; the rest are standard):

```bash
pip install transformers accelerate torch scikit-learn pandas numpy
```

## How to Run

### Option A: Kaggle (what the notebook was run on)

1. Upload `assignment-10-15-10.ipynb` to Kaggle and open it.
2. Turn **Internet** on, and set the accelerator to a **GPU** (e.g. T4).
3. Click **Run All**.

### Option B: Locally or on Colab

1. Install the libraries above and open the notebook in Jupyter.
2. **Change the two `/kaggle/working/` paths** (Cell 3 and Cell 9) to a folder that exists on your machine, e.g. `"./"`, otherwise saving the CSV files will fail.
3. Run all cells from top to bottom.

## Notebook Walkthrough

| Cell | What it does |
|---|---|
| 1 | Installs `transformers` and `accelerate` |
| 2 | Imports, sets seed 42, detects GPU |
| 3 | Generates the synthetic 6-language dataset and saves it as CSV |
| 4 | Label encoding, train/test split, and the `evaluate_predictions` helper (accuracy, macro-F1, per-class report) |
| 5 | Logistic Regression and MLP on character TF-IDF |
| 6 | FastText-style classifier (word + 3–5 char n-gram features, 100-d embeddings, 8 epochs) |
| 7 | Character vocabulary, then BiLSTM and BiGRU (64-d embeddings, 128 hidden units, 8 epochs) |
| 8 | Fine-tunes MuRIL (2 epochs, batch size 16, max length 128, lr 2e-5) |
| 9 | Builds the comparison table and saves it to `model_comparison_results.csv` |
| 10 | `identify_language(text)` demo using the fine-tuned MuRIL model |

## Key Settings

| Setting | Value |
|---|---|
| Seed | 42 |
| Test split | 20% (stratified) |
| TF-IDF | char n-grams 1–5, `min_df=2`, up to 50,000 features |
| FastText-style | n-grams 3–5, min feature frequency 2, embedding dim 100 |
| BiLSTM / BiGRU | emb 64, hidden 128, dropout 0.3, max 160 characters |
| MuRIL | `google/muril-base-cased`, 2 epochs, lr 2e-5 |

## Results (from the saved run)

Test set: 1,440 sentences.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (char TF-IDF) | 1.0000 | 1.0000 |
| MLP (char TF-IDF) | 1.0000 | 1.0000 |
| BiLSTM (character-level) | 1.0000 | 1.0000 |
| BiGRU (character-level) | 1.0000 | 1.0000 |
| MuRIL fine-tuned | 1.0000 | 1.0000 |
| FastText-style (word + subword n-grams) | 0.9986 | 0.9986 |

**Demo predictions (MuRIL):**

| Input | Predicted | Confidence |
|---|---|---|
| `આજે હું શાળાએ જાઉં છું.` | Gujarati | 0.46 |
| `आज मौसम अच्छा है।` | Hindi | 0.46 |
| `আমি বাংলা ভাষা শিখছি।` | Bengali | 0.45 |

## Important Notes on the Results

- **The near-perfect scores are expected and should not be read as real-world accuracy.** The data is synthetic: every sentence comes from only 7 templates and 28 words per language, and the train and test sets share the same templates and vocabulary. Most of the languages also use different scripts, which makes them easy to separate.
- **Hindi vs Marathi** share the Devanagari script, so they are the only genuinely similar pair. Even so, the models separate them because of language-specific function words in the templates (e.g. `है` vs `आहे`).
- **MuRIL's confidence is low (about 0.45)** even though it is correct, because it was fine-tuned for only 2 epochs and the loss was still falling (0.93 at epoch 2). More epochs would raise its confidence.
- To get a more realistic comparison, evaluate on a real dataset (e.g. news or Wikipedia sentences) or on templates and words held out from training.

## Outputs

| File | Description |
|---|---|
| `indian_language_identification_dataset.csv` | The generated 7,200-sentence dataset |
| `model_comparison_results.csv` | Accuracy and macro-F1 for every model |

## Troubleshooting

| Problem | Fix |
|---|---|
| `OSError` / path error saving CSVs | Replace `/kaggle/working/` with a local folder (see "How to Run"). |
| MuRIL download fails | Enable internet access; check that `google/muril-base-cased` is reachable. |
| CUDA out of memory | Lower `BERT_BATCH_SIZE` (Cell 8) or `MAX_LENGTH`. |
| MuRIL is very slow | Use a GPU, or skip Cell 8 and Cell 10 (the other five models run fine on CPU). |
| Warnings about `UNEXPECTED` / `MISSING` keys when loading MuRIL | Normal. The pretraining head is unused and the new classifier layer starts untrained. |

## Tech Stack

Python · PyTorch · scikit-learn · Hugging Face Transformers · pandas · NumPy
