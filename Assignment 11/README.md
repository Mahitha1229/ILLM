# Layer-wise Probing of BERT with SentEval Tasks

A Jupyter notebook (`assignment-11-22-10.ipynb`) that probes what each of the **12 encoder layers of `bert-base-uncased`** encodes about a sentence. BERT is kept **frozen**. The `[CLS]` vector from each layer is extracted and used to train a separate **Logistic Regression probe**, giving 12 probes per task and 36 probes in total.

## Probing Tasks (from [SentEval](https://github.com/facebookresearch/SentEval))

| Task | What it tests | Classes |
|---|---|---|
| **Sentence Length** | Surface information: roughly how long is the sentence? | 6 length bins (`0`–`5`) |
| **Bigram Shift** | Syntax / word order: were two adjacent words swapped? | `O` (original), `I` (inverted) |
| **Subject Number** | Morphosyntax: is the subject of the main clause singular or plural? | `NN`, `NNS` |

## Method

1. **Download** the three SentEval probing files from GitHub and the `bert-base-uncased` checkpoint from Hugging Face (automatic, in Cell 1).
2. **Parse** the tab-separated files (partition, label, sentence).
3. **Balanced sample** of up to 50,000 sentences per task (equal examples per class). Sentence Length gives 49,998 because 50,000 isn't divisible by 6.
4. **90 / 10 train/test split**, balanced per class (45,000 train; about 5,000 test).
5. **Extract** the `[CLS]` vector (768-d) from each of the 12 layers (max length 128, batch size 16) and store them as memory-mapped arrays.
6. **Train 12 probes per task:** `StandardScaler` + `LogisticRegression` (C=1.0, `max_iter=1000`, `lbfgs`), fitted on the training set only.
7. **Evaluate** accuracy and macro-F1 on the test set, then plot results per layer.

## Requirements

- **Python 3.9+**
- A **GPU is strongly recommended** (the saved run used a Tesla T4; extraction took about 1.6 min per task on GPU). The notebook warns that CPU extraction will be very slow.
- **Internet access** for the SentEval files (`raw.githubusercontent.com`) and the BERT checkpoint (Hugging Face).
- About **several GB of free disk**: the 12 layer-wise arrays are saved to disk per task, plus the BERT weights.
- Packages (installed by Cell 1): `transformers`, `accelerate`, `huggingface_hub`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`. `torch` must also be available (preinstalled on Kaggle/Colab).

## How to Run

### Option A: Kaggle (what the notebook was run on)

1. Upload `assignment-11-22-10.ipynb` to Kaggle and open it.
2. Turn **Internet** on and set the accelerator to a **GPU**.
3. Click **Run All**. No manual dataset or model upload is needed.

### Option B: Locally or on Colab

1. Install PyTorch and the packages above.
2. Change `WORK_DIR = "/kaggle/working"` in Cell 1 to a folder on your machine (e.g. `"./work"`). The same folder is used for the output CSVs in Cells 8 and 10, so check it there as well.
3. Run all cells in order.

**Approximate time (T4 GPU):** extraction about 2 min per task. The Sentence Length probes take about 35–60 s each to fit, so expect roughly 10 min for that task and less for the other two.

## Notebook Walkthrough

| Cell | What it does |
|---|---|
| 1 | Installs packages, sets the seed (42), downloads the 3 SentEval files and the BERT checkpoint |
| 2 | Imports and checks that all files are present |
| 3 | Parses the SentEval tab-separated format |
| 4 | Draws the balanced sample (up to 50,000 per task) |
| 5 | Creates class-balanced 90/10 train/test splits and saves them as CSVs |
| 6 | Loads frozen `bert-base-uncased` (12 layers, hidden size 768) |
| 7 | Defines `extract_layer_representations`, which saves the `[CLS]` vector of each layer |
| 8 | Extracts features and trains the 12 Logistic Regression probes for each task |
| 9 | Plots accuracy and macro-F1 per layer for each task and shows the best layer |
| 10 | Saves the detailed per-layer classification reports and an accuracy summary |

## Results (from the saved run)

Test accuracy per layer (macro-F1 is almost identical):

| Layer | Sentence Length | Bigram Shift | Subject Number |
|---|---|---|---|
| 1 | 0.9260 | 0.4900 | 0.5062 |
| 2 | 0.9260 | 0.4900 | 0.5124 |
| 3 | 0.9260 | 0.4886 | 0.5124 |
| 4 | 0.9260 | 0.4900 | 0.5132 |
| 5 | 0.9260 | 0.4918 | 0.5132 |
| 6 | 0.9260 | 0.4900 | 0.5132 |
| 7 | 0.9260 | 0.4918 | 0.5062 |
| 8 | 0.9260 | 0.4900 | 0.5062 |
| 9 | 0.9260 | 0.4900 | 0.5132 |
| 10 | 0.9260 | 0.4900 | 0.5054 |
| 11 | 0.9260 | 0.4918 | 0.5132 |
| 12 | 0.9260 | 0.4900 | 0.5054 |

## ⚠️ Please Check These Results Before Submitting

The saved numbers do not look like a normal BERT probing result, so I would not report them as findings without checking them first:

- **Sentence Length** has the *same* accuracy (0.92597) for all 12 layers, to five decimal places. Different layers produce different vectors, so identical scores are very unlikely.
- **Bigram Shift and Subject Number** stay at about **49–51%**, which is **chance level** for two balanced classes, on every layer. BERT's `[CLS]` vectors normally score well above chance on these tasks, especially in the middle layers.

This pattern usually points to a pipeline problem rather than a real property of BERT. Things worth checking:

- Whether the per-layer vectors really differ (compare a few rows from different `*_layer_N.mmap` files).
- Whether each sentence is paired with the correct feature row (feature order vs the `train_df` / `test_df` order and labels).
- Whether `StandardScaler` + `LogisticRegression` is converging (`max_iter=1000` may not be enough; look for convergence warnings).
- A quick sanity check: train a probe on a single layer's saved features and a shuffled-label control. Chance-level on shuffled labels with clearly higher accuracy on the real labels would confirm the pipeline works.

Also note that the best layer in Cell 9 is chosen using the **test** macro-F1. The notebook itself says a validation set should be used for stricter model selection.

## Outputs

Saved to the working folder (`/kaggle/working` by default):

| File | Description |
|---|---|
| `senteval_probing/*_balanced_sample.csv`, `*_train.csv`, `*_test.csv` | Sampled data and splits for each task |
| `layerwise_bert_probing_results.csv` | Accuracy, macro-F1 and fit time for all 36 probes |
| `layerwise_classification_reports.csv` | Per-class precision, recall and F1 for every layer and task |
| `layerwise_accuracy_summary.csv` | Layer × task accuracy table |

## Troubleshooting

| Problem | Fix |
|---|---|
| Download errors | Enable internet access. The notebook sets `HF_HUB_DISABLE_XET=1` to avoid a Hugging Face endpoint issue on Kaggle. |
| CUDA out of memory | Lower `BATCH_SIZE` (Cell 7) to 8 or 4. |
| Very slow extraction | Use a GPU. |
| Disk space errors | Delete the `*_layer_N.mmap` files after probing, or write to a larger folder. |
| `/kaggle/working` not found | Change `WORK_DIR` (see "How to Run"). |

## Tech Stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · NumPy · Matplotlib
