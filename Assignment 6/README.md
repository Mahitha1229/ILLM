
# Language Identification for 23 Indian Languages: LR, MLP, LSTM, GRU and FastText

This repository contains my ILLM assignment on **language identification** over **23 Indian languages** (IndicCorp V2). I compare five classifiers and evaluate them with a **macro-F1 score implemented from scratch** (no `sklearn.metrics`):

1. Logistic Regression on averaged word vectors
2. MLP with 0 / 1 / 2 hidden layers
3. LSTM
4. GRU
5. FastText supervised classifier

The whole pipeline lives in one notebook, `assignment-6-3-9-illm.ipynb`, and was run on **Kaggle**. The dataset is the 23-language sentence dataset from my earlier language-identification assignment (the notebook regenerates it from IndicCorp V2 if the files are not found).

---

## IMPORTANT: Note on the Word Vectors Used

The notebook tries to stream pretrained **FastText Wikipedia vectors** (`wiki.{code}.vec.gz`, 300-d) for each language. **In my run, none of the 23 downloads succeeded** (the loader reported "NOT AVAILABLE" for every language, probably because the file URL pattern returned a non-200 response). As the notebook's designed fallback, every word therefore got a **fixed, randomly generated 300-d vector** (one per word, seeded from the word itself).

What this means for the results:

- **Logistic Regression, MLP, LSTM and GRU ran on random word vectors, not pretrained FastText vectors.** Each word still has its own unique vector, so the models can learn word identity, but the vectors carry no pretrained semantic or cross-lingual information.
- **The FastText supervised classifier does not use pretrained vectors at all.** It learns its own embeddings from the labelled sentences, so it is unaffected.
- The comparison below is a comparison of these models under that setup. Numbers for the first four methods would likely change with real pretrained vectors.

---

## Dataset

- Source: `ai4bharat/IndicCorpV2`, streamed with Hugging Face `datasets`.
- **23 languages**, 1,000 unique sentences each (23,000 total). Sentences are at least 15 characters and 3 words long.
- **Stratified 80 / 10 / 10 split** per language: **18,400 train | 2,300 validation | 2,300 test** (100 test sentences per language).
- Seeds are fixed (`42`).

Languages (dataset labels):
`asm_Beng, ben_Beng, brx_Deva, doi_Deva, gom_Deva, guj_Gujr, hin_Deva, kan_Knda, kas_Arab, khasi, mai_Deva, mal_Mlym, mar_Deva, mni_Mtei, npi_Deva, ory_Orya, pan_Guru, san_Deva, santhali, snd_Deva, tam_Taml, tel_Telu, urd_Arab`

---

## Methods

### Evaluation Metric: Macro-F1 From Scratch
Per-class TP / FP / FN are counted from the predictions, per-class precision, recall and F1 are computed, and macro-F1 is the **unweighted mean of the 23 per-class F1 scores**.

### Input Features
- **Tokenization:** lowercase, whitespace split.
- **Averaged vectors** (Methods 1-2): the mean of the sentence's 300-d word vectors.
- **Sequences** (Methods 3-4): the sentence kept as a sequence of word vectors, truncated to 40 tokens and padded per batch.

### Models

| # | Method | Details |
|---|--------|---------|
| 1 | Logistic Regression | One linear layer + softmax, trained with a manual PyTorch loop, 15 epochs |
| 2 | MLP | 0 / 1 / 2 hidden layers (128 units, ReLU, dropout 0.2), 15 epochs. The 0-hidden-layer version is the same linear model as Method 1 (sanity check). |
| 3 | LSTM | 1 layer, hidden size 128, last hidden state -> linear layer, 8 epochs |
| 4 | GRU | Same as the LSTM with a GRU cell, 8 epochs |
| 5 | FastText supervised | Official `fasttext` library: `lr=1.0`, `epoch=25`, `wordNgrams=2`, `dim=300`, `loss="softmax"` |

All PyTorch models use Adam with `lr=1e-3` and cross-entropy loss (batch size 128 for LR/MLP, 64 for LSTM/GRU).

---

## Results (Test Set, Macro-F1)

| Rank | Model | Macro-F1 |
|-----:|-------|---------:|
| 1 | **FastText supervised** | **0.8977** |
| 2 | GRU | 0.6244 |
| 3 | LSTM | 0.6106 |
| 4 | MLP (1 hidden layer) | 0.5370 |
| 5 | MLP (2 hidden layers) | 0.5022 |
| 6 | Logistic Regression | 0.4436 |
| 7 | MLP (0 hidden layers) | 0.4407 |

For reference, random guessing over 23 classes would score about 0.04.

---

## Key Observations

- **FastText supervised is by far the best (0.898).** It learns its own word and bigram features directly from the labelled sentences, which suits language identification very well. It also does not depend on the (failed) pretrained vector download.
- **Sequence models beat averaged-vector models.** GRU (0.624) and LSTM (0.611) are well ahead of the MLPs and Logistic Regression. With random word vectors, averaging many vectors blurs the identity of the individual words, while the recurrent models keep each token and its order. This gap is probably larger than it would be with real pretrained vectors.
- **GRU vs LSTM:** the GRU is slightly ahead (+0.014), which is a small difference from a single run and may be within run-to-run noise.
- **Sanity check passes:** the 0-hidden-layer MLP (0.441) matches Logistic Regression (0.444), as expected since they are the same linear model.
- **One hidden layer helps, two do not:** the MLP improves from 0.441 to 0.537 with one hidden layer but drops to 0.502 with two. The extra depth did not add useful capacity on top of the averaged vectors.
- **Several models were still improving when training stopped.** Validation accuracy was still rising at the last epoch for Logistic Regression, the 1-hidden-layer MLP, the LSTM and the GRU, so they would likely gain from more epochs.

---

## Output Files

| File | Description |
|------|-------------|
| `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` | The 23-language dataset splits (`text`, `label`) |
| `ft_train.txt`, `ft_val.txt`, `ft_test.txt` | The same splits in FastText's `__label__` format |
| Comparison table and bar chart | Macro-F1 of all models, shown in the notebook |

