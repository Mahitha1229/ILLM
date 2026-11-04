# ILLM Assignments

This repository holds the notebooks for the course assignments on **Indian-language NLP**, from language identification and tokenisation through language models, embeddings, sequence labelling, machine translation and probing. Each section below is labelled by assignment and covers the task, data, method, how to run it, and the results from the saved run.

---

## Assignment 1: Language Identification from Scratch (TF-IDF + Logistic Regression)

**Notebook:** `Assignment-1_30_7.ipynb`

**Task:** Identify which of **23 Indian languages** a sentence is written in, with every component (sentence splitter, TF-IDF, classifier, metrics) implemented by hand.

**Languages:** Assamese, Bengali, Bodo, Dogri, Konkani, Gujarati, Hindi, Kannada, Kashmiri, Maithili, Malayalam, Marathi, Manipuri, Nepali, Odia, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu, Khasi, Santali.

**Data:** 1,000 sentences per language streamed from IndicCorp V2 (min 15 characters, min 3 words, de-duplicated), giving 23,000 sentences. Stratified split 80/10/10: **18,400 train / 2,300 val / 2,300 test**, saved as `data/train.jsonl`, `val.jsonl`, `test.jsonl`.

**Method**
- Rule-based sentence tokenizer that handles Indic and Urdu sentence-end marks (`। ॥ ۔ ؟ . ! ?`).
- Custom TF-IDF vectorizer over word 1-grams, word 2-grams and character 2/3/4-grams (up to 20,000 features, `min_df=2`).
- Custom softmax logistic regression (lr 0.5, 20 epochs, batch 64, L2 1e-4).
- Accuracy, per-class precision/recall/F1 and macro-F1 written from scratch.

**How to run:** Open in Colab, run all cells (it mounts Google Drive and works in `MyDrive/indic_langid`). Data collection streams from Hugging Face, so it takes a while on the first run; per-language raw files are cached.

**Results (test set):** **Accuracy 0.9483, macro-F1 0.9479.** Several languages reach F1 = 1.00 (Gujarati, Kannada, Malayalam, Odia, Santali, Tamil, Telugu). The weakest are **Konkani (0.708)** and **Marathi (0.714)**, which share the Devanagari script and are easily confused.

---

## Assignment 2: TF and IDF Weighting Variants

**Notebook:** `Assignment_2_6_8.ipynb`

**Task:** Reuse Assignment 1's dataset, features and classifier, but change only the TF and IDF weighting, and compare the six combinations.

**Weighting functions**
- TF: `UN` (raw counts), `NT1` (counts ÷ total terms in the sentence), `NT2` (augmented: 0.5 + 0.5 × count ÷ max count).
- IDF: `UN` (smoothed log((1+N)/(1+df))+1), `NI1` (log(max_df / df)).

**How to run:** Run in Colab with the Assignment 1 data in the same Drive folder (the first cells rebuild it if missing). The last cell trains all six models (20 epochs each).

**Results (test set)**

| TF | IDF | Accuracy | Macro-F1 |
|---|---|---|---|
| UN | UN | 0.9483 | 0.9479 |
| NT1 | UN | 0.9483 | 0.9479 |
| NT2 | UN | 0.9496 | 0.9492 |
| UN | NI1 | 0.9543 | 0.9540 |
| NT1 | NI1 | 0.9543 | 0.9540 |
| **NT2** | **NI1** | **0.9552** | **0.9551** |

The best combination is **NT2 + NI1**. NT1 gives exactly the same score as raw TF because it divides every term in a sentence by the same constant and the vectors are L2-normalised afterwards, which cancels the scaling.

---

## Assignment 3: Sentence Tokenisation and BPE / WordPiece (Telugu)

**Notebook:** `assignment-3-13-8-illm.ipynb`

**Task (language: Telugu, `tel_Telu`)**
1. Build a sentence corpus from IndicCorp V2 paragraphs (Indic NLP sentence splitter with a regex fallback; 3–100 words; de-duplicated).
2. Create length-stratified **dev (1,000)** and **test (1,000)** sets whose sentence-length distribution matches the full corpus, and a **1,050,000-sentence** training pool.
3. Train **BPE** and **WordPiece** tokenisers (Hugging Face `tokenizers`) and tokenise dev/test.
4. Study the effect of training size (100k / 300k / 500k / 1M sentences, vocab fixed at 30k) and vocabulary size (20k / 30k / 50k, training size fixed at 500k).

**How to run (Kaggle):** Turn Internet on, set `LANG_SPLIT` (and `ISO_CODE`) in Cell 1 for your language, and run all cells. A CPU is enough. Outputs go to `/kaggle/working/indiccorp_experiment` (splits, tokenizer JSON files, CSVs, plots).

**Results (test split, avg tokens per sentence)**

| Study | BPE | WordPiece |
|---|---|---|
| Train size 100k → 1M (vocab 30k) | 15.84 → 15.82 | 16.30 → 16.28 |
| Vocab 20k / 30k / 50k (train 500k) | 16.77 / 15.83 / 14.86 | 17.32 / 16.24 / 15.18 |

- Training size barely changes the tokenisation once there are about 100k sentences.
- A **larger vocabulary clearly reduces tokens per sentence** (more characters per token) for both algorithms.
- BPE produces slightly fewer tokens than WordPiece at the same settings; the UNK rate is 0% everywhere.

---

## Assignment 4: N-gram Language Models with Smoothing (Telugu)

**Notebook:** `assignment-4-20-8-illm.ipynb` (Part 1 = Assignment 3 code, **Part 2 = this assignment**)

**Task:** Build unigram to quadgram word language models on the same 1,000,000-sentence Telugu corpus and compare **5 estimation methods** by perplexity on the dev and test sets.

**Method**
- Words seen fewer than 5 times become `<unk>`; vocabulary size V = 138,058.
- Count tables for orders 1–4 (about 9.3M distinct 4-grams).
- Methods: **Unsmoothed (MLE), Laplace, Add-k, Interpolation (Jelinek–Mercer), Kneser–Ney** (discount D = 0.75).
- Add-k's `k` and the interpolation λ per order are tuned on the dev set.

**How to run (Kaggle):** Run Part 1 first (or keep its saved files), then Part 2. Quadgram counting over 1M sentences uses several GB of RAM; if the kernel runs out of memory, lower `LM_TRAIN_SENTENCES` (e.g. to 300,000).

**Results (test perplexity, lower is better)**

| Method | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|
| Unsmoothed | 3,468 | ∞ | ∞ | ∞ |
| Laplace | 3,473 | 8,814 | 42,843 | 66,619 |
| Add-k (k=0.01) | 3,468 | 1,571 | 15,144 | 41,130 |
| Interpolation | 3,505 | 795 | 764 | 778 |
| **Kneser–Ney** | 5,157 | **672** | **662** | 705 |

**Kneser–Ney** is best (trigram ≈ 662). Unsmoothed models hit unseen n-grams (infinite perplexity), and Laplace and Add-k get worse as the order grows because they spread too much probability onto unseen events.

---

## Assignment 5: Word2Vec, fastText, Doc2Vec, K-means and Sentence Retrieval (Telugu)

**Notebook:** `assignment-5-27-8-illm.ipynb` (Parts 1–2 = Assignments 3–4, **Part 3 = this assignment**)

**Task:** Train embeddings on the same 1M Telugu sentences, cluster the word vectors, and find the nearest training sentence for every dev/test sentence.

**Method**
- **Word2Vec** (CBOW) and **fastText** (gensim): 100-d, window 5, `min_count` 5, 5 epochs; vocabulary 138,056 words.
- **Doc2Vec** (PV-DM): 100-d, 10 epochs; 1,000,000 document vectors; dev/test vectors are inferred.
- **K-means** (k = 50) on the Word2Vec word vectors; the 20 words nearest each centroid are saved to `kmeans_clusters.json`.
- Sentence vectors: averaged Word2Vec, averaged fastText, and Doc2Vec. The nearest training sentence is found by cosine similarity, scanning the training set in batches.

**How to run (Kaggle):** Run after Parts 1–2 (or with the saved split files). The notebook installs `gensim` and `scikit-learn`. Use 4+ CPU workers.

**Results**
- Clusters are semantically coherent. For example, one holds cricket terms, one political parties, one body/health terms and one conflict/casualty words.
- Mean cosine similarity of the nearest match:

| Method | Dev | Test |
|---|---|---|
| Averaged fastText | 0.904 | 0.903 |
| Averaged Word2Vec | 0.895 | 0.895 |
| Doc2Vec | 0.730 | 0.737 |

- In the printed sample, the Word2Vec and fastText matches were topically related (COVID/virus), while the Doc2Vec match was unrelated. A high cosine score does not on its own prove a better match, so check the retrieved sentences too.

---

## Assignment 6: Language Identification with FastText Wiki Vectors

**Notebook:** `assignment-6-3-9-illm.ipynb` (titled "Assignment 2" inside the notebook)

**Task:** Re-do the 23-language identification from Assignment 1 with neural models on pretrained FastText wiki vectors, and compare five classifiers using **macro-F1 implemented from scratch**.

**Models:** (1) Logistic Regression and (2) MLP with 0 / 1 / 2 hidden layers on averaged word vectors; (3) LSTM and (4) GRU on word-vector sequences (max 40 tokens); (5) FastText supervised classifier.

**How to run:** Kaggle with Internet on (GPU optional). Point `DATA_DIR` to the Assignment 1 `train/val/test.jsonl`; if they are missing, the notebook regenerates them from IndicCorp V2.

**Results (test macro-F1)**

| Model | Macro-F1 |
|---|---|
| **FastText supervised** | **0.8977** |
| GRU | 0.6244 |
| LSTM | 0.6106 |
| MLP (1 hidden layer) | 0.5370 |
| MLP (2 hidden layers) | 0.5022 |
| Logistic Regression | 0.4436 |
| MLP (0 hidden layers) | 0.4407 |

---

## Assignment 7: HMM Noun Phrase Chunking

**Notebook:** `assignment-7-10-9-illm.ipynb` ("Lab 7")

**Task:** Tag words with **B-NP / I-NP / O** using a Hidden Markov Model.

**Data:** 823 training and 77 test sentences (one token per line, blank line between sentences); 3 tags, vocabulary 4,570 words plus an UNK index. The paths in the notebook point to Kaggle datasets (`train_NP.txt`, `test_NP.txt`); change them to your files.

**Method**
- Count start, transition and emission probabilities, with **Add-K smoothing**.
- **Forward algorithm** for sentence probability (log-space, `logsumexp`).
- **Viterbi decoding** for the best tag sequence.
- Chunk-level precision, recall and F1 (the `seqeval` install failed, so the notebook uses a built-in equivalent).

**Results (test set, chunk level)**

| K | Precision | Recall | F1 |
|---|---|---|---|
| 1 | 0.6774 | 0.6632 | 0.6702 |
| **0.1** | 0.6716 | 0.7579 | **0.7122** |
| 0.01 | 0.6381 | 0.7200 | 0.6766 |
| 0.001 | 0.6292 | 0.7074 | 0.6660 |

The best **K = 0.1** gives F1 **0.7122** (475 gold chunks). Per-sentence log-probabilities from the forward algorithm are saved to `test_sentence_probs.csv`.

---

## Assignment 8: Lab 8, English → Hindi NMT with OpenNMT-py (Baseline)

**Notebook:** `08_lab_illm.ipynb`

**Task:** Train a small Transformer NMT model on the ILCI English–Hindi subset of BPCC and evaluate it overall and by sentence length.

**Data:** 165,649 sentence pairs, split into 4 length bins by English word count (bin1 ≤10, bin2 11–20, bin3 21–29, bin4 ≥30). Per bin: up to 5,000 train / 100 valid / 500 test, giving 20,000 train and 2,000 test sentences.

**Method:** SentencePiece unigram (8,000 vocab, separate source/target models) → OpenNMT-py Transformer (3 layers, 4 heads, hidden 256, FF 1024, 5,000 steps, CPU) → beam search (size 5) → sacreBLEU and chrF.

**How to run:** Needs Python 3.11 (tested on an Apple-silicon Mac with a conda env) and a Hugging Face token (`login()`). Run all cells in order. Training took about 35 minutes on CPU.

**Results (test set)**

| Bin | Avg src length | BLEU | chrF |
|---|---|---|---|
| bin1 | 7.4 | 1.36 | 9.24 |
| bin2 | 15.2 | 0.78 | 8.84 |
| bin3 | 24.3 | 0.41 | 8.09 |
| bin4 | 37.0 | 0.67 | 6.99 |
| **Overall** | – | **0.70** | **7.88** |

Final validation perplexity 285.7 and accuracy 20.5%. The scores are very low because the model is tiny and trained for only 5,000 steps; some outputs are generic or repeat phrases. The analysis cell at the end is a template to fill in.

---

## Assignment 9: NMT with Pretrained Embeddings (FastText vs BPEmb)

**Notebook:** `assignment-9-8-10-illm.ipynb`

**Task:** Repeat Assignment 8 with the same data and splits, initialising the Transformer's embeddings from **FastText** (word-level wiki vectors) or **BPEmb** (subword vectors), and compare the two.

**Method:** Same ILCI English → Hindi data and length bins. FastText: lower-cased word tokens, top 200k vectors, 30k vocab. BPEmb: SentencePiece subwords, vocab 10,000. Both use 300-d embeddings, a 3-layer / 6-head Transformer, 5,000 steps, beam 5. Evaluation uses sacreBLEU/chrF after the same normalisation for both systems.

**How to run:** Kaggle/Colab or local, with a Hugging Face token. The first cell builds a separate Python 3.11 environment for OpenNMT-py if your Python is newer. FastText about 14 min, BPEmb about 7 min (CPU).

**Results (test set, 2,000 sentences)**

| Bin | FastText BLEU | FastText chrF | BPEmb BLEU | BPEmb chrF |
|---|---|---|---|---|
| bin1 | 13.80 | 26.52 | 12.13 | 25.79 |
| bin2 | 9.93 | 25.04 | 9.27 | 23.39 |
| bin3 | 7.55 | 23.99 | 7.20 | 22.41 |
| bin4 | 8.83 | 25.20 | 6.59 | 21.61 |
| **Overall** | **9.11** | **24.94** | **7.75** | **22.56** |

FastText scored higher overall; BPEmb has near-complete embedding coverage (about 99.5% vs 69–77%) and almost no OOV tokens, but its BLEU falls steadily as sentences get longer. Assignment 9's BLEU uses normalised text and a different tokeniser setting than Assignment 8, so the two BLEU numbers are **not directly comparable**.

---

## Assignment 10: Indian Language Identification (Classical vs Deep vs MuRIL)

**Notebook:** `assignment-10-15-10.ipynb`

**Task:** Classify sentences into 6 languages (Hindi, Gujarati, Bengali, Marathi, Tamil, Telugu) with six models: Logistic Regression and MLP on character TF-IDF, a FastText-style classifier (pure PyTorch), character-level BiLSTM and BiGRU, and a fine-tuned **MuRIL** (`google/muril-base-cased`).

**Data:** Synthetic: 1,200 sentences per language (7,200 total) generated from 28-word banks and 7 templates per language; 80/20 stratified split (5,760 train / 1,440 test).

**How to run (Kaggle):** Internet on, GPU recommended (needed for MuRIL). Run all cells. Change the `/kaggle/working/` paths if running locally.

**Results (test set)**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression, MLP, BiLSTM, BiGRU, MuRIL | 1.0000 | 1.0000 |
| FastText-style | 0.9986 | 0.9986 |


---

## Assignment 11: Layer-wise Probing of BERT (SentEval)

**Notebook:** `assignment-11-22-10.ipynb`

**Task:** Probe what each of the 12 layers of frozen `bert-base-uncased` encodes. The `[CLS]` vector from every layer trains a separate Logistic Regression probe (36 probes in total).

**Tasks (SentEval):** Sentence Length (6 classes), Bigram Shift (2 classes, word-order sensitivity), Subject Number (2 classes, singular vs plural subject).

**Method:** Balanced sample of up to 50,000 sentences per task, class-balanced 90/10 train/test split (about 45,000 / 5,000), `[CLS]` features per layer (max length 128), `StandardScaler` + Logistic Regression (C = 1.0).

**How to run (Kaggle):** Internet on, GPU recommended. Run all cells; datasets and the BERT checkpoint are downloaded automatically (several GB of disk for the layer features).

**Results (test accuracy per layer):** Sentence Length **0.926 on all 12 layers**; Bigram Shift about **0.49** and Subject Number about **0.51** on every layer, i.e. chance level.
