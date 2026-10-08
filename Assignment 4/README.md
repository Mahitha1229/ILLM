# N-gram Language Models with Smoothing on IndicCorp V2 (Telugu)

This repository contains my ILLM assignment on building **word-level n-gram language models (unigram to quadgram)** for **Telugu** (`tel_Telu`) and comparing **five smoothing methods** by perplexity. It reuses the sentence corpus and the dev/test splits from my previous assignment on BPE / WordPiece tokenization.

The whole pipeline lives in one notebook, `assignment-4-20-8-illm.ipynb`, and was run on **Kaggle**. The first half of the notebook (corpus + tokenizer experiments) is the previous assignment, and the second half ("PART 2") is the language-modelling work described here.

---

## What This Assignment Does

1. Builds the same **1,000,000-sentence** training set and the same **1,000-sentence dev / 1,000-sentence test** splits as the tokenization assignment (same seed, same length-stratified sampling).
2. Builds a closed vocabulary with `<unk>` handling.
3. Counts **1-, 2-, 3- and 4-grams** over the training set.
4. Implements five probability estimators from scratch (no LM libraries):
   - Unsmoothed (MLE)
   - Laplace (add-one)
   - Add-k
   - Linear interpolation (Jelinek-Mercer)
   - Interpolated Kneser-Ney
5. Tunes hyperparameters on the **dev** set and reports **perplexity** on dev and test for every (method x order) combination.

---

## Pipeline Overview

### 1. Data
- Source: `ai4bharat/IndicCorpV2`, Telugu split, streamed with Hugging Face `datasets`.
- Paragraphs are split into sentences (`indic-nlp-library`, regex fallback), de-duplicated and filtered to 3-100 words.
- Splits: **Train 1,000,000 | Dev 1,000 | Test 1,000**. Dev/test are length-stratified so their length distribution mirrors the full corpus (mean 10.5 words, median 9).
- Seeds are fixed (`42`), so the splits reproduce exactly if rebuilt.

### 2. Vocabulary
- Whitespace word tokenization.
- Words with train frequency **< 5** are mapped to `<unk>` (`MIN_COUNT = 5`). This keeps the vocabulary and quadgram memory under control for a morphologically rich language.
- Raw word types: **976,821**. Kept (count >= 5): **138,056**.
- Special tokens: `<s>` (context only, never a prediction target, so excluded from V), `</s>` and `<unk>` (both predictable).
- **Vocabulary size V = 138,058** (predictable tokens).

### 3. N-gram Counts
Each sentence is padded with `n-1` copies of `<s>` and one `</s>`.

| Order | Distinct n-grams | Distinct contexts |
|------:|-----------------:|------------------:|
| 1 (unigram) | 138,058 | 1 |
| 2 (bigram) | 4,109,630 | 138,058 |
| 3 (trigram) | 7,937,823 | 4,079,382 |
| 4 (quadgram) | 9,331,536 | 7,525,380 |

### 4. Smoothing Methods

| Method | Estimate |
|--------|----------|
| Unsmoothed (MLE) | `c(h, w) / c(h)` |
| Laplace | `(c(h, w) + 1) / (c(h) + V)` |
| Add-k | `(c(h, w) + k) / (c(h) + k*V)`, with `k` tuned per order on dev |
| Interpolation | Recursive Jelinek-Mercer: `lam_n * P_MLE + (1 - lam_n) * P_(n-1)`, with the unigram level interpolated with the uniform `1/V`. `lam` is tuned per order on dev, sequentially (order 1 first, then 2, 3, 4, holding lower orders fixed). |
| Kneser-Ney | Interpolated KN with fixed discount **D = 0.75**, recursing down to a continuation-probability unigram `N1+(. w) / N1+(. .)`. Unseen contexts back off fully to the lower order. |

### 5. Evaluation
- **Perplexity** = `2^(-average log2 P)` over every n-gram of the held-out set (including the `</s>` prediction).
- If any n-gram gets probability 0, perplexity is reported as `inf` and the fraction of zero-probability n-grams is saved to the CSV.

### Tuned Hyperparameters (on dev)

| Order | Best add-k `k` | Best interpolation `lambda` |
|------:|:--------------:|:---------------------------:|
| 1 | 0.01 | 0.95 |
| 2 | 0.01 | 0.65 |
| 3 | 0.01 | 0.10 |
| 4 | 0.01 | 0.05 |

---

## Results

### Test-Set Perplexity (lower is better)

| Method | N=1 | N=2 | N=3 | N=4 |
|--------|----:|----:|----:|----:|
| Unsmoothed | 3,468.15 | inf | inf | inf |
| Laplace | 3,473.29 | 8,814.37 | 42,843.21 | 66,618.75 |
| Add-k (k=0.01) | 3,468.19 | 1,570.88 | 15,144.20 | 41,129.56 |
| Interpolation | 3,504.71 | 795.33 | 763.92 | 778.25 |
| **Kneser-Ney** | 5,156.90 | **671.80** | **661.72** | 705.34 |

### Dev-Set Perplexity

| Method | N=1 | N=2 | N=3 | N=4 |
|--------|----:|----:|----:|----:|
| Unsmoothed | 3,796.44 | inf | inf | inf |
| Laplace | 3,799.76 | 9,382.76 | 44,087.77 | 67,105.53 |
| Add-k (k=0.01) | 3,796.46 | 1,653.40 | 15,020.85 | 40,299.74 |
| Interpolation | 3,828.20 | 846.23 | 801.09 | 813.04 |
| Kneser-Ney | 5,582.43 | 709.86 | 682.11 | 721.43 |

**Best model: interpolated Kneser-Ney trigram, test perplexity 661.72.**

---

## Key Observations

- **Unsmoothed models break at N >= 2.** MLE gives probability zero to any n-gram unseen in training, so perplexity is infinite for bigrams and above. This is the problem smoothing is meant to fix.
- **Laplace and Add-k get worse as N grows.** With V = 138k, add-one moves a huge share of probability mass onto unseen events, and this is much worse for higher orders where almost every context is sparse (Laplace goes from 8.8k at N=2 to 66.6k at N=4). Add-k with a small `k` is far better than Laplace but still degrades with order.
- **Interpolation fixes the sparsity problem.** Mixing in lower orders keeps perplexity in the hundreds. The tuned lambdas fall from 0.95 (unigram) to 0.05 (quadgram), so the model leans more and more on lower orders as higher-order contexts get sparser.
- **Kneser-Ney is the best smoothing method for N >= 2** (for example 671.80 vs 795.33 for interpolation at N=2). Its continuation probabilities handle words that are frequent but appear in only a few contexts.
- **Kneser-Ney is worse at N=1.** The pure continuation-count unigram (5,156.90) is a worse standalone unigram model than the frequency-based one (about 3,468), because it is designed to be a lower-order backoff distribution, not a final predictor.
- **Perplexity is best at the trigram and rises at the quadgram** for both Interpolation and Kneser-Ney. Telugu is morphologically rich (a large number of word types), so quadgram counts are very sparse and add noise instead of useful context.
- **Dev and test results agree.** The ranking of methods is the same on both splits, which suggests the hyperparameters tuned on dev did not overfit.

### Notes and Limitations

- The best add-k value (`k = 0.01`) is at the **lower edge of the search grid** (0.01 to 0.98), so an even smaller `k` might do better for Add-k.
- Perplexities are computed over a closed vocabulary with `<unk>`, so they are only directly comparable to models using the same vocabulary and the same `MIN_COUNT`.
- Quadgram counting over all 1M sentences is memory-heavy. If Kaggle runs out of RAM, lower `LM_TRAIN_SENTENCES` (for example to 300,000); everything else runs unchanged.

---


## Output Files

Saved to `/kaggle/working/indiccorp_experiment/`:

| File | Description |
|------|-------------|
| `dev.txt`, `test.txt`, `train_1000000.txt` | Splits used for the language models |
| `lm_perplexity_results.csv` | Perplexity for every (method x order x split), plus zero-probability n-gram fraction |
| `lm_perplexity_pivot_test.csv` | Test perplexity table (method x order) |
| `lm_perplexity_comparison.png` | Plot of test perplexity vs n-gram order for each method (log scale) |
| *(tokenization outputs)* | Tokenizer files, tokenized dev/test, CSVs and plots from the first half of the notebook |


