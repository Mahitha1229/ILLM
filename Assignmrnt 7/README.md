# HMM Noun Phrase Chunking (B-NP / I-NP / O) with Forward Algorithm and Viterbi Decoding

This repository contains my ILLM Lab 7 assignment: a **Hidden Markov Model for noun phrase (NP) chunking**, implemented from scratch with NumPy. Each word in a sentence is tagged **B-NP** (begins an NP), **I-NP** (inside an NP) or **O** (outside), and chunks are scored at the phrase level.

The lab covers two tasks plus evaluation:

| Task | What is implemented |
|------|---------------------|
| Task 1 | **Forward algorithm**: the probability of a sentence, `P(words)`, summed over all tag sequences |
| Task 2 | **Viterbi decoding**: the most likely tag sequence for a sentence |
| Evaluation | Add-K smoothing for several values of K, scored with chunk-level precision / recall / F1 (seqeval-style) |

Everything lives in one notebook, `assignment-7-10-9-illm.ipynb`, and was run on **Kaggle**.

---

## Dataset

CoNLL-style noun-phrase chunking data, one token per line (first column = word, last column = tag), with a blank line between sentences. Any columns in between are ignored.

| Split | File | Sentences |
|-------|------|----------:|
| Train | `train_NP.txt` | 823 |
| Test | `test_NP.txt` | 77 |

- Tags: `B-NP`, `I-NP`, `O`
- Training vocabulary: **4,570 distinct words**, plus one reserved `UNK` index for unseen words (so V = 4,571).
- Test set: **475 gold NP chunks**.

---

## How the HMM Works

**Training (counting).** From the training sentences I count:
- start counts: how often each tag begins a sentence
- transition counts: tag -> next tag
- emission counts: tag -> word

**Add-K smoothing.** Each distribution is smoothed by adding `K` to every count and renormalising:
- `pi(t) = (c_start(t) + K) / (sum c_start + K*T)`
- `A(t', t) = (c(t' -> t) + K) / (c(t') + K*T)`
- `B(t, w) = (c(t, w) + K) / (c(t) + K*V)`

with `T = 3` tags. Words not seen in training all map to the shared `UNK` word, which has zero count under every tag, so its emission probability comes entirely from smoothing.

**Forward algorithm (Task 1).** Computed in log space with `scipy.special.logsumexp` to avoid underflow:
`alpha_1 = log pi + log B(., o_1)`, then `alpha_t = logsumexp_over_prev(alpha_(t-1) + log A) + log B(., o_t)`. The sentence log-probability is `logsumexp(alpha_n)`.

**Viterbi decoding (Task 2).** Dynamic programming in log space: for each position and tag, keep the best score `delta` and a backpointer, then trace back from the best final state.

**Evaluation.** Chunk-level precision, recall and F1 for the NP chunk type (a chunk counts as correct only if both its boundaries match).

---

## Results

### Effect of the Smoothing Constant K (Test Set, Chunk-Level)

| K | Precision | Recall | F1 |
|--:|----------:|-------:|---:|
| 1 | 0.6774 | 0.6632 | 0.6702 |
| **0.1** | 0.6716 | **0.7579** | **0.7122** |
| 0.01 | 0.6381 | 0.7200 | 0.6766 |
| 0.001 | 0.6292 | 0.7074 | 0.6660 |

**Best K = 0.1**, with chunk-level F1 = **0.7122** (precision 0.6716, recall 0.7579, support 475).

### Sentence Probabilities (Forward Algorithm, K = 0.1)

The forward algorithm gives `log P(sentence)` for every test sentence. For the first test sentences:

| Length (words) | log P | P |
|---------------:|------:|--:|
| 37 | -271.51 | 1.22e-118 |
| 27 | -197.99 | 1.03e-86 |
| 29 | -221.50 | 6.39e-97 |
| 36 | -242.70 | 3.96e-106 |
| 31 | -229.19 | 2.91e-100 |

All sentence probabilities are extremely small and get smaller as sentences get longer, which is why log probabilities are used (the raw probability can underflow to 0 for long sentences). All values are saved to `test_sentence_probs.csv`.

### Example Prediction (test sentence #1, K = 0.1)

```
Words: Confidence in the pound is widely expected to take another sharp dive ...
True : B-NP  O  B-NP I-NP O  O      O        O  O    B-NP    I-NP  I-NP ...
Pred : B-NP  O  B-NP I-NP O  B-NP   O        O  O    B-NP    I-NP  I-NP ...
```

The model gets the main noun phrases right but wrongly tags the adverb `widely` as the start of an NP.

---

## Key Observations

- **K = 0.1 is the best smoothing value (F1 0.712).** Both a large K (1) and very small K values (0.01, 0.001) are worse. With K = 1 too much probability mass is moved to unseen events, and with very small K the model trusts the sparse training counts too much (see the next point).
- **Smaller K raises recall but lowers precision.** From K = 1 to K = 0.001, precision falls from 0.677 to 0.629 while recall peaks at K = 0.1 (0.758). Because the training set is small (823 sentences), many test words are unseen, and how much probability `UNK` gets under each tag depends directly on K.
- **Errors are mostly at chunk boundaries and on unseen or ambiguous words.** Typical mistakes: tagging an adverb or verb as part of an NP (`widely`, `reckon`), or merging two adjacent NPs / missing a split between them (`July and August 's`, `Mansion House speech last Thursday`).
- **This is a simple model.** The emission model only sees the (case-sensitive) word identity and the transition model only sees the previous tag, so it has no features such as capitalisation, suffixes or part-of-speech information. This limits it to roughly the 0.67-0.71 F1 range here.


---



## Output Files

| File | Description |
|------|-------------|
| `test_sentence_probs.csv` | For each test sentence: the sentence text, its length, `log_prob` and `prob` from the forward algorithm (K = 0.1) |

