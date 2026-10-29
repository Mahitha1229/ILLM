
# English → Hindi Neural Machine Translation with OpenNMT-py

This repository contains my ILLM Lab 8 assignment: a small Transformer model for English → Hindi machine translation, trained with OpenNMT-py on CPU and evaluated with sacreBLEU, both overall and by sentence-length bin.

The lab covers the full NMT pipeline:

| Step | What is implemented |
|---|---|
| Data | Load the ILCI subset of `ai4bharat/BPCC` and build length-binned train / valid / test splits |
| Preprocessing | Separate SentencePiece (unigram) models for English and Hindi |
| Training | Transformer encoder-decoder trained with OpenNMT-py (CPU only) |
| Translation | Beam search decoding of the test set |
| Evaluation | sacreBLEU and chrF, overall and per length bin |

Everything lives in one notebook, `08_lab_illm.ipynb`.

## Dataset

The ILCI subset of [AI4Bharat BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) (English–Hindi, `hin_Deva`), loaded from Hugging Face.

- Sentence pairs after cleaning and de-duplication: **165,649**
- Length = number of whitespace-separated words in the English sentence
- Each bin is split separately, and the validation set is taken from non-test data so the test set stays untouched (seed = 42)

| Bin | Source length (words) | Train | Valid | Test |
|---|---|---|---|---|
| bin1 | ≤ 10 | 5,000 | 100 | 500 |
| bin2 | 11–20 | 5,000 | 100 | 500 |
| bin3 | 21–29 | 5,000 | 100 | 500 |
| bin4 | ≥ 30 | 5,000 | 100 | 500 |

Total: 20,000 training pairs (capped from the full data to keep CPU training feasible), 400 validation pairs, 2,000 test pairs.

## How It Works

**Tokenization.** Two SentencePiece unigram models (one for English, one for Hindi), each with an 8,000-token vocabulary, trained on the training split only.

**Model.** A Transformer encoder-decoder with 3 encoder and 3 decoder layers, 4 attention heads, hidden size 256 and feed-forward size 1024, dropout 0.1.

**Training.** Adam (β1 = 0.9, β2 = 0.998) with Noam learning-rate decay and 1,000 warmup steps, label smoothing 0.1, token-based batches of 2,048. Trained for 5,000 steps on CPU, with a checkpoint every 1,000 steps. The final checkpoint (step 5,000) is used for translation.

**Decoding.** Beam search with beam size 5 and max length 200. The SentencePiece output is decoded back to Hindi text.

**Evaluation.** Corpus-level BLEU and chrF from sacreBLEU, computed on the full test set and separately for each length bin.

## Results

### BLEU and chrF by Sentence Length (Test Set)

| Bin | Test sentences | Avg. source length | BLEU | chrF |
|---|---|---|---|---|
| bin1 | 500 | 7.4 | 1.36 | 9.24 |
| bin2 | 500 | 15.2 | 0.78 | 8.84 |
| bin3 | 500 | 24.3 | 0.41 | 8.09 |
| bin4 | 500 | 37.0 | 0.67 | 6.99 |
| **Overall** | **2,000** | | **0.70** | **7.88** |

Overall BLEU signature: 13.8 / 2.4 / 0.4 / 0.1 (1- to 4-gram precisions), brevity penalty 0.623, hypothesis length 32,711 vs reference length 48,169.

### Example Prediction

```
Source : Which has clinical fruit smell and sharp taste.
Ref    : जिसमें लाक्षणिक फल गंध और तीखा स्वाद होता है।
Output : एक बार-बारी रोग है।
```

The output is fluent-looking Hindi but not faithful to the source, which is typical of an under-trained model.
