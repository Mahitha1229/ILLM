
# Subword Tokenization on IndicCorp V2 (Telugu): BPE vs WordPiece

This repository contains my ILLM assignment on sentence-level corpus construction and subword tokenizer training for **Telugu** (`tel_Telu`) using the **IndicCorp V2** corpus. I train **BPE** and **WordPiece** tokenizers and study how **training-data size** and **vocabulary size** affect tokenization quality.

The whole pipeline lives in a single notebook, `assignment-3-13-8-illm.ipynb`, and was run on **Kaggle** (CPU is enough).

---

## Assignment Parts

| Part | Task |
|------|------|
| 1 | Build sentence-level train / dev (1,000) / test (1,000) splits from IndicCorp V2 paragraphs. Dev/test are sampled so their sentence-length distribution mirrors the full corpus. |
| 2 | Train BPE and WordPiece tokenizers and tokenize the dev and test sets with each. |
| 3 | Repeat training with 100k / 300k / 500k / 1M sentences (vocab fixed at 30k) and compare. |
| 4 | Repeat training with 20k / 30k / 50k vocab sizes (training size fixed at 500k) and compare. |

---

## Pipeline Overview

1. **Streaming the data**: paragraphs are streamed from `ai4bharat/IndicCorpV2` (Hugging Face `datasets`, `streaming=True`), so the multi-GB file is never fully downloaded.
2. **Sentence splitting**: paragraph -> sentences with `indic-nlp-library`'s sentence tokenizer (ISO code `te`), with a regex fallback that splits on the danda (`।`), double danda (`॥`), `.`, `!` and `?`.
3. **Cleaning**: sentences are de-duplicated and length-filtered to **3-100 words**. A total of **1,052,000** unique Telugu sentences were collected.
4. **Length-stratified dev/test sampling**: sentences are binned by word count and dev/test are drawn proportionally from each bin (largest-remainder allocation). This keeps them representative instead of clustering on one length band.
5. **Nested training subsets**: the train pool (1,050,000 sentences) is shuffled once, and each smaller subset is a prefix of the larger ones. Size comparisons are therefore not confounded by different content.
6. **Tokenizer training** (Hugging Face `tokenizers`, Rust-backed):
   - Normalizer: `NFKC`
   - Pre-tokenizer: `Whitespace`
   - Special tokens: `[UNK] [PAD] [CLS] [SEP] [MASK]`
   - `min_frequency = 2`
   - Models: `BPE` (with `BPEDecoder`) and `WordPiece` (with `WordPiece` decoder)
7. **Evaluation** on dev and test with the metrics below.

### Evaluation Metrics

- **Avg tokens / sentence**: lower means a more compact encoding
- **Avg tokens / word**: the fertility of the tokenizer (1.0 = one token per word)
- **Chars / token**: compression, higher is better
- **UNK rate (%)**: fraction of `[UNK]` tokens

### Dataset Statistics

| Split | Sentences | Mean length (words) | Median | Max |
|-------|-----------|---------------------|--------|-----|
| Full pool | 1,052,000 | 10.5 | 9 | 100 |
| Dev | 1,000 | 10.5 | 9 | 51 |
| Test | 1,000 | 10.5 | 9 | 52 |
| Train pool | 1,050,000 | - | - | - |

The dev and test length statistics (mean, std, quartiles) closely match the full pool, so the stratified sampling worked as intended.

---

## Results (Test Split)

### Part 3: Effect of Training-Data Size (vocab = 30,000)

| Algorithm | Train size | Tokens/sentence | Tokens/word | Chars/token | UNK % |
|-----------|-----------:|----------------:|------------:|------------:|------:|
| BPE | 100,000 | 15.842 | 1.514 | 5.172 | 0.0 |
| BPE | 300,000 | 15.809 | 1.511 | 5.183 | 0.0 |
| BPE | 500,000 | 15.831 | 1.513 | 5.175 | 0.0 |
| BPE | 1,000,000 | 15.816 | 1.512 | 5.180 | 0.0 |
| WordPiece | 100,000 | 16.298 | 1.558 | 5.027 | 0.0 |
| WordPiece | 300,000 | 16.240 | 1.552 | 5.045 | 0.0 |
| WordPiece | 500,000 | 16.241 | 1.552 | 5.045 | 0.0 |
| WordPiece | 1,000,000 | 16.275 | 1.556 | 5.034 | 0.0 |

### Part 4: Effect of Vocabulary Size (train = 500,000)

| Algorithm | Vocab size | Tokens/sentence | Tokens/word | Chars/token | UNK % |
|-----------|-----------:|----------------:|------------:|------------:|------:|
| BPE | 20,000 | 16.765 | 1.602 | 4.887 | 0.0 |
| BPE | 30,000 | 15.831 | 1.513 | 5.175 | 0.0 |
| BPE | 50,000 | 14.864 | 1.421 | 5.512 | 0.0 |
| WordPiece | 20,000 | 17.324 | 1.656 | 4.729 | 0.0 |
| WordPiece | 30,000 | 16.243 | 1.553 | 5.044 | 0.0 |
| WordPiece | 50,000 | 15.177 | 1.451 | 5.398 | 0.0 |

---

## Key Observations

- **Training-data size has very little effect.** Going from 100k to 1M sentences changes avg tokens/sentence by only about 0.03 for BPE and about 0.06 for WordPiece. At a fixed 30k vocab, the learned merges/pieces are already well determined by 100k sentences, so more data brings diminishing returns.
- **Vocabulary size has a clear, monotonic effect.** Growing the vocab from 20k to 50k cuts tokens/sentence by roughly 11% for both algorithms (BPE 16.77 -> 14.86, WordPiece 17.32 -> 15.18) and raises chars/token. A larger vocab stores longer, more word-like units, so words are split into fewer pieces. The cost is a bigger embedding table downstream.
- **BPE is slightly more compact than WordPiece** in every configuration (about 0.4-0.5 fewer tokens per sentence at the same settings).
- **UNK rate is 0.0% everywhere.** Both tokenizers always have a fallback to smaller pieces and characters, and NFKC normalization plus a large training corpus covers the Telugu script well on dev/test.
- **Qualitative behaviour.** Telugu is agglutinative, so suffixes get split off as separate pieces (for example `కౌంటీ` + `లో`, where WordPiece marks the continuation as `##లో`). Frequent whole words (such as `తాజాగా`, `అనే`, `కరోనా`) stay as a single token.

---

---
## Output Files

Everything is written to `/kaggle/working/indiccorp_experiment/`:

| File | Description |
|------|-------------|
| `dev.txt`, `test.txt` | Length-stratified dev and test sentences (1,000 each) |
| `train_pool_full.txt` | Full shuffled training pool |
| `train_{100000,300000,500000,1000000}.txt` | Nested training subsets |
| `bpe_size*.json`, `wp_size*.json` | Tokenizers from the training-size study |
| `bpe_vocab*.json`, `wp_vocab*.json` | Tokenizers from the vocab-size study |
| `{dev,test}_{bpe,wp}_{size,vocab}*.txt` | Tokenized dev/test sets for every configuration |
| `results_train_size_study.csv` | Part 3 metrics |
| `results_vocab_size_study.csv` | Part 4 metrics |
| `results_all.json` | All results in one JSON file |
| `length_distribution_full.png` and metric plots | Sentence-length histogram and metric-vs-size / metric-vs-vocab plots |
