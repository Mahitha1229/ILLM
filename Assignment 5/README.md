# Word Embeddings, K-means Clustering and Nearest-Sentence Retrieval on IndicCorp V2 (Telugu)

This repository contains my ILLM assignment on training **Word2Vec, fastText and Doc2Vec** models on a Telugu corpus (`tel_Telu`, IndicCorp V2), **clustering the word vectors with K-means**, and **retrieving the closest training sentence** for every dev/test sentence using three different sentence representations.

Everything lives in one notebook, `assignment-5-27-8-illm.ipynb`, and was run on **Kaggle**. The notebook is cumulative: it contains the earlier assignments too, and this assignment is **PART 3** (the last section).

| Notebook section | Assignment |
|------------------|------------|
| Part 1 | Sentence corpus + BPE / WordPiece tokenizers |
| Part 2 | N-gram language models with smoothing |
| **Part 3 (this assignment)** | **Word embeddings, K-means clustering, nearest-sentence retrieval** |

---

## What This Assignment Does

1. Reuses the same splits as the earlier assignments: **1,000,000 train / 1,000 dev / 1,000 test** sentences (same seed, length-stratified dev/test).
2. Trains **Word2Vec**, **fastText** and **Doc2Vec** with `gensim`.
3. Runs **K-means (k = 50)** over the Word2Vec word vectors and lists the 20 words closest to each centroid.
4. Represents every sentence as a vector in three ways:
   - averaged Word2Vec vectors
   - averaged fastText vectors
   - Doc2Vec document vectors
5. For each dev and test sentence, finds the **closest training sentence by cosine similarity** under each representation, and saves the results.

---

## Pipeline Overview

### 1. Data
- Source: `ai4bharat/IndicCorpV2`, Telugu split, streamed with Hugging Face `datasets`.
- Sentences are split, de-duplicated and length-filtered (3-100 words).
- Loaded from the saved `train_1000000.txt`, `dev.txt` and `test.txt`; if they are missing, the notebook rebuilds identical splits automatically.
- Tokenization is simple whitespace splitting (`sentence.split()`).

### 2. Embedding Models

| Setting | Value |
|---------|-------|
| Vector size | 100 |
| Window | 5 |
| Min word count | 5 |
| Workers | 4 |
| Word2Vec | CBOW (`sg=0`), 5 epochs |
| fastText | CBOW (`sg=0`), 5 epochs |
| Doc2Vec | PV-DM (`dm=1`), 10 epochs |

- Vocabulary size is **138,056** words for all three models (words with count >= 5).
- Doc2Vec trains **1,000,000 document vectors**, one per training sentence, tagged by sentence index, so they line up with the training file.

### 3. K-means Clustering
- Algorithm: `sklearn.cluster.KMeans`, `n_clusters = 50`, `n_init = 10`, `random_state = 42`.
- Input: the Word2Vec word vectors (138,056 x 100).
- For each cluster, the **20 words nearest the centroid** (Euclidean distance) are saved to `kmeans_clusters.json`.

### 4. Sentence Vectors and Retrieval
- **Averaged Word2Vec / fastText**: mean of the word vectors of the sentence's words (zero vector if no word is found). fastText can also build vectors for out-of-vocabulary words from character n-grams.
- **Doc2Vec**: training sentences use the vectors learned during training; dev/test sentences (unseen) get vectors via `infer_vector` (10 epochs).
- **Nearest neighbour**: cosine similarity of each dev/test vector against all 1M training vectors, scanned in batches of 50,000 so the full similarity matrix is never held in memory.

---

## Results

### Mean Cosine Similarity of the Closest Training Sentence

| Method | Dev | Test |
|--------|----:|-----:|
| Averaged fastText | 0.9040 | 0.9034 |
| Averaged Word2Vec | 0.8950 | 0.8947 |
| Doc2Vec | 0.7295 | 0.7370 |

### Example Retrieval (dev sentence #0)

**Query:** `తాజాగా ఇజిన్‌ అనే కౌంటీలో కరోనా డెల్టా వేరియంట్‌ ప్రబలింది.`
(A COVID Delta variant has spread in a county called Izin.)

| Method | Closest training sentence | Cosine |
|--------|---------------------------|-------:|
| Averaged Word2Vec | `అందులోనూ కరోనా కొత్త రకం స్ట్రైయిన్ అనే వైరస్ విజృంభిస్తుండటంతో ప్రపంచదేశాలు అప్రమత్తమయ్యాయి.` (new COVID strain, countries on alert) | 0.8980 |
| Averaged fastText | `ఇప్పటికే అమెరికా ఫైజర్ అనే కరోనా నిర్మూలనకు టీకాను తయారుచేసింది.` (US has made a vaccine called Pfizer for COVID) | 0.8526 |
| Doc2Vec | `టోలోను-ఒబ్బి అనే జీబూ పశువుల కుస్తీ, అనేక ప్రాంతాలలో అభ్యసించబడుతుంది.` (a zebu cattle wrestling sport practiced in many regions) | 0.8397 |

### K-means Cluster Examples (top words near the centroid)

| Cluster | Theme (from the top words) | Sample words |
|--------:|----------------------------|--------------|
| 2 | Courts / legal | `పిటిషన్‌`, `హైకోర్టుకు`, `బెయిల్‌`, `సీబీఐకి`, `అఫిడవిట్` |
| 4 | Numbers | `42`, `55`, `72`, `120`, `95` |
| 5 | Cricket / sports | `కివీస్`, `బుమ్రా`, `రహానే`, `ఐపీఎల్‌లో`, `ప్రపంచకప్` |
| 6 | Politics / parties | `టీడీపీ`, `భాజపా`, `తెరాస`, `వైసిపి`, `శివసేన` |
| 7 | Health / biology / chemistry | `కాలేయం`, `లవణాలు`, `అమైనో`, `కలబంద`, `మూత్రపిండ` |
| 8 | Accidents / violence / casualties | `మృతిచెందారు.`, `జవాన్లు`, `మావోయిస్టులు`, `ఉగ్రవాదుల` |
| 1 | Place names + numbers | `విజయవాడలోను,`, `మంచిర్యాలలోను,`, `నరసన్నపేటలోను,`, `531030.` |
| 0, 3 | Conversational / narrative text | `నేనెప్పుడూ`, `నాన్నా.`, `నీతో`, `అమ్మకు`, `వాళ్ళకి` |
| 9 | Generic descriptive / instructional words | `సాధారణంగా,`, `ఉపయోగిస్తారు`, `క్లిష్టమైన`, `సృష్టించడానికి` |

All 50 clusters are saved in `kmeans_clusters.json`.

---

## Key Observations

- **The clusters are topically coherent.** Word2Vec groups words into clear themes (legal, sports, politics, health, accidents, numbers, place names, conversational language), so the embeddings capture semantic and domain similarity, not just spelling.
- **Averaged Word2Vec and fastText retrieve topically related sentences.** For the COVID query, both return COVID-related sentences. Doc2Vec's match shares only the common word `అనే` ("called") and is off-topic.
- **Doc2Vec performed worst here.** Its mean similarity is clearly lower (about 0.73 vs about 0.90) and the example match is poor. Likely reasons: Doc2Vec trains one vector per *sentence*, and with sentences averaging only about 10 words there is little signal per document; dev/test vectors are also *inferred* with a few gradient steps, so they are noisier than the vectors learned in training.
- **fastText is slightly ahead of Word2Vec on mean cosine similarity** (0.904 vs 0.895), and it can still embed words it never saw by using character n-grams. This is useful for Telugu, where a single root takes many suffixed forms.
- **Dev and test behave the same** (differences of at most about 0.007), so the results are stable across both splits.


---



## Output Files

Saved to `/kaggle/working/indiccorp_experiment/`:

| File | Description |
|------|-------------|
| `word2vec.model`, `fasttext.model`, `doc2vec.model` | Trained embedding models |
| `kmeans_clusters.json` | Top 20 words per cluster for all 50 clusters |
| `train_vecs_w2v.npy`, `train_vecs_ft.npy`, `train_vecs_d2v.npy` | Training sentence vectors for each method |
| `nearest_sentence_results.csv` | For every dev/test sentence and every method: the query, the closest training sentence, its index, and the cosine similarity |

