# Assignment 2 — TF/IDF Normalization and Logistic Regression

## Overview

This assignment extends the work completed in the previous assignment.

The same dataset created in the previous assignment is used to investigate how different **Term Frequency (TF)** and **Inverse Document Frequency (IDF)** normalization techniques affect the performance of a logistic regression text classification model.

The assignment includes:

1. Two different normalization methods for Term Frequency (TF).
2. Normalization of IDF based on the most frequent word across sentences.
3. Logistic regression using different combinations of normalized and unnormalized TF and IDF vectors.
4. Comparison of the performance of all six TF/IDF combinations.

The main objective is to understand whether normalization of TF and IDF improves the performance of the classification model.

---

# Objectives

The objectives of this assignment are:

- Implement two different TF normalization techniques.
- Implement normalized IDF.
- Generate TF-IDF feature vectors using different combinations of TF and IDF.
- Train a logistic regression classifier using each representation.
- Compare normalized TF/IDF representations with the unnormalized representation.
- Evaluate the classification performance using appropriate evaluation metrics.
- Determine how normalization affects the final classification results.

---

# Dataset

The same dataset from the previous assignment is used.

The dataset consists of text documents/sentences along with their corresponding class labels.

The text data is converted into numerical feature vectors using TF-IDF before being provided to the logistic regression classifier.

The vocabulary and feature extraction process are kept consistent with the previous assignment so that the effect of TF and IDF normalization can be compared fairly.

---

# Term Frequency (TF)

Term Frequency represents how frequently a particular term occurs in a document.

The basic or unnormalized TF is calculated as:

\[
TF(t,d) = count(t,d)
\]

where:

- `t` = term
- `d` = document
- `count(t,d)` = number of times term `t` occurs in document `d`

For example, if a word occurs 5 times in a document:

```text
TF = 5
