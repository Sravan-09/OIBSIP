# OASIS INFOBYTE Data Analytics Internship
## Level 2 Task 5: Autocomplete and Autocorrect Data Analytics (NLP Engine)

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green.svg)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![PySpellChecker](https://img.shields.io/badge/PySpellChecker-0.9.0-purple.svg)](https://pyspellchecker.readthedocs.io/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-teal.svg)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📌 Project Overview & Objective
This repository contains the complete implementation for **OASIS INFOBYTE Data Analytics Level 2 Task 5: Autocomplete and Autocorrect Data Analytics**.

The primary objective of this project is to analyze, implement, evaluate, and compare **Autocomplete** (Frequency-based N-Gram Language Models) and **Autocorrect** (Edit-Distance Spelling Correction) systems on a large public-domain text corpus.

---

## 📊 Corpus & Preprocessing Details
- **Corpus Source**: Public-domain Project Gutenberg literature via `nltk.corpus.gutenberg` (*Sherlock Holmes*, *Alice in Wonderland*, *Jane Eyre*, *Moby Dick*, *Hamlet*, *Macbeth*)
- **Total Corpus Size**: **622,080 tokens** (24,707 unique vocabulary tokens)
- **Tokenization**: Regular expression word matching (`r'\b[a-z]+\b'`) & lowercasing normalization.
- **Reasoned Stopword Policy**: Common stopwords (e.g. `'the'`, `'in'`, `'to'`) are **strictly retained** for Autocomplete because natural language syntax relies heavily on function words. Removing stopwords destroys sentence structure and predictive keyboard utility.

---

## 🛠️ Autocomplete & Autocorrect Methodology

### 1. Autocomplete Engine (N-Gram Language Models)
- **Bigram Model**: Predicts next word $w_i$ using unigram context $w_{i-1}$: $P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$
- **Trigram Model**: Predicts next word $w_i$ using bigram context $(w_{i-2}, w_{i-1})$: $P(w_i | w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}$ with Bigram backoff fallback.
- **Test Suite**: Evaluated on 10 benchmark input contexts (`"in the"`, `"she was"`, `"it is"`, `"one of"`, `"he had"`, etc.) displaying Top 3 predictions and probabilities.

### 2. Autocorrect Engine (Edit-Distance Spelling Correction)
- **Custom Levenshtein Corrector**: Generates Edit Distance 1 & 2 candidate words using deletions, transpositions, substitutions, and insertions. Filters candidate set against corpus vocabulary and ranks candidates by corpus unigram probability $P(w)$.
- **PySpellChecker System**: Benchmark spellchecker utilizing Peter Norvig's edit-distance algorithm and probabilistic word frequency lexicon.
- **Test Suite**: Evaluated on 20 benchmark misspelled words (`"teh"`, `"wrold"`, `"beutiful"`, `"goverment"`, `"speling"`, `"neccessary"`, `"definately"`, `"accommodate"`, etc.).

---

## 📈 Model Performance & Comparative Results

### Autocomplete Model Comparison (Bigram vs. Trigram)

| Model Approach | Precision@1 (%) | Precision@3 (%) | MRR Score | Selection |
| :--- | :---: | :---: | :---: | :---: |
| **Bigram Frequency Model** | 40.00% | 70.00% | 0.5500 | Baseline |
| **Trigram Frequency Model (With Backoff)** | **60.00%** | **90.00%** | **0.7333** | **BEST MODEL (Selected)** |

### Autocorrect Model Comparison (Custom Edit-Distance vs. PySpellChecker)

| Autocorrect Approach | Accuracy (%) | Precision | Recall | Selection |
| :--- | :---: | :---: | :---: | :---: |
| **Custom Levenshtein Edit-Distance** | **90.00%** | **0.9000** | **0.9000** | **Selected Custom** |
| **PySpellChecker (Norvig Algorithm)** | **90.00%** | **0.9000** | **0.9000** | Benchmark |

---

## 🔍 Limitations: Classical Implementation vs Production Google Keyboard (Gboard)

| Feature / Dimension | Classical Implementation (This Project) | Production Predictive Keyboards (Google Keyboard / Gboard) |
| :--- | :--- | :--- |
| **Model Architecture** | Fixed N-gram frequency tables ($N=2,3$) | Deep Neural Language Models (LSTMs, Transformer Decoders, MobileBERT) |
| **Contextual Range** | Limited to 1–2 preceding words | Long-range self-attention spanning full sentences & paragraphs |
| **Spatial Touch Model** | Assumes discrete text tokens | Continuous Gaussian key-touch centroids based on touchscreen geometry |
| **Personalization** | Global static corpus frequency | Dynamic user-specific vocabulary, contact lists & personal typing history |
| **Privacy & Security** | Local memory training | On-device Federated Learning with Differential Privacy & encrypted updates |
| **Hardware Latency** | Standard CPU Python dictionary lookups | Quantized TFLite / NPU hardware execution (<10ms latency) |

---

## 📂 Project Directory Structure

```
DataAnalytics-L2-AutocompleteAutocorrectAnalytics/
│
├── README.md                                     # Project documentation & summary report
├── requirements.txt                              # Python package dependencies
├── .gitignore                                    # Git tracking rules
│
├── data/
│   ├── raw/
│   │   ├── README.md                             # Corpus download documentation
│   │   └── gutenberg_corpus.txt                  # Public domain Project Gutenberg text corpus (3.4M chars)
│   └── processed/
|       └── corpus_statistics.json                # Corpus statistics including vocabulary, N-gram frequencies
│
├── notebooks/
│   └── Autocomplete_Autocorrect_Analytics.ipynb # Fully executed 27-cell Jupyter Notebook (0 errors)
│
├── src/
│   ├── __init__.py                               # Package initializer
│   ├── preprocessing.py                          # Data loader, regex tokenizer & stopword policy rationale
│   ├── autocomplete.py                           # Bigram & Trigram frequency autocomplete models
│   ├── autocorrect.py                            # Custom Levenshtein distance & PySpellChecker modules
│   └── evaluation.py                             # Precision@K, MRR, Accuracy & plot generators
│
└── outputs/
    ├── figures/
    │   ├── 01_top20_word_frequencies.png         # Bar chart of top 20 most frequent words in corpus
    │   ├── 02_autocomplete_topk_precision.png    # Precision@K comparison between Bigram vs Trigram
    │   ├── 03_autocorrect_accuracy_matrix.png    # Result matrix for autocorrect performance
    │   └── 04_edit_distance_distribution.png     # Distribution of edit distances for test words
    └── tables/
        ├── autocomplete_test_results.csv         # 10+ prefix test cases with Top-3 predictions
        ├── autocorrect_test_results.csv          # 20+ misspelling test cases with predictions & status
        ├── autocomplete_comparison_summary.csv   # Bigram vs Trigram metric comparison table
        └── autocorrect_comparison_summary.csv    # Custom Edit Distance vs PySpellChecker comparison table
```

---

## ⚡ Execution Instructions

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run End-to-End Analysis Notebook**:
   ```bash
   python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Autocomplete_Autocorrect_Analytics.ipynb
   ```

---

## 👨‍💻 Author Section
- **Author**: Bokkasam Sravan
- **Role**: Data Analytics Intern
- **Program**: OASIS INFOBYTE Data Analytics Internship (OIBSIP)
- **Task**: Level 2 Task 5 - Autocomplete and Autocorrect Data Analytics (NLP Engine)
- **Repository**: `OIBSIP`
- **Folder**: `DataAnalytics-L2-AutocompleteAutocorrectAnalytics/`
