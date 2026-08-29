# Raw Data Directory: Project Gutenberg Text Corpus

## 📌 Dataset Overview
This directory contains the raw text corpus used for building, training, and evaluating the **Autocomplete** (Frequency-based N-Gram Language Models) and **Autocorrect** (Edit-Distance Spelling Correction) systems.

## 📖 Corpus Sources
The corpus is compiled from public-domain literature sourced from **Project Gutenberg** via the NLTK library (`nltk.corpus.gutenberg`), including:
1. *The Adventures of Sherlock Holmes* by Arthur Conan Doyle (`carroll-alice.txt` / `doyle-sherlock.txt`)
2. *Alice's Adventures in Wonderland* by Lewis Carroll
3. *Jane Eyre* by Charlotte Brontë (`bronte-jane.txt`)
4. *Moby Dick* by Herman Melville (`melville-moby_dick.txt`)
5. *Shakespeare Plays & Sonnets* (`shakespeare-hamlet.txt`, `shakespeare-macbeth.txt`)

## 💾 Acquisition & Reproduction
The corpus is automatically downloaded and compiled into `gutenberg_corpus.txt` via `src/preprocessing.py` during execution.

If automated download is unavailable, raw public domain text files can be fetched directly from:
- **Project Gutenberg Official Archive**: `https://www.gutenberg.org/`
- **NLTK Gutenberg Corpus**: `https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip`
