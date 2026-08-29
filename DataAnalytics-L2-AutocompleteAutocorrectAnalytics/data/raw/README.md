# Raw Corpus Information: Project Gutenberg Text Corpus

## Dataset Overview
- **Dataset Name**: Project Gutenberg Public Domain Literature Corpus
- **Primary Source**: Project Gutenberg / NLTK Gutenberg Corpus (`nltk.corpus.gutenberg`)
- **Direct Raw Download URL**: `https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip`
- **Expected Filename**: `gutenberg_corpus.txt`
- **Target Location**: `data/raw/gutenberg_corpus.txt`

## Dataset Description
This raw text corpus compiles classic public domain English literature from Project Gutenberg (including works by Arthur Conan Doyle, Lewis Carroll, Charlotte Brontë, Herman Melville, and William Shakespeare). It contains 294,232 words (1.68 MB of uncompressed raw text) serving as the baseline linguistic dataset for training n-gram autocomplete language models and building edit-distance autocorrect vocabulary dictionaries.

## Data Schema & Attributes
| Document Property | Format / Type | Description |
| :--- | :--- | :--- |
| `Corpus Content` | Plain Text (UTF-8) | Raw English prose and dialogue from classic literature used for n-gram frequency modeling and vocabulary extraction. |

## Automated Acquisition
The corpus loader utility in `src/preprocessing.py` automatically downloads and compiles `gutenberg_corpus.txt` into this directory when `download_and_compile_corpus()` is executed. Manual download is only required if running offline.
