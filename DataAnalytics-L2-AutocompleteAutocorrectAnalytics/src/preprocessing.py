"""
preprocessing.py - Data loader, regex tokenizer, text normalizer,
corpus stats calculator, and stopword policy rationale for Autocomplete & Autocorrect.
"""

import os
import re
import urllib.request
from collections import Counter
import nltk

def download_and_compile_corpus(raw_dir: str) -> str:
    """
    Ensures a large public-domain Project Gutenberg text corpus exists in raw_dir.
    Compiles text from multiple classic works (Sherlock Holmes, Alice in Wonderland, Jane Eyre, etc.).
    """
    os.makedirs(raw_dir, exist_ok=True)
    corpus_path = os.path.join(raw_dir, "gutenberg_corpus.txt")
    
    if os.path.exists(corpus_path) and os.path.getsize(corpus_path) > 100000:
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        print(f"Loaded existing corpus from {corpus_path} ({len(text):,} characters)")
        return text

    combined_text = []
    
    # Try loading NLTK gutenberg corpus
    try:
        nltk.download('gutenberg', quiet=True)
        nltk.download('punkt', quiet=True)
        from nltk.corpus import gutenberg
        
        fileids = ['carroll-alice.txt', 'doyle-sherlock.txt', 'bronte-jane.txt', 'melville-moby_dick.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt']
        for fid in fileids:
            if fid in gutenberg.fileids():
                raw_txt = gutenberg.raw(fid)
                combined_text.append(raw_txt)
                print(f"Loaded NLTK Gutenberg file: {fid} ({len(raw_txt):,} chars)")
    except Exception as e:
        print(f"NLTK Gutenberg download warning: {e}")

    # Fallback to direct raw URL downloads if combined_text is empty or small
    if not combined_text or len("".join(combined_text)) < 100000:
        urls = [
            "https://raw.githubusercontent.com/datasets/gutenberg/master/data/alice.txt",
            "https://raw.githubusercontent.com/datasets/gutenberg/master/data/sherlock-holmes.txt"
        ]
        headers = {'User-Agent': 'Mozilla/5.0'}
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as resp:
                    txt = resp.read().decode('utf-8', errors='ignore')
                    combined_text.append(txt)
            except Exception as ex:
                print(f"Fallback download warning for {url}: {ex}")

    full_text = "\n\n".join(combined_text)
    if len(full_text) < 10000:
        raise ValueError("Failed to acquire sufficient corpus text from NLTK or fallback URLs.")
        
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"Compiled and saved Gutenberg corpus to {corpus_path} ({len(full_text):,} characters)")
    return full_text


def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes raw text into clean lowercase alphabetic tokens using regex matching.
    Strips numbers, punctuation marks, and non-ASCII symbols.
    """
    clean_text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', clean_text)
    return tokens


def get_stopword_policy_explanation() -> str:
    """
    Returns the analytical justification for why stopwords are retained for Autocomplete.
    """
    return (
        "### Reasoned Stopword Policy Rationale\n"
        "In traditional text classification or document clustering, stopwords (e.g., 'the', 'is', 'in', 'at', 'to') "
        "are frequently removed because they lack domain-specific semantic content. However, **removing stopwords is "
        "harmful for Autocomplete systems**.\n\n"
        "1. **Sequential Syntax**: Natural language syntax heavily relies on function words (e.g., 'in the', 'one of the', 'going to'). "
        "Predictive keyboards must predict 'the' after 'in' or 'to' after 'going'. Stripping stopwords breaks sentence structure.\n"
        "2. **User Intent Frequency**: Common stopwords are among the most frequently typed words in daily human communication.\n\n"
        "**Decision**: Stopwords are **strictly retained** during tokenization and n-gram construction to ensure realistic, syntax-aware autocomplete prediction."
    )


def get_corpus_statistics(tokens: list[str]) -> dict:
    """
    Computes statistical summary of the corpus tokens.
    """
    total_tokens = len(tokens)
    vocab = set(tokens)
    vocab_size = len(vocab)
    counter = Counter(tokens)
    top_20 = counter.most_common(20)
    avg_len = sum(len(t) for t in tokens) / max(1, total_tokens)
    ttr = vocab_size / max(1, total_tokens)
    
    return {
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "type_token_ratio": round(ttr, 4),
        "avg_token_length": round(avg_len, 2),
        "top_20_words": top_20
    }
