"""
autocomplete.py - Frequency-based Bigram & Trigram N-Gram Autocomplete Language Models.
"""

from collections import defaultdict, Counter
import pandas as pd


class BigramAutocomplete:
    """
    Bigram Frequency Autocomplete Model: P(w_i | w_{i-1}) = C(w_{i-1}, w_i) / C(w_{i-1})
    """
    def __init__(self, tokens: list[str]):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.build_model(tokens)
        
    def build_model(self, tokens: list[str]):
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            self.unigram_counts[w1] += 1
            self.bigram_counts[w1][w2] += 1
        if tokens:
            self.unigram_counts[tokens[-1]] += 1
            
    def predict_next_word(self, context: str, top_k: int = 3) -> list[tuple[str, int, float]]:
        """
        Given a context string (takes last word), returns top_k predicted next words.
        Returns list of tuples: (predicted_word, count, probability)
        """
        tokens = context.lower().strip().split()
        if not tokens:
            return []
            
        last_word = tokens[-1]
        candidates = self.bigram_counts.get(last_word, Counter())
        total_context_count = self.unigram_counts.get(last_word, 0)
        
        if not candidates or total_context_count == 0:
            return []
            
        top_candidates = candidates.most_common(top_k)
        results = [
            (word, count, round(count / total_context_count, 4))
            for word, count in top_candidates
        ]
        return results


class TrigramAutocomplete:
    """
    Trigram Frequency Autocomplete Model: P(w_i | w_{i-2}, w_{i-1}) = C(w_{i-2}, w_{i-1}, w_i) / C(w_{i-2}, w_{i-1})
    With Bigram Backoff Fallback.
    """
    def __init__(self, tokens: list[str]):
        self.trigram_counts = defaultdict(Counter)
        self.bigram_context_counts = Counter()
        self.bigram_model = BigramAutocomplete(tokens)
        self.build_model(tokens)
        
    def build_model(self, tokens: list[str]):
        for w1, w2, w3 in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
            context = (w1, w2)
            self.bigram_context_counts[context] += 1
            self.trigram_counts[context][w3] += 1

    def predict_next_word(self, context: str, top_k: int = 3) -> list[tuple[str, int, float]]:
        """
        Given a context string (takes last 2 words), returns top_k predicted next words.
        Falls back to Bigram model if trigram context is unobserved.
        """
        tokens = context.lower().strip().split()
        if not tokens:
            return []
            
        if len(tokens) >= 2:
            ctx = (tokens[-2], tokens[-1])
            candidates = self.trigram_counts.get(ctx, Counter())
            ctx_count = self.bigram_context_counts.get(ctx, 0)
            
            if candidates and ctx_count > 0:
                top_candidates = candidates.most_common(top_k)
                return [
                    (word, count, round(count / ctx_count, 4))
                    for word, count in top_candidates
                ]
                
        # Bigram backoff fallback
        return self.bigram_model.predict_next_word(tokens[-1], top_k=top_k)


def run_autocomplete_test_suite(model, test_contexts: list[str], top_k: int = 3) -> pd.DataFrame:
    """
    Runs an autocomplete model across a list of test contexts and returns a structured DataFrame.
    """
    records = []
    for ctx in test_contexts:
        preds = model.predict_next_word(ctx, top_k=top_k)
        pred_words = [p[0] for p in preds]
        scores = [f"{p[0]} ({p[2]:.2f})" for p in preds]
        
        records.append({
            "Input Context": ctx,
            "Top Predictions": ", ".join(pred_words) if pred_words else "No Prediction",
            "Top 3 Predictions with Probabilities": " | ".join(scores) if scores else "N/A"
        })
        
    return pd.DataFrame(records)
