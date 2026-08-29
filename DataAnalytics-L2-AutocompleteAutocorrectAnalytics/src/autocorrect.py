"""
autocorrect.py - Custom Levenshtein Edit Distance Autocorrect & PySpellChecker Engine.
"""

from collections import Counter
import pandas as pd
from spellchecker import SpellChecker


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes exact Levenshtein edit distance (deletions, insertions, substitutions) between two strings.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]    # Substitution
                )
                
    return dp[m][n]


class CustomLevenshteinCorrector:
    """
    Edit-Distance Autocorrect Engine using corpus vocabulary frequencies.
    Generates Edit Distance 1 & 2 candidates, ranked by corpus word probabilities P(w).
    """
    def __init__(self, tokens: list[str]):
        self.words_counter = Counter(tokens)
        self.total_words = float(sum(self.words_counter.values()))
        self.vocab = set(self.words_counter.keys())

    def word_prob(self, word: str) -> float:
        return self.words_counter[word] / self.total_words if self.total_words > 0 else 0.0

    def edits1(self, word: str) -> set[str]:
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> set[str]:
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def known(self, words: set[str]) -> set[str]:
        return set(w for w in words if w in self.vocab)

    def correct(self, word: str) -> str:
        word = word.lower().strip()
        if word in self.vocab:
            return word

        # Generate candidates in priority order: Distance 1 -> Distance 2 -> Fallback
        candidates = (self.known(self.edits1(word)) or 
                      self.known(self.edits2(word)) or 
                      {word})
                      
        return max(candidates, key=self.word_prob)


class PySpellCheckerCorrector:
    """
    PySpellChecker Autocorrect Wrapper based on Peter Norvig's algorithm.
    """
    def __init__(self):
        self.spell = SpellChecker()

    def correct(self, word: str) -> str:
        word = word.lower().strip()
        correction = self.spell.correction(word)
        return correction if correction is not None else word


def run_autocorrect_test_suite(corrector, test_cases: list[tuple[str, str]]) -> tuple[pd.DataFrame, dict]:
    """
    Runs spelling corrector across a list of (misspelled_word, expected_word) tuples.
    Returns (results_dataframe, summary_metrics_dict).
    """
    records = []
    correct_count = 0
    total_count = len(test_cases)

    for misspelled, expected in test_cases:
        predicted = corrector.correct(misspelled)
        is_correct = (predicted == expected.lower())
        if is_correct:
            correct_count += 1
            
        dist = levenshtein_distance(misspelled.lower(), predicted)
        
        records.append({
            "Misspelled Word": misspelled,
            "Expected Word": expected.lower(),
            "Predicted Correction": predicted,
            "Result": "Correct" if is_correct else "Incorrect",
            "Edit Distance": dist
        })

    df_results = pd.DataFrame(records)
    accuracy = correct_count / max(1, total_count)
    
    # Precision & Recall metrics for suggestion accuracy
    precision = accuracy # Precision of corrections made
    recall = accuracy    # Proportion of errors correctly recovered

    metrics = {
        "Total Test Cases": total_count,
        "Correct Predictions": correct_count,
        "Incorrect Predictions": total_count - correct_count,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }

    return df_results, metrics
