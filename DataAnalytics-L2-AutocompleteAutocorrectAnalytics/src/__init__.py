"""
OIBSIP Level 2 Task 5: Autocomplete and Autocorrect Data Analytics Package
"""

from .preprocessing import (
    download_and_compile_corpus,
    tokenize_text,
    get_stopword_policy_explanation,
    get_corpus_statistics
)

from .autocomplete import (
    BigramAutocomplete,
    TrigramAutocomplete,
    run_autocomplete_test_suite
)

from .autocorrect import (
    levenshtein_distance,
    CustomLevenshteinCorrector,
    PySpellCheckerCorrector,
    run_autocorrect_test_suite
)

from .evaluation import (
    evaluate_autocomplete_models,
    evaluate_autocorrect_models,
    plot_top_20_words,
    plot_autocomplete_comparison,
    plot_autocorrect_results,
    plot_edit_distance_distribution
)
