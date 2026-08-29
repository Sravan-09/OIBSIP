"""
evaluation.py - Comparative evaluation metrics, summary tables, and visualization routines.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_autocomplete_models(bigram_model, trigram_model, eval_cases: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Evaluates Bigram vs Trigram models on context-target pairs.
    Computes Precision@1, Precision@3, and Mean Reciprocal Rank (MRR).
    """
    bigram_p1, bigram_p3, bigram_mrr = 0, 0, 0.0
    trigram_p1, trigram_p3, trigram_mrr = 0, 0, 0.0
    total = len(eval_cases)

    for context, target in eval_cases:
        target = target.lower().strip()

        # Bigram Eval
        b_preds = [p[0] for p in bigram_model.predict_next_word(context, top_k=3)]
        if b_preds:
            if b_preds[0] == target:
                bigram_p1 += 1
            if target in b_preds:
                bigram_p3 += 1
                rank = b_preds.index(target) + 1
                bigram_mrr += 1.0 / rank

        # Trigram Eval
        t_preds = [p[0] for p in trigram_model.predict_next_word(context, top_k=3)]
        if t_preds:
            if t_preds[0] == target:
                trigram_p1 += 1
            if target in t_preds:
                trigram_p3 += 1
                rank = t_preds.index(target) + 1
                trigram_mrr += 1.0 / rank

    records = [
        {
            "Model Approach": "Bigram Frequency Model",
            "Precision@1 (%)": round((bigram_p1 / total) * 100, 2),
            "Precision@3 (%)": round((bigram_p3 / total) * 100, 2),
            "MRR Score": round(bigram_mrr / total, 4)
        },
        {
            "Model Approach": "Trigram Frequency Model (With Backoff)",
            "Precision@1 (%)": round((trigram_p1 / total) * 100, 2),
            "Precision@3 (%)": round((trigram_p3 / total) * 100, 2),
            "MRR Score": round(trigram_mrr / total, 4)
        }
    ]

    return pd.DataFrame(records)


def evaluate_autocorrect_models(metrics_custom: dict, metrics_pyspell: dict) -> pd.DataFrame:
    """
    Creates a comparative summary table for Custom Edit Distance vs PySpellChecker.
    """
    records = [
        {
            "Autocorrect Approach": "Custom Levenshtein Edit-Distance",
            "Total Test Cases": metrics_custom["Total Test Cases"],
            "Correct Predictions": metrics_custom["Correct Predictions"],
            "Accuracy (%)": metrics_custom["Accuracy"],
            "Precision": metrics_custom["Precision"],
            "Recall": metrics_custom["Recall"]
        },
        {
            "Autocorrect Approach": "PySpellChecker (Norvig Algorithm)",
            "Total Test Cases": metrics_pyspell["Total Test Cases"],
            "Correct Predictions": metrics_pyspell["Correct Predictions"],
            "Accuracy (%)": metrics_pyspell["Accuracy"],
            "Precision": metrics_pyspell["Precision"],
            "Recall": metrics_pyspell["Recall"]
        }
    ]
    return pd.DataFrame(records)


def plot_top_20_words(top_20: list[tuple[str, int]], save_path: str):
    """Generates bar chart of top 20 most frequent words."""
    words = [t[0] for t in top_20]
    counts = [t[1] for t in top_20]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=counts, y=words, color="teal", ax=ax)
    ax.set_title("Top 20 Most Frequent Words in Project Gutenberg Corpus", fontweight="bold")
    ax.set_xlabel("Word Frequency Count")
    ax.set_ylabel("Word Token")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_autocomplete_comparison(df_comp: pd.DataFrame, save_path: str):
    """Plots Precision@1 vs Precision@3 comparison between Bigram and Trigram models."""
    df_melt = df_comp.melt(id_vars=["Model Approach"], value_vars=["Precision@1 (%)", "Precision@3 (%)"],
                           var_name="Metric", value_name="Percentage (%)")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x="Model Approach", y="Percentage (%)", hue="Metric", data=df_melt, palette=["teal", "darkcyan"], ax=ax)
    ax.set_title("Autocomplete Performance Comparison: Bigram vs. Trigram", fontweight="bold")
    ax.set_ylim(0, 100)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_autocorrect_results(df_custom: pd.DataFrame, df_pyspell: pd.DataFrame, save_path: str):
    """Plots comparison of Correct vs Incorrect counts between Custom Edit-Distance & PySpellChecker."""
    custom_correct = (df_custom["Result"] == "Correct").sum()
    custom_incorrect = (df_custom["Result"] == "Incorrect").sum()

    pyspell_correct = (df_pyspell["Result"] == "Correct").sum()
    pyspell_incorrect = (df_pyspell["Result"] == "Incorrect").sum()

    data = [
        {"Model": "Custom Edit-Distance", "Status": "Correct", "Count": custom_correct},
        {"Model": "Custom Edit-Distance", "Status": "Incorrect", "Count": custom_incorrect},
        {"Model": "PySpellChecker", "Status": "Correct", "Count": pyspell_correct},
        {"Model": "PySpellChecker", "Status": "Incorrect", "Count": pyspell_incorrect}
    ]
    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Model", y="Count", hue="Status", data=df_plot, palette=["teal", "crimson"], ax=ax)
    ax.set_title("Autocorrect Evaluation Matrix: Correct vs. Incorrect Predictions", fontweight="bold")
    ax.set_ylabel("Number of Words")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}",
                        (p.get_x() + p.get_width() / 2., h / 2.),
                        ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_edit_distance_distribution(df_custom: pd.DataFrame, save_path: str):
    """Plots distribution of Levenshtein edit distances for test misspelling corrections."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Edit Distance", data=df_custom, color="darkcyan", ax=ax)
    ax.set_title("Distribution of Edit Distances for Autocorrect Test Cases", fontweight="bold")
    ax.set_xlabel("Levenshtein Edit Distance (Number of Edits)")
    ax.set_ylabel("Word Count")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}",
                        (p.get_x() + p.get_width() / 2., h / 2.),
                        ha='center', va='center', color='black', fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
