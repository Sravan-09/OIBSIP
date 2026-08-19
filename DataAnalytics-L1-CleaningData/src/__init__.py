"""
src package initialization for Data Cleaning task.
"""
from .cleaning_utils import (
    download_and_load_data,
    generate_data_quality_report,
    handle_missing_values,
    remove_duplicates,
    standardize_categorical_features,
    detect_and_treat_outliers_iqr,
    enforce_data_types,
    generate_before_after_comparison,
    save_cleaned_data,
    save_figure,
    save_table
)

__all__ = [
    "download_and_load_data",
    "generate_data_quality_report",
    "handle_missing_values",
    "remove_duplicates",
    "standardize_categorical_features",
    "detect_and_treat_outliers_iqr",
    "enforce_data_types",
    "generate_before_after_comparison",
    "save_cleaned_data",
    "save_figure",
    "save_table"
]
