"""
Wine Quality Prediction Package
Oasis Infobyte Data Analytics Level 2 Task 2
"""

from .preprocessing import (
    fetch_and_save_raw_data,
    load_raw_data,
    clean_wine_data,
    engineer_quality_target,
)

from .model_utils import (
    get_project_root,
    evaluate_classifier,
    extract_misclassifications,
    save_model,
    save_figure,
    save_table,
)

__all__ = [
    'fetch_and_save_raw_data',
    'load_raw_data',
    'clean_wine_data',
    'engineer_quality_target',
    'get_project_root',
    'evaluate_classifier',
    'extract_misclassifications',
    'save_model',
    'save_figure',
    'save_table',
]
