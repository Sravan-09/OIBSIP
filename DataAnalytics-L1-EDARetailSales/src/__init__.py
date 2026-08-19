"""
src package initialization for EDA Retail Sales Data Analytics task.
"""
from .data_utils import (
    download_and_load_dataset,
    get_initial_inspection,
    clean_retail_data,
    calculate_descriptive_stats,
    save_figure,
    save_table
)

__all__ = [
    "download_and_load_dataset",
    "get_initial_inspection",
    "clean_retail_data",
    "calculate_descriptive_stats",
    "save_figure",
    "save_table"
]
