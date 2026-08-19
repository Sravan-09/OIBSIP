"""
src package initialization for Customer Segmentation Analysis task.
"""
from .segmentation_utils import (
    download_and_load_data,
    inspect_data,
    clean_transaction_data,
    compute_rfm_metrics,
    scale_features,
    evaluate_elbow_and_silhouette,
    fit_kmeans_clustering,
    compute_cluster_profiles,
    save_figure,
    save_table
)

__all__ = [
    "download_and_load_data",
    "inspect_data",
    "clean_transaction_data",
    "compute_rfm_metrics",
    "scale_features",
    "evaluate_elbow_and_silhouette",
    "fit_kmeans_clustering",
    "compute_cluster_profiles",
    "save_figure",
    "save_table"
]
