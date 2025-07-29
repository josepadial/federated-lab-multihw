"""
metrics_utils.py

Functions to compute and display advanced evaluation metrics in deep learning projects.
"""

from typing import List, Optional, Dict, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


def compute_confusion_matrix(
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Computes the confusion matrix from true and predicted labels.

    Args:
        y_true (list or np.ndarray): True labels.
        y_pred (list or np.ndarray): Predicted labels.
        labels (list, optional): Class names.

    Returns:
        np.ndarray: Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=range(len(labels)) if labels else None)


def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        normalize: bool = False,
        figsize: tuple = (7, 6)
) -> None:
    """
    Plots the confusion matrix with labels and color scale.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list, optional): Class names.
        title (str): Plot title.
        cmap (str): Color map.
        normalize (bool): Whether to normalize the matrix by rows.
        figsize (tuple): Figure size.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compute_classification_metrics(
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        average: str = 'macro',
        labels: Optional[List[str]] = None,
        as_dataframe: bool = False
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Computes F1-score, precision, recall, and global and per-class accuracy.

    Args:
        y_true (list or np.ndarray): True labels.
        y_pred (list or np.ndarray): Predicted labels.
        average (str): Type of global average ('macro', 'micro', 'weighted').
        labels (list, optional): Class names.
        as_dataframe (bool): Whether to return results as DataFrame.

    Returns:
        dict or pd.DataFrame: Global and per-class metrics.
    """
    metrics = {}
    # Global metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    # Per-class metrics
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    if labels:
        class_labels = labels
    else:
        class_labels = [str(i) for i in range(len(metrics['f1_per_class']))]
    if as_dataframe:
        df = pd.DataFrame({
            'F1': metrics['f1_per_class'],
            'Precision': metrics['precision_per_class'],
            'Recall': metrics['recall_per_class']
        }, index=class_labels)
        global_df = pd.DataFrame({
            'Accuracy': [metrics['accuracy']],
            f'F1 ({average})': [metrics['f1']],
            f'Precision ({average})': [metrics['precision']],
            f'Recall ({average})': [metrics['recall']]
        })
        return {'global': global_df, 'per_class': df}
    return metrics
