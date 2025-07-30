"""
metrics_utils.py

Utilities to compute and display advanced evaluation metrics for deep learning projects.
Includes confusion matrix, F1, precision, recall, and accuracy metrics, with plotting and DataFrame support.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Y_EMPTY_ERROR = "y_true and y_pred must not be empty."
Y_LENGTH_ERROR = "y_true and y_pred must have the same length."


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
    Raises:
        ValueError: If y_true or y_pred are empty or mismatched.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.error(Y_EMPTY_ERROR)
        raise ValueError(Y_EMPTY_ERROR)
    if len(y_true) != len(y_pred):
        logger.error(Y_LENGTH_ERROR)
        raise ValueError(Y_LENGTH_ERROR)
    label_range = range(len(labels)) if labels else None
    return confusion_matrix(y_true, y_pred, labels=label_range)


def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        normalize: bool = False,
        figsize: Tuple[int, int] = (7, 6)
) -> None:
    """
    Plots the confusion matrix with labels and color scale.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list, optional): Class names for axes.
        title (str): Plot title.
        cmap (str): Color map.
        normalize (bool): Whether to normalize the matrix by rows.
        figsize (tuple): Figure size.
    Raises:
        ValueError: If cm is not a 2D square matrix.
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        logger.error("Confusion matrix must be a 2D square matrix.")
        raise ValueError("Confusion matrix must be a 2D square matrix.")
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
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
) -> Union[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Computes F1-score, precision, recall, and global and per-class accuracy.

    Args:
        y_true (list or np.ndarray): True labels.
        y_pred (list or np.ndarray): Predicted labels.
        average (str): Type of global average ('macro', 'micro', 'weighted').
        labels (list, optional): Class names.
        as_dataframe (bool): Whether to return results as DataFrame.

    Returns:
        dict or dict of pd.DataFrame: Global and per-class metrics.
    Raises:
        ValueError: If y_true or y_pred are empty or mismatched.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.error(Y_EMPTY_ERROR)
        raise ValueError(Y_EMPTY_ERROR)
    if len(y_true) != len(y_pred):
        logger.error(Y_LENGTH_ERROR)
        raise ValueError(Y_LENGTH_ERROR)
    metrics = {'accuracy': accuracy_score(y_true, y_pred),
               'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
               'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
               'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
               'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0),
               'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0),
               'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0)}
    # Global metrics
    # Per-class metrics
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
