"""
viz_utils.py

Visualization utilities for comparing model performance and characteristics in edge/federated learning.
Includes loss/accuracy curves, comparative tables, and bar charts.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_loss_accuracy_curves(
        histories: List[Dict[str, List[float]]],
        labels: Optional[List[str]] = None,
        val_histories: Optional[List[Dict[str, List[float]]]] = None,
        title: str = "Loss and Accuracy Curves",
        figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plots the training (and validation if provided) loss and accuracy curves for one or more models.

    Args:
        histories (List[Dict]): List of training histories (dict with keys 'loss', 'accuracy').
        labels (List[str], optional): Labels for each model.
        val_histories (List[Dict], optional): List of validation histories (dict with keys 'loss', 'accuracy').
        title (str): Plot title.
        figsize (tuple): Figure size.
    Raises:
        ValueError: If input lists are empty or mismatched.
    """
    def _plot_metric(ax, metric: str):
        for i, hist in enumerate(histories):
            label = labels[i] if labels and i < len(labels) else f"Model {i + 1}"
            ax.plot(hist[metric], label=f"Train {label}")
            if val_histories is not None:
                ax.plot(val_histories[i][metric], '--', label=f"Val {label}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} per epoch')
        ax.legend()
        ax.grid(True)

    if not histories or not all(isinstance(h, dict) for h in histories):
        logger.error("histories must be a non-empty list of dicts.")
        raise ValueError("histories must be a non-empty list of dicts.")
    if val_histories and len(val_histories) != len(histories):
        logger.error("val_histories must match histories in length if provided.")
        raise ValueError("val_histories must match histories in length if provided.")
    _, axes = plt.subplots(1, 2, figsize=figsize)
    _plot_metric(axes[0], 'loss')
    _plot_metric(axes[1], 'accuracy')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close()


def show_comparative_table(
        metrics: List[Dict[str, Any]],
        model_names: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        tablefmt: str = 'github',
        sort_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates and displays a comparative table of key metrics for several models.

    Args:
        metrics (List[Dict]): List of dictionaries with metrics per model.
        model_names (List[str], optional): Model names.
        columns (List[str], optional): Columns to display.
        tablefmt (str): Table format for printing (using tabulate).
        sort_by (str, optional): Column to sort the table by.

    Returns:
        pd.DataFrame: DataFrame with the comparative table.
    Raises:
        ValueError: If metrics is empty or columns are invalid.
    """
    if not metrics or not all(isinstance(m, dict) for m in metrics):
        logger.error("metrics must be a non-empty list of dicts.")
        raise ValueError("metrics must be a non-empty list of dicts.")
    df = pd.DataFrame(metrics)
    if model_names:
        if len(model_names) != len(df):
            logger.warning("model_names length does not match metrics; ignoring model_names.")
        else:
            df.index = model_names
    if columns:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Columns not found in metrics: {missing_cols}")
            raise ValueError(f"Columns not found in metrics: {missing_cols}")
        df = df[columns]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    try:
        from tabulate import tabulate
        print(tabulate(df, headers='keys', tablefmt=tablefmt, showindex=True))
    except ImportError:
        logger.warning("tabulate not installed; printing DataFrame directly.")
        print(df)
    return df


def plot_bar_comparison(
        values: List[float],
        labels: List[str],
        ylabel: str = '',
        title: str = '',
        colors: Optional[List[str]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        annotate: bool = True,
        figsize: Tuple[int, int] = (18, 8)
) -> None:
    """
    Creates a professional bar chart to compare values between models or variants.

    Args:
        values (List[float]): Values to compare.
        labels (List[str]): Labels for each bar.
        ylabel (str): Y-axis label.
        title (str): Plot title.
        colors (List[str], optional): Bar colors.
        ylim (tuple, optional): Y-axis limits.
        annotate (bool): Whether to annotate bars with values.
        figsize (tuple): Figure size.
    Raises:
        ValueError: If values and labels are empty or mismatched.
    """
    # Convert pandas Series to list if needed
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()
    if not values or not labels or len(values) != len(labels):
        logger.error("values and labels must be non-empty and of the same length.")
        raise ValueError("values and labels must be non-empty and of the same length.")
    plt.figure(figsize=figsize, facecolor='white')  # White background
    bar_width = 0.5
    bars = plt.bar(labels, values, color=colors, width=bar_width, edgecolor='black')
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.7)
    plt.subplots_adjust(bottom=0.25, left=0.08, right=0.98)
    if annotate:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:,.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    plt.tight_layout()
    plt.show()
    plt.close()
