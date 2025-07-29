"""
viz_utils.py

Visualization functions to compare the performance and characteristics of models in edge/federated learning.
"""

from typing import List, Dict, Optional, Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_accuracy_curves(
        histories: List[Dict[str, List[float]]],
        labels: Optional[List[str]] = None,
        val_histories: Optional[List[Dict[str, List[float]]]] = None,
        title: str = "Loss and Accuracy Curves",
        figsize: tuple = (12, 5)
) -> None:
    """
    Plots the training (and validation if provided) loss and accuracy curves for one or more models.

    Args:
        histories (List[Dict]): List of training histories (dict with keys 'loss', 'accuracy').
        labels (List[str], optional): Labels for each model.
        val_histories (List[Dict], optional): List of validation histories (dict with keys 'loss', 'accuracy').
        title (str): Plot title.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    # Loss
    plt.subplot(1, 2, 1)
    for i, hist in enumerate(histories):
        label = labels[i] if labels else f"Model {i + 1}"
        plt.plot(hist['loss'], label=f"Train {label}")
        if val_histories is not None:
            plt.plot(val_histories[i]['loss'], '--', label=f"Val {label}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.legend()
    plt.grid(True)
    # Accuracy
    plt.subplot(1, 2, 2)
    for i, hist in enumerate(histories):
        label = labels[i] if labels else f"Model {i + 1}"
        plt.plot(hist['accuracy'], label=f"Train {label}")
        if val_histories is not None:
            plt.plot(val_histories[i]['accuracy'], '--', label=f"Val {label}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per epoch')
    plt.legend()
    plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


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
    """
    df = pd.DataFrame(metrics)
    if model_names:
        df.index = model_names
    if columns:
        df = df[columns]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    try:
        from tabulate import tabulate
        print(tabulate(df, headers='keys', tablefmt=tablefmt, showindex=True))
    except ImportError:
        print(df)
    return df


def plot_bar_comparison(
        values: List[float],
        labels: List[str],
        ylabel: str = '',
        title: str = '',
        colors: Optional[List[str]] = None,
        ylim: Optional[tuple] = None,
        annotate: bool = True,
        figsize: tuple = (18, 8)
) -> None:
    """
    Creates a professional bar chart to compare values between models or variants.
    """
    plt.figure(figsize=figsize, facecolor='white')  # White background

    # Smaller bar width for spacing
    bar_width = 0.5

    bars = plt.bar(labels, values, color=colors, width=bar_width, edgecolor='black')
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, pad=20)

    # X-axis labels: smaller and less rotation
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=12)

    if ylim:
        plt.ylim(*ylim)

    # More subtle grid line, only on Y
    plt.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.7)

    # Margins to avoid compressing bars
    plt.subplots_adjust(bottom=0.25, left=0.08, right=0.98)

    # Cleaner annotations
    if annotate:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:,.2f}',  # Two decimals and thousands separator
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    plt.tight_layout()
    plt.show()
