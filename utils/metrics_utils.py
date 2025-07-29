"""
metrics_utils.py

Funciones para calcular y mostrar métricas de evaluación avanzadas en proyectos de deep learning.
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
    Calcula la matriz de confusión a partir de etiquetas reales y predichas.

    Args:
        y_true (list o np.ndarray): Etiquetas reales.
        y_pred (list o np.ndarray): Etiquetas predichas.
        labels (list, opcional): Nombres de las clases.

    Returns:
        np.ndarray: Matriz de confusión.
    """
    return confusion_matrix(y_true, y_pred, labels=range(len(labels)) if labels else None)


def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Matriz de confusión",
        cmap: str = "Blues",
        normalize: bool = False,
        figsize: tuple = (7, 6)
) -> None:
    """
    Grafica la matriz de confusión con etiquetas y escala de colores.

    Args:
        cm (np.ndarray): Matriz de confusión.
        class_names (list, opcional): Nombres de las clases.
        title (str): Título del gráfico.
        cmap (str): Mapa de colores.
        normalize (bool): Si normalizar la matriz por filas.
        figsize (tuple): Tamaño de la figura.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
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
    Calcula F1-score, precisión, recall y accuracy global y por clase.

    Args:
        y_true (list o np.ndarray): Etiquetas reales.
        y_pred (list o np.ndarray): Etiquetas predichas.
        average (str): Tipo de promedio global ('macro', 'micro', 'weighted').
        labels (list, opcional): Nombres de las clases.
        as_dataframe (bool): Si devolver resultados como DataFrame.

    Returns:
        dict o pd.DataFrame: Métricas globales y por clase.
    """
    metrics = {}
    # Globales
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    # Por clase
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
