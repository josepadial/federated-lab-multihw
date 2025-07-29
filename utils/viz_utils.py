"""
viz_utils.py

Funciones de visualización para comparar el rendimiento y características de modelos en edge/federated learning.
"""

from typing import List, Dict, Optional, Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_accuracy_curves(
        histories: List[Dict[str, List[float]]],
        labels: Optional[List[str]] = None,
        val_histories: Optional[List[Dict[str, List[float]]]] = None,
        title: str = "Curvas de pérdida y accuracy",
        figsize: tuple = (12, 5)
) -> None:
    """
    Grafica las curvas de pérdida y accuracy de entrenamiento (y validación si se proporciona) para uno o varios modelos.

    Args:
        histories (List[Dict]): Lista de historiales de entrenamiento (dict con claves 'loss', 'accuracy').
        labels (List[str], opcional): Etiquetas para cada modelo.
        val_histories (List[Dict], opcional): Lista de historiales de validación (dict con claves 'loss', 'accuracy').
        title (str): Título del gráfico.
        figsize (tuple): Tamaño de la figura.
    """
    plt.figure(figsize=figsize)
    # Pérdida
    plt.subplot(1, 2, 1)
    for i, hist in enumerate(histories):
        label = labels[i] if labels else f"Modelo {i + 1}"
        plt.plot(hist['loss'], label=f"Train {label}")
        if val_histories is not None:
            plt.plot(val_histories[i]['loss'], '--', label=f"Val {label}")
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.title('Pérdida por epoch')
    plt.legend()
    plt.grid(True)
    # Accuracy
    plt.subplot(1, 2, 2)
    for i, hist in enumerate(histories):
        label = labels[i] if labels else f"Modelo {i + 1}"
        plt.plot(hist['accuracy'], label=f"Train {label}")
        if val_histories is not None:
            plt.plot(val_histories[i]['accuracy'], '--', label=f"Val {label}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy por epoch')
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
    Genera y muestra una tabla comparativa de métricas clave para varios modelos.

    Args:
        metrics (List[Dict]): Lista de diccionarios con métricas por modelo.
        model_names (List[str], opcional): Nombres de los modelos.
        columns (List[str], opcional): Columnas a mostrar.
        tablefmt (str): Formato de tabla para impresión (usando tabulate).
        sort_by (str, opcional): Columna por la que ordenar la tabla.

    Returns:
        pd.DataFrame: DataFrame con la tabla comparativa.
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
        figsize: tuple = (18, 8)  # Más ancho y más alto
) -> None:
    """
    Crea una gráfica de barras profesional para comparar valores entre modelos o variantes.
    """
    plt.figure(figsize=figsize, facecolor='white')  # Fondo blanco

    # Ancho de barra más pequeño para que haya espacio
    bar_width = 0.5

    bars = plt.bar(labels, values, color=colors, width=bar_width, edgecolor='black')
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, pad=20)

    # Etiquetas eje X: más pequeñas y menos rotación
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=12)

    if ylim:
        plt.ylim(*ylim)

    # Línea de cuadrícula más discreta y solo en Y
    plt.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.7)

    # Márgenes para no comprimir tanto las barras
    plt.subplots_adjust(bottom=0.25, left=0.08, right=0.98)

    # Anotaciones más limpias
    if annotate:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:,.2f}',  # Dos decimales y separador de miles
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    plt.tight_layout()
    plt.show()
