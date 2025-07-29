"""
data_utils.py

Utilidades para descarga, carga y preprocesamiento de datasets (CIFAR-10 y FashionMNIST)
para proyectos de deep learning en edge/federated learning.
"""

import os
from typing import Tuple, Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Carpeta base para almacenar los datos
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Estadísticas recomendadas para normalización
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD = (0.3530,)


def get_cifar10_dataloaders(
        batch_size: int = 128,
        shuffle_train: bool = True,
        shuffle_test: bool = False,
        num_workers: int = 2,
        data_augmentation: bool = True,
        download: bool = True,
        data_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Descarga (si es necesario), prepara y retorna los DataLoaders de entrenamiento y test para CIFAR-10.

    Args:
        batch_size (int): Tamaño de batch para los DataLoaders.
        shuffle_train (bool): Si barajar el set de entrenamiento.
        shuffle_test (bool): Si barajar el set de test.
        num_workers (int): Número de workers para los DataLoaders.
        data_augmentation (bool): Si aplicar aumentación de datos al entrenamiento.
        download (bool): Si descargar el dataset si no está disponible.
        data_dir (str, optional): Ruta personalizada para almacenar los datos.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    dir_ = data_dir or data_root
    # Transforms para entrenamiento
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ] if data_augmentation else []
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]
    train_dataset = datasets.CIFAR10(
        root=dir_, train=True, download=download,
        transform=transforms.Compose(train_transforms)
    )
    test_dataset = datasets.CIFAR10(
        root=dir_, train=False, download=download,
        transform=transforms.Compose(test_transforms)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers
    )
    return train_loader, test_loader


def get_fashionmnist_dataloaders(
        batch_size: int = 128,
        shuffle_train: bool = True,
        shuffle_test: bool = False,
        num_workers: int = 2,
        data_augmentation: bool = True,
        download: bool = True,
        data_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Descarga (si es necesario), prepara y retorna los DataLoaders de entrenamiento y test para FashionMNIST.

    Args:
        batch_size (int): Tamaño de batch para los DataLoaders.
        shuffle_train (bool): Si barajar el set de entrenamiento.
        shuffle_test (bool): Si barajar el set de test.
        num_workers (int): Número de workers para los DataLoaders.
        data_augmentation (bool): Si aplicar aumentación de datos al entrenamiento.
        download (bool): Si descargar el dataset si no está disponible.
        data_dir (str, optional): Ruta personalizada para almacenar los datos.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    dir_ = data_dir or data_root
    # Transforms para entrenamiento
    train_transforms = []
    if data_augmentation:
        train_transforms += [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(FASHIONMNIST_MEAN, FASHIONMNIST_STD)
    ]
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(FASHIONMNIST_MEAN, FASHIONMNIST_STD)
    ]
    train_dataset = datasets.FashionMNIST(
        root=dir_, train=True, download=download,
        transform=transforms.Compose(train_transforms)
    )
    test_dataset = datasets.FashionMNIST(
        root=dir_, train=False, download=download,
        transform=transforms.Compose(test_transforms)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers
    )
    return train_loader, test_loader


def get_cifar10_testloader(batch_size: int = 128, num_workers: int = 2, data_dir: Optional[str] = None):
    """
    Devuelve solo el DataLoader de test de CIFAR-10, normalizado.
    """
    dir_ = data_dir or data_root
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    testset = datasets.CIFAR10(root=dir_, train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return testloader
