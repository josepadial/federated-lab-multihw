"""
data_utils.py

Utilities for downloading, loading, and preprocessing datasets (CIFAR-10 and FashionMNIST)
for deep learning projects in edge/federated learning.
"""

import os
from typing import Tuple, Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Base folder to store data
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Recommended statistics for normalization
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
    Downloads (if necessary), prepares, and returns the training and test DataLoaders for CIFAR-10.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        shuffle_train (bool): Whether to shuffle the training set.
        shuffle_test (bool): Whether to shuffle the test set.
        num_workers (int): Number of workers for the DataLoaders.
        data_augmentation (bool): Whether to apply data augmentation to training.
        download (bool): Whether to download the dataset if not available.
        data_dir (str, optional): Custom path to store the data.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    dir_ = data_dir or data_root
    # Training transforms
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
    Downloads (if necessary), prepares, and returns the training and test DataLoaders for FashionMNIST.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        shuffle_train (bool): Whether to shuffle the training set.
        shuffle_test (bool): Whether to shuffle the test set.
        num_workers (int): Number of workers for the DataLoaders.
        data_augmentation (bool): Whether to apply data augmentation to training.
        download (bool): Whether to download the dataset if not available.
        data_dir (str, optional): Custom path to store the data.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    dir_ = data_dir or data_root
    # Training transforms
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
    Returns only the normalized CIFAR-10 test DataLoader.
    """
    dir_ = data_dir or data_root
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    testset = datasets.CIFAR10(root=dir_, train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return testloader
