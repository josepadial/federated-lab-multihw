"""
data_utils.py

Utilities for downloading, loading, and preprocessing datasets (CIFAR-10 and FashionMNIST)
for deep learning projects in edge/federated learning.
"""

import logging
import os
from typing import Tuple, Optional, Callable

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base folder to store data
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Recommended statistics for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD = (0.3530,)


def _validate_loader_args(batch_size: int, num_workers: int):
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, got {num_workers}")


def _get_transforms(mean: tuple, std: tuple, data_augmentation: bool = False, aug_transforms: Optional[list] = None) -> \
        Tuple[Callable, Callable]:
    """
    Returns train and test transforms for a dataset.
    """
    train_transforms = aug_transforms if data_augmentation and aug_transforms else []
    train_transforms += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    test_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


class DataLoaderFactory:
    """
    Factory class for creating DataLoaders for supported datasets.
    """

    @staticmethod
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
        """
        _validate_loader_args(batch_size, num_workers)
        dir_ = data_dir or data_root
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        train_tf, test_tf = _get_transforms(CIFAR10_MEAN, CIFAR10_STD, data_augmentation, aug)
        try:
            train_dataset = datasets.CIFAR10(
                root=dir_, train=True, download=download, transform=train_tf
            )
            test_dataset = datasets.CIFAR10(
                root=dir_, train=False, download=download, transform=test_tf
            )
        except Exception as e:
            logger.error(f"Failed to load CIFAR-10: {e}")
            raise
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers
        )
        return train_loader, test_loader

    @staticmethod
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
        """
        _validate_loader_args(batch_size, num_workers)
        dir_ = data_dir or data_root
        aug = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        train_tf, test_tf = _get_transforms(FASHIONMNIST_MEAN, FASHIONMNIST_STD, data_augmentation, aug)
        try:
            train_dataset = datasets.FashionMNIST(
                root=dir_, train=True, download=download, transform=train_tf
            )
            test_dataset = datasets.FashionMNIST(
                root=dir_, train=False, download=download, transform=test_tf
            )
        except Exception as e:
            logger.error(f"Failed to load FashionMNIST: {e}")
            raise
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers
        )
        return train_loader, test_loader

    @staticmethod
    def get_cifar10_testloader(
            batch_size: int = 128,
            num_workers: int = 2,
            data_dir: Optional[str] = None
    ) -> DataLoader:
        """
        Returns only the normalized CIFAR-10 test DataLoader.
        """
        _validate_loader_args(batch_size, num_workers)
        dir_ = data_dir or data_root
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        try:
            testset = datasets.CIFAR10(root=dir_, train=False, download=True, transform=test_transform)
        except Exception as e:
            logger.error(f"Failed to load CIFAR-10 test set: {e}")
            raise
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return testloader

    @staticmethod
    def get_fashionmnist_testloader(
            batch_size: int = 128,
            num_workers: int = 2,
            data_dir: Optional[str] = None
    ) -> DataLoader:
        """
        Returns only the normalized FashionMNIST test DataLoader.
        """
        _validate_loader_args(batch_size, num_workers)
        dir_ = data_dir or data_root
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHIONMNIST_MEAN, FASHIONMNIST_STD)
        ])
        try:
            testset = datasets.FashionMNIST(root=dir_, train=False, download=True, transform=test_transform)
        except Exception as e:
            logger.error(f"Failed to load FashionMNIST test set: {e}")
            raise
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return testloader


# For backward compatibility
get_cifar10_dataloaders = DataLoaderFactory.get_cifar10_dataloaders
get_fashionmnist_dataloaders = DataLoaderFactory.get_fashionmnist_dataloaders
get_cifar10_testloader = DataLoaderFactory.get_cifar10_testloader
get_fashionmnist_testloader = DataLoaderFactory.get_fashionmnist_testloader
