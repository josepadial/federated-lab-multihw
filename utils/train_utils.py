"""
train_utils.py

General functions for training and evaluating PyTorch models
for deep learning projects in edge/federated learning.
"""

import time
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .energy import GpuEnergyMeterNVML
from .logging_utils import get_logger


def _train_epoch(model: nn.Module, train_loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, logger) -> tuple[float, int, int]:
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        try:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        except Exception as ex:
            logger.exception("Error in training step: %s", ex)
            continue
    return running_loss, correct, total


def _run_epoch_with_optional_energy(model: nn.Module, train_loader: DataLoader, device: torch.device,
                                    optimizer: torch.optim.Optimizer, criterion: nn.Module, logger,
                                    meter: Optional[GpuEnergyMeterNVML]) -> tuple[float, float, float, float]:
    """Run one epoch and return (epoch_loss, epoch_acc, epoch_time, epoch_energy)."""
    epoch_energy = -1.0
    epoch_start = time.time()
    if meter is not None:
        try:
            epoch_stats: dict[str, float | int] = {}

            def _measured_epoch(stats=epoch_stats):  # default binds current dict for Sonar
                rl, cor, tot = _train_epoch(model, train_loader, device, optimizer, criterion, logger)
                stats['rl'] = rl; stats['cor'] = cor; stats['tot'] = tot

            e_j, _ = meter.measure(_measured_epoch)
            epoch_energy = e_j
            running_loss = float(epoch_stats.get('rl', 0.0))
            correct = int(epoch_stats.get('cor', 0))
            total = int(epoch_stats.get('tot', 0))
        except Exception as ex:
            logger.warning("Energy measurement failed: %s", ex)
            running_loss, correct, total = _train_epoch(model, train_loader, device, optimizer, criterion, logger)
    else:
        running_loss, correct, total = _train_epoch(model, train_loader, device, optimizer, criterion, logger)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, epoch_time, epoch_energy


def _update_history(history: Dict[str, List[Any]], loss: float, acc: float, t: float, energy: float,
                    measure_energy: bool) -> None:
    history['loss'].append(loss)
    history['accuracy'].append(acc)
    history['epoch_time'].append(t)
    if measure_energy:
        history.setdefault('epoch_energy_j', []).append(energy)


def _early_stopping_check(epoch_loss: float, best_loss: float, epochs_no_improve: int,
                          patience: Optional[int], model: nn.Module, best_state: Optional[Dict[str, Any]]):
    """Return (new_best_loss, new_epochs_no_improve, best_state, should_stop)."""
    if patience is None:
        return best_loss, epochs_no_improve, best_state, False
    if epoch_loss < best_loss:
        return epoch_loss, 0, model.state_dict(), False
    epochs_no_improve += 1
    return best_loss, epochs_no_improve, best_state, (epochs_no_improve >= patience)


def train_model(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        num_epochs: int = 20,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = True,
        measure_energy: bool = False,
) -> Dict[str, List[Any]]:
    """
    Trains a PyTorch model for a configurable number of epochs.

    Args:
        model (nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        train_loader (DataLoader): Training DataLoader.
        device (torch.device): Device ('cpu' or 'cuda').
        criterion (nn.Module, optional): Loss function. If None, uses CrossEntropyLoss.
        num_epochs (int): Number of training epochs.
        early_stopping_patience (int, optional): Number of epochs without improvement for early stopping.
        verbose (bool): Whether to print progress logs.

    Returns:
        Dict[str, List[Any]]: History of metrics per epoch (loss, accuracy, times, etc).
    """
    logger = get_logger("train_utils")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.to(device)
    history = {'loss': [], 'accuracy': [], 'epoch_time': []}
    if measure_energy:
        history['epoch_energy_j'] = []
    best_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    start_time = time.time()
    meter = GpuEnergyMeterNVML(0) if (measure_energy and torch.cuda.is_available()) else None
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, epoch_acc, epoch_time, epoch_energy = _run_epoch_with_optional_energy(
            model, train_loader, device, optimizer, criterion, logger, meter
        )
        _update_history(history, epoch_loss, epoch_acc, epoch_time, epoch_energy, measure_energy)
        logger.info("Epoch %d/%d | Loss: %.4f | Acc: %.4f | Time: %.1fs", epoch, num_epochs, epoch_loss, epoch_acc,
                    epoch_time)
        # Early stopping
        best_loss, epochs_no_improve, best_state, should_stop = _early_stopping_check(
            epoch_loss, best_loss, epochs_no_improve, early_stopping_patience, model, best_state
        )
        if should_stop:
            logger.info("Early stopping at epoch %d (no improvement in %d epochs)", epoch, early_stopping_patience)
            if best_state is not None:
                model.load_state_dict(best_state)
            break
    total_time = time.time() - start_time
    history['total_time'] = total_time
    if measure_energy:
        try:
            # Sum positive energies only
            total_energy = float(sum(e for e in history['epoch_energy_j'] if isinstance(e, (int, float)) and e > 0))
        except Exception:
            total_energy = -1.0
        history['total_energy_j'] = total_energy
    logger.info("Training completed in %.1fs", total_time)
    return history


def evaluate_model(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        return_confusion_matrix: bool = False,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluates a PyTorch model on a validation or test DataLoader.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (DataLoader): Validation or test DataLoader.
        device (torch.device): Device ('cpu' or 'cuda').
        criterion (nn.Module, optional): Loss function. If None, uses CrossEntropyLoss.
        return_confusion_matrix (bool): Whether to return the confusion matrix.
        verbose (bool): Whether to print progress logs.

    Returns:
        Dict[str, Any]: Dictionary with accuracy, loss, confusion matrix (optional), times, etc.
    """
    logger = get_logger("train_utils")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in data_loader:
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                if return_confusion_matrix:
                    all_targets.append(targets.cpu().numpy())
                    all_preds.append(predicted.cpu().numpy())
            except Exception as ex:
                logger.exception("Error in eval step: %s", ex)
                continue
    total_time = time.time() - start_time
    avg_loss = running_loss / total
    accuracy = correct / total
    avg_inference_time = total_time / total if total > 0 else 0.0
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_time': total_time,
        'avg_inference_time': avg_inference_time
    }
    if return_confusion_matrix:
        y_true = np.concatenate(all_targets) if all_targets else np.array([])
        y_pred = np.concatenate(all_preds) if all_preds else np.array([])
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred) if y_true.size > 0 else None
    logger.info(
        "Eval | Loss: %.4f | Acc: %.4f | Time: %.1fs | Avg/sample: %.2fms",
        avg_loss, accuracy, total_time, avg_inference_time * 1000.0,
    )
    return metrics
