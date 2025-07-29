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


def train_model(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        num_epochs: int = 20,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = True
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
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.to(device)
    history = {'loss': [], 'accuracy': [], 'epoch_time': []}
    best_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        for inputs, targets in train_loader:
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
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_time = time.time() - epoch_start
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['epoch_time'].append(epoch_time)
        if verbose:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {epoch_time:.1f}s")
        # Early stopping
        if early_stopping_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                best_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (no improvement in {early_stopping_patience} epochs)")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
    total_time = time.time() - start_time
    history['total_time'] = total_time
    if verbose:
        print(f"Training completed in {total_time:.1f}s")
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
    if verbose:
        print(
            f"Eval | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Time: {total_time:.1f}s | Avg/sample: {avg_inference_time * 1000:.2f}ms")
    return metrics
