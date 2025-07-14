import typing

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor


DiagnosticList: typing.TypeAlias = typing.List[typing.Tuple[int, float, float, float, float, float]]


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    eval_iterations: int,
    flatten: bool = False,
    device: torch.device = None
) -> typing.Tuple[float, float]:
    '''
    Estimates the average loss and accuracy over a subset of data.

    Args:
        model: The model to evaluate.
        loader: DataLoader for the dataset.
        eval_iterations: Number of batches to evaluate.
        flatten: Whether to flatten input images (for MLPs) or not (for CNNs).
        device: Device to run evaluation on.

    Returns:
        Mean loss and accuracy over the evaluated batches.
    '''

    losses = torch.zeros((eval_iterations,), dtype=torch.float32)
    correct = 0.0
    total = 0

    i = 0
    images: Tensor # (B, 1, H, W)
    labels: Tensor # (B,)
    for images, labels in loader:
        if  i == eval_iterations:
            break

        x = images.to(device)
        if flatten:
            x = x.view(-1, 28*28)
        y = labels.to(device)

        output = model(x)
        loss: Tensor = F.cross_entropy(output, y)

        losses[i] = loss.item()

        predictions = torch.argmax(output, dim=1)
        correct += (predictions == y).sum().item()
        total += predictions.shape[0]

        i += 1

    return float(losses.mean().item()), correct / total


def print_diagnostics(
    epoch: int,
    lr: float,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float
) -> None:
    '''
    Prints training and validation diagnostics for the current epoch.

    Args:
        epoch: Current epoch number.
        lr: Current learning rate.
        train_loss: Training loss.
        train_accuracy: Training accuracy.
        val_loss: Validation loss.
        val_accuracy: Validation accuracy.
    '''

    print(
        f'epoch: {epoch}, '
        f'lr: {lr:.4e}, '
        f'train loss: {train_loss:.4f}, '
        f'train accuracy: {train_accuracy:.4f}, '
        f'val loss: {val_loss:.4f}, '
        f'val accuracy: {val_accuracy:.4f}'
    )


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    max_lr: float,
    min_lr: float,
    flatten: bool = False,
    device: torch.device = None
) -> pd.DataFrame:
    '''
    Trains the model using SGD and a learning rate scheduler.

    Args:
        model : The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of epochs to train.
        max_lr: Initial learning rate.
        min_lr: Final learning rate after scheduling.
        flatten: Whether to flatten input images (for MLPs) or not (for CNNs).
        device: Device to run training on.

    Returns:
        DataFrame containing diagnostic information from the training.
    '''

    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr)
    gamma = (min_lr / max_lr)**(1.0 / epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    columns = ['Epoch', 'Learning Rate', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']
    diagnostics: DiagnosticList = []
    epoch = 0

    while True:
        model.eval()
        train_loss, train_accuracy = estimate_loss(model, train_loader, 100, flatten=flatten, device=device)
        val_loss, val_accuracy = estimate_loss(model, val_loader, 100, flatten=flatten, device=device)
        lr = scheduler.get_last_lr()[0]

        print_diagnostics(epoch, lr, train_loss, train_accuracy, val_loss, val_accuracy)
        diagnostics.append((epoch, lr, train_loss, train_accuracy, val_loss, val_accuracy))

        if epoch == epochs:
            break
        epoch += 1

        images: Tensor # (B, 1, H, W)
        labels: Tensor # (B,)

        model.train()

        for images, labels in train_loader:
            x = images.to(device)
            if flatten:
                x = x.view(-1, 28*28)
            y = labels.to(device)
            model.zero_grad()
            output = model(x)
            loss: Tensor = F.cross_entropy(output, y)

            loss.backward()
            optimizer.step()

        scheduler.step()

    return pd.DataFrame(diagnostics, columns=columns)
