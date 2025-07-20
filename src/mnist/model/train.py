import typing

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from mnist.model.dataloader import DataLoaderScheduler


DiagnosticList: typing.TypeAlias = typing.List[typing.Tuple[int, float, float, float, float, float]]

DataLoaderType: typing.TypeAlias = typing.Union[DataLoaderScheduler, typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]


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
        eval_iterations: Number of batches to evaluate. If set to -1 the entire
            dataset will be used.
        flatten: Whether to flatten input images (for MLPs) or not (for CNNs).
        device: Device to run evaluation on.

    Returns:
        Mean loss and accuracy over the evaluated batches.
    '''

    n = eval_iterations if eval_iterations != -1 else len(loader)
    losses = torch.zeros((n,), dtype=torch.float32)
    correct = 0.0
    total = 0

    i = 0
    images: Tensor # (B, 1, H, W)
    labels: Tensor # (B,)
    for images, labels in loader:
        if eval_iterations != -1:
            if  i == eval_iterations:
                break

        x = images.to(device)
        if flatten:
            x = x.view(-1, 28*28)
        y = labels.to(device)

        output = model(x) # (B, 10)
        loss: Tensor = F.cross_entropy(output, y)

        losses[i] = loss.item()

        predictions = torch.argmax(output, dim=1)
        correct += (predictions == y).sum().item()
        total += predictions.shape[0]

        i += 1

    return float(losses.mean().item()), correct / total


def dataset_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    flatten: bool = False,
    device: torch.device = None
) -> typing.Tuple[float, float]:
    '''
    Calculates the average loss and accuracy over an entire dataset.

    Args:
        model: The model to evaluate.
        loader: DataLoader for the dataset.
        flatten: Whether to flatten input images (for MLPs) or not (for CNNs).
        device: Device to run evaluation on.

    Returns:
        Mean loss and accuracy over the entire dataset.
    '''
    return estimate_loss(
        model=model,
        loader=loader,
        eval_iterations=-1,
        flatten=flatten,
        device=device
    )


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
    data_loader: DataLoaderType,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    flatten: bool = False,
    device: torch.device = None
) -> pd.DataFrame:
    '''
    Trains the model using a custom optimizer and learning rate scheduler.

    Args:
        model : The model to train.
        data_loader: Either a DataLoaderScheduler or a tuple containing the
            training dataloader and the validation dataloader.
        epochs: Number of epochs to train.
        max_lr: Initial learning rate.
        min_lr: Final learning rate after scheduling.
        flatten: Whether to flatten input images (for MLPs) or not (for CNNs).
        device: Device to run training on.

    Returns:
        DataFrame containing diagnostic information from the training.
    '''

    step_data_loader = False
    if isinstance(data_loader, DataLoaderScheduler):
        train_loader = data_loader.loader('train')
        val_loader = data_loader.loader('val')
        step_data_loader = True
    elif isinstance(data_loader, tuple):
        train_loader = data_loader[0]
        val_loader = data_loader[1]
    else:
        raise Exception('Invalid data_loader type')

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
        if step_data_loader:
            data_loader.step()

    return pd.DataFrame(diagnostics, columns=columns)


def train_basic_sdg(
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
    Trains the model using SGD and a step learning rate scheduler where the
    learning rate is reduced by the same factor each epoch to get from `max_lr`
    to `min_lr` in `epochs`.

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

    return train(
        model=model,
        data_loader=(train_loader, val_loader),
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        flatten=flatten,
        device=device
    )
