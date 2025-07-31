from pathlib import Path
import typing

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch import Tensor
from torchvision import datasets, transforms

from mnist.utils.preprocessing import tensor2img


PltFigure: typing.TypeAlias = Figure

PltAxes: typing.TypeAlias = typing.Union[
    Axes,
    typing.Sequence[Axes],
    typing.Sequence[typing.Sequence[Axes]],
    np.ndarray
]


def show_images_with_labels(
    images: Tensor,
    class_names: typing.List[str],
    actual_labels: Tensor,
    predicted_labels: typing.Optional[Tensor] = None,
    title: typing.Optional[str] = 'Batch of Images'
) -> None:
    '''
    Displays a batch of images with their actual and optionally predicted
    labels.

    Args:
        images: Batch of images as a tensor of shape (B, C, H, W).
        class_names: List of class names corresponding to label indices.
        actual_labels: Tensor of actual label indices for each image.
        predicted_labels: Tensor of predicted label indices for each image.
            If provided, both actual and predicted labels are shown.
        title: Title of the entire plot.
    '''

    number_of_images = images.shape[0]

    fig: Figure
    axes: typing.List[Axes]

    fig, axes = plt.subplots(1, number_of_images, figsize=(8, 2))

    for i in range(number_of_images):
        npimg = tensor2img(images[i])

        if predicted_labels is None:
            axes[i].set_title(class_names[actual_labels[i]])
        else:
            label_text = TextArea('Label: ')
            actual_label = TextArea(
                class_names[actual_labels[i]],
                textprops=dict(color='blue')
            )

            prediction_text = TextArea('Prediction: ')
            prediction_label = TextArea(
                class_names[predicted_labels[i]],
                textprops=dict(color='red')
            )

            hbox_actual = HPacker(
                children=[label_text, actual_label],
                align='center',
                pad=0,
                sep=0
            )

            hbox_prediction = HPacker(
                children=[prediction_text, prediction_label],
                align='center',
                pad=0,
                sep=0
            )

            vbox = VPacker(
                children=[hbox_actual, hbox_prediction],
                align='right',
                pad=0,
                sep=0
            )

            anchor = AnchoredOffsetbox(
                loc='upper center',
                child=vbox, pad=0.1,
                frameon=False,
                bbox_to_anchor=(0.5, 1.25),
                bbox_transform=axes[i].transAxes,
                borderpad=0
            )
            axes[i].add_artist(anchor)

        axes[i].imshow(npimg, cmap='hot')
        axes[i].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def generate_confusion_matrix(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: typing.Optional[torch.device] = None
):
    '''
    Generates a confusion matrix for model predictions on a dataset.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader for the dataset.
        device: Device to run evaluation on.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    '''

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(10))
    return cm


def generate_transformation_confusion_matrices(
    model: torch.nn.Module,
    data_root: Path,
    x_mean: float,
    x_std: float,
    axes: PltAxes,
    device: torch.device,
    resize: typing.Optional[typing.Tuple[int, int]] = None
) -> None:
    '''
    Generates and plots confusion matrices for various affine transformations.

    Args:
        model: The model to evaluate.
        data_root: Path to the MNIST dataset.
        x_mean: MNIST dataset mean for normalization.
        x_std: MNIST dataset standard deviation for normalization.
        axes: Matplotlib axes to plot the confusion matrices.
        device: Device to run evaluation on.
        resize: Optionally resize images to this size.
    '''

    transforms_to_apply = [
        transforms.RandomAffine( # Random affine transformation placeholder. For now there are no transformations applied.
            degrees=0,
            translate=(0, 0),
            scale=(1, 1),
            shear=0
        ),
        transforms.ToTensor(),
        transforms.Normalize((x_mean,), (x_std,))
    ]

    if resize is not None:
        transforms_to_apply.insert(0, transforms.Resize(resize))

    val_transform = transforms.Compose(transforms_to_apply)

    val_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

    for i in range(4):
        if i == 0:
            ax_title = 'Rotation up to 45 degrees'
            val_transform.transforms[0].shear = (0, 0)
            val_transform.transforms[0].degrees = (-45, 45)
        elif i == 1:
            ax_title = 'Translation up to 0.25x image size'
            val_transform.transforms[0].degrees = (0, 0)
            val_transform.transforms[0].translate = (0.25, 0.25)
        elif i == 2:
            ax_title = 'Scale up to 0.25x image size'
            val_transform.transforms[0].translate = (0, 0)
            val_transform.transforms[0].scale = (0.75, 1.25)
        else:
            ax_title = 'Shear up to 45 degrees'
            val_transform.transforms[0].scale = (1, 1)
            val_transform.transforms[0].shear = (45, 45)

        cm = generate_confusion_matrix(model, val_loader, device)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10), ax=axes[i])
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        axes[i].set_title(ax_title)


def generate_gradcam_class_plots(
    image: Tensor,
    label: int,
    predictions: Tensor,
    probabilities: Tensor,
    cam: GradCAM,
    axes: PltAxes,
    base_image: bool
) -> None:
    '''
    Generates Grad-CAM visualizations for each class for a given image.

    Args:
        image: Input image tensor of shape (1, C, H, W).
        label: Actual label of the image.
        predictions: Model predictions.
        probabilities: Model output probabilities.
        cam: GradCAM object for visualization.
        axes: Matplotlib axes to plot the images.
        base_image: Whether the image is the base image or the transformed
            image.
    '''

    cam.batch_size = 1

    axes[0].imshow(tensor2img(image.cpu()[0]), cmap='hot')
    axes[0].set_title(f'Actual: {label}\nPredicted: {predictions[0]}')
    axes[0].set_ylabel('Base Image' if base_image == True else 'Transformed Image')

    for i in range(10):
        grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget([i])])
        axes[i+1].imshow(grayscale_cam[0, :, :], cmap='hot')
        axes[i+1].set_title(f'Grad-CAM class {i}\nProb: {probabilities[0][i]:.2e}')

    for i in range(11):
        axes[i].set_xticks([])
        axes[i].set_yticks([])


def plot_diagnostics(diagnostics: pd.DataFrame, axes: PltAxes) -> None:
    '''
    Plots training diagnostics including loss, accuracy, and learning rate.

    Args:
        diagnostics: DataFrame containing training diagnostics.
        axes: Matplotlib axes to plot the diagnostics.
    '''

    axes[0].plot(diagnostics['Epoch'], diagnostics['Training Loss'], label='Training Loss')
    axes[0].plot(diagnostics['Epoch'], diagnostics['Validation Loss'], label='Validation Loss')
    axes[0].set_title('Loss during training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (cross entropy)')
    axes[0].set_yscale('log')
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(diagnostics['Epoch'], diagnostics['Training Accuracy'] * 100, label='Training Accuracy')
    axes[1].plot(diagnostics['Epoch'], diagnostics['Validation Accuracy'] * 100, label='Validation Accuracy')
    axes[1].set_title('Accuracy during training')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (percent)')
    axes[1].grid(True)
    axes[1].legend()
    axes[2].plot(diagnostics['Epoch'], diagnostics['Learning Rate'] * 100, label='Learning Rate')
    axes[2].set_title('Learning rate during training')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning rate')
    axes[2].grid(True)
