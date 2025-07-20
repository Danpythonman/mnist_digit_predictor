import typing

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from mnist.utils.preprocessing import tensor2img


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
