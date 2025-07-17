from pathlib import Path
import typing

import torch
from torchvision import datasets, transforms


def _calculate_current_step_value(
    initial: float,
    final: float,
    step: int,
    epochs: int,
    step_size: int,
    warmup_steps: int
) -> float:
    x_0 = initial
    x_n = final
    i = step
    N = epochs
    delta = step_size
    w = warmup_steps

    n = int(N / delta) # Number of steps


    x_i = x_0 + ((i - w) * ((x_n - x_0) / (n - w)))

    return x_i


class DataLoaderScheduler:
    '''
    Schedules and manages MNIST DataLoaders with progressive data augmentation.

    This class creates training and validation DataLoaders for the MNIST
    dataset, applying a transform that can be incrementally updated (stepped)
    each epoch to gradually increase augmentation parameters such as rotation,
    translation, scaling, and shearing.
    '''

    transform: transforms.Compose
    train_dataset = datasets.MNIST
    val_dataset = datasets.MNIST

    train_loader = torch.utils.data.DataLoader
    val_loader = torch.utils.data.DataLoader

    epochs: int
    warmup_steps: int
    step_size: int

    current_epoch: int
    current_step: int

    degrees_f: float
    degrees_i: float

    translate_f: float
    translate_i: float

    scale_max: float
    scale_min: float
    scale_i: float

    shear_f: float
    shear_i: float

    def __init__(
        self,
        root: Path,
        x_mean: float,
        x_std: float,
        batch_size: int,
        epochs: int,
        degrees: float,
        translate: float,
        scale: float,
        shear: float,
        warmup_steps: int = 0,
        step_size: int = 1
    ):
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.step_size = step_size

        self.current_epoch = 0
        self.current_step = 0

        self.degrees_f = degrees
        self.degrees_i = 0.0

        self.translate_f = translate
        self.translate_i = 0.0

        self.scale_i = 1.0
        self.scale_max = self.scale_i + scale
        self.scale_min = self.scale_i - scale

        self.shear_f = shear
        self.shear_i = 0.0

        self.transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=(self.degrees_i, self.degrees_i),
                translate=(self.translate_i, self.translate_i),
                scale=(self.scale_i, self.scale_i),
                shear=(self.shear_i, self.shear_i)
            ),                                        # PIL Image
            transforms.ToTensor(),                    # (C, H, W)
            transforms.Normalize((x_mean,), (x_std,)) # (C, H, W)
        ])

        self.train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        self.val_dataset = datasets.MNIST(root=root, train=False, download=True, transform=self.transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)

    def step(self) -> None:
        '''
        Increments the augmentation parameters for the next epoch and updates
        the transform's affine parameters (degrees, translate, scale, shear)
        by their respective step sizes, gradually increasing augmentation
        strength.
        '''

        self.current_epoch += 1
        if self.current_epoch % self.step_size == 0:
            self.current_step += 1

        if self.current_step <= self.warmup_steps:
            return

        self.transform.transforms[0].degrees = (
            _calculate_current_step_value(self.degrees_i, -self.degrees_f, self.current_step, self.epochs, self.step_size, self.warmup_steps),
            _calculate_current_step_value(self.degrees_i, self.degrees_f, self.current_step, self.epochs, self.step_size, self.warmup_steps)
        )

        self.transform.transforms[0].translate = (
            _calculate_current_step_value(self.translate_i, self.translate_f, self.current_step, self.epochs, self.step_size, self.warmup_steps),
            _calculate_current_step_value(self.translate_i, self.translate_f, self.current_step, self.epochs, self.step_size, self.warmup_steps)
        )

        self.transform.transforms[0].scale = (
            _calculate_current_step_value(self.scale_i, self.scale_min, self.current_step, self.epochs, self.step_size, self.warmup_steps),
            _calculate_current_step_value(self.scale_i, self.scale_max, self.current_step, self.epochs, self.step_size, self.warmup_steps)
        )

        self.transform.transforms[0].shear = (
            _calculate_current_step_value(self.shear_i, -self.shear_f, self.current_step, self.epochs, self.step_size, self.warmup_steps),
            _calculate_current_step_value(self.shear_i, self.shear_f, self.current_step, self.epochs, self.step_size, self.warmup_steps)
        )

    def loader(
        self,
        split: typing.Union[typing.Literal['train'], typing.Literal['val']]
    ) -> torch.utils.data.DataLoader:
        '''
        Returns the DataLoader for the specified split.

        Args:
            split: Which DataLoader to return.

        Returns:
            The requested DataLoader.

        Raises:
            Exception: If an invalid split is provided.
        '''

        if split == 'train':
            return self.train_loader
        elif split == 'val':
            return self.val_loader
        else:
            raise Exception(f'Invalid split {split}')
