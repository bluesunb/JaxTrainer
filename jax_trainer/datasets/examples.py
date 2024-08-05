import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from ml_collections import ConfigDict
from torchvision.datasets import CIFAR10, MNIST

from jax_trainer.datasets.collate import build_batch_collate
from jax_trainer.datasets.data_struct import DatasetModule, SupervisedBatch
from jax_trainer.datasets.transforms import image_to_numpy, normalize_transform
from jax_trainer.datasets.utils import build_dataloaders


def build_cifar10_datasets(dataset_config: ConfigDict):
    normalize = dataset_config.get("normalize", True)
    if normalize:
        normalize_fn = normalize_transform(np.array([0.4914, 0.4822, 0.4465]), np.array([0.2023, 0.1994, 0.2010]))
    else:
        normalize_fn = transforms.Lambda(lambda x: x)

    transform = transforms.Compose([image_to_numpy, normalize_fn])
    train_set = CIFAR10(dataset_config.data_dir, train=True, download=True, transform=transform)
    
    valid_size = dataset_config.get("valid_size", 5000)
    split_seed = dataset_config.get("split_seed", 42)
    train_set, valid_set = data.random_split(
        train_set,
        lengths=[len(train_set) - valid_size, valid_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    test_set = CIFAR10(dataset_config.data_dir, train=False, download=True, transform=transform)
    
    train_loader, valid_loader, test_loader = build_dataloaders(
        train_set, 
        valid_set,
        test_set,
        train=[True, False, False],
        collate_fn=build_batch_collate(SupervisedBatch),
        config=dataset_config
    )

    return DatasetModule(
        dataset_config,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader
    )
