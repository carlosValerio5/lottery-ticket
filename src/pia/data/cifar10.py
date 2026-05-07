"""
Carga de CIFAR-10 con partición train/validación y transformaciones estándar.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def _transforms_train() -> transforms.Compose:
    """Augmentación ligera: recorte aleatorio y volteo horizontal."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )


def _transforms_eval() -> transforms.Compose:
    """Solo tensor + normalización CIFAR-10."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )


def build_cifar10_loaders(
    *,
    data_root: str,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Construye ``DataLoader`` de entrenamiento y validación sobre CIFAR-10.

    Args:
        data_root: Directorio raíz para descargar/almacenar el dataset.
        batch_size: Tamaño de batch en train y val.
        val_fraction: Fracción del train oficial reservada a validación.
        num_workers: Workers de ``DataLoader`` (0 en macOS suele ser más estable).
        seed: Semilla para el reparto train/val.
        pin_memory: Si es ``None``, se activa solo si hay CUDA disponible.

    Returns:
        Tupla ``(train_loader, val_loader)``.
    """
    if not 0.0 < val_fraction < 1.0:
        msg = "val_fraction debe estar en (0, 1)."
        raise ValueError(msg)
    gen = torch.Generator().manual_seed(seed)
    train_set = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=_transforms_train()
    )
    n_total = len(train_set)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    train_subset, val_subset = random_split(train_set, [n_train, n_val], generator=gen)
    val_base = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=_transforms_eval()
    )
    val_indices = val_subset.indices
    val_ds: Subset[Any] = Subset(val_base, val_indices)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_cifar10_test_loader(
    *,
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader[Any]:
    """DataLoader sobre el conjunto de test oficial (evaluación final)."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    test_set = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_transforms_eval()
    )
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
