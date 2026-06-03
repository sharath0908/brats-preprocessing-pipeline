"""
BraTS DataLoader Creation and Management
Handles data splitting and DataLoader initialization
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset

from dataset import BraTSDataset, BraTSDatasetFull3D
import config


class BraTSDataModule:
    """
    Data module for managing BraTS dataset splits and DataLoaders
    Handles train/validation/test splits and batch loading
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
        use_3d: bool = False,
    ):
        """
        Args:
            root_dir (str): Path to BraTS dataset
            batch_size (int): Batch size for DataLoader
            num_workers (int): Number of workers for DataLoader
            train_split (float): Proportion of data for training
            val_split (float): Proportion of data for validation
            test_split (float): Proportion of data for testing
            random_seed (int): Random seed for reproducibility
            use_3d (bool): Use 3D dataset instead of 2D
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Train/val/test splits must sum to 1.0"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.use_3d = use_3d

        # Initialize datasets and dataloaders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._setup()

    def _setup(self):
        """Create datasets and dataloaders"""
        # Determine dataset class
        dataset_class = BraTSDatasetFull3D if self.use_3d else BraTSDataset

        # Create full dataset (without split)
        full_dataset = dataset_class(
            root_dir=self.root_dir,
            split="train",  # Will be overridden for val/test
        )

        # Get patient indices
        num_patients = len(full_dataset)
        indices = np.arange(num_patients)

        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=self.train_split,
            random_state=self.random_seed,
        )

        # Second split: val vs test
        val_ratio = self.val_split / (self.val_split + self.test_split)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_ratio,
            random_state=self.random_seed,
        )

        print(f"Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        # Create datasets with appropriate splits
        self.train_dataset = dataset_class(
            root_dir=self.root_dir,
            augment=True,
            split="train",
        )
        self.train_dataset = Subset(self.train_dataset, train_indices)

        self.val_dataset = dataset_class(
            root_dir=self.root_dir,
            augment=False,
            split="val",
        )
        self.val_dataset = Subset(self.val_dataset, val_indices)

        self.test_dataset = dataset_class(
            root_dir=self.root_dir,
            augment=False,
            split="test",
        )
        self.test_dataset = Subset(self.test_dataset, test_indices)

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=config.PIN_MEMORY,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=config.PIN_MEMORY,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=config.PIN_MEMORY,
        )

    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader"""
        return self.train_loader

    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader"""
        return self.val_loader

    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader"""
        return self.test_loader

    def get_all_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all dataloaders"""
        return self.train_loader, self.val_loader, self.test_loader


def create_dataloaders(
    dataset_path: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    train_split: float = config.TRAIN_SPLIT,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
    random_seed: int = config.RANDOM_SEED,
    use_3d: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create dataloaders
    
    Args:
        dataset_path (str): Path to BraTS dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers
        train_split (float): Training split ratio
        val_split (float): Validation split ratio
        test_split (float): Test split ratio
        random_seed (int): Random seed
        use_3d (bool): Use 3D dataset
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data_module = BraTSDataModule(
        root_dir=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed,
        use_3d=use_3d,
    )

    return data_module.get_all_loaders()


def visualize_batch(loader: DataLoader, num_samples: int = 4):
    """
    Visualize a batch from dataloader
    
    Args:
        loader (DataLoader): DataLoader to visualize
        num_samples (int): Number of samples to show
    """
    import matplotlib.pyplot as plt

    # Get first batch
    batch_images, batch_masks, batch_metadata = next(iter(loader))

    print(f"Batch size: {batch_images.shape[0]}")
    print(f"Image shape: {batch_images.shape}")
    print(f"Mask shape: {batch_masks.shape}")
    print(f"Metadata keys: {batch_metadata[0].keys()}")

    # Visualize
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))

    for i in range(min(num_samples, len(batch_images))):
        # Show each modality
        for c in range(min(4, batch_images.shape[1])):
            ax = axes[i, c] if num_samples > 1 else axes[c]
            im = batch_images[i, c].numpy()
            ax.imshow(im, cmap='gray')
            ax.set_title(f"Modality {c}")
            ax.axis('off')

        # Show mask
        ax = axes[i, 4] if num_samples > 1 else axes[4]
        mask = batch_masks[i, 0].numpy()
        ax.imshow(mask, cmap='jet')
        ax.set_title("Mask")
        ax.axis('off')

        # Print patient info
        print(f"Sample {i}: {batch_metadata[i]}")

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "batch_visualization.png")
    print(f"Saved visualization to {config.OUTPUT_DIR / 'batch_visualization.png'}")
    plt.close()
