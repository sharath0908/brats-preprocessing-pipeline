"""
Example Usage and Training Script for BraTS 2023 Preprocessing Pipeline
Demonstrates how to use the complete pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from dataloader import BraTSDataModule, create_dataloaders, visualize_batch
import config


def example_basic_usage():
    """
    Basic example: Load data and visualize a batch
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic DataLoader Usage")
    print("=" * 80)

    # Create data module
    data_module = BraTSDataModule(
        root_dir=config.DATASET_PATH,
        batch_size=4,
        num_workers=2,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=config.RANDOM_SEED,
        use_3d=False,
    )

    # Get dataloaders
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()
    test_loader = data_module.get_test_loader()

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Iterate through one batch
    for batch_idx, (images, masks, metadata) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Image shape: {images.shape}")  # Expected: [B, C, H, W]
        print(f"  Mask shape: {masks.shape}")    # Expected: [B, 1, H, W]
        print(f"  Batch size: {images.shape[0]}")
        print(f"  Modalities: {images.shape[1]}")
        print(f"  Height x Width: {images.shape[2]} x {images.shape[3]}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Mask dtype: {masks.dtype}")
        print(f"  Patient IDs: {[m['patient_id'] for m in metadata]}")

        # Show statistics
        print(f"  Image stats - Min: {images.min():.4f}, Max: {images.max():.4f}, Mean: {images.mean():.4f}, Std: {images.std():.4f}")
        print(f"  Mask stats - Min: {masks.min()}, Max: {masks.max()}, Unique values: {torch.unique(masks)}")

        break  # Just show first batch


def example_full_training_loop():
    """
    Example: Complete training loop with a simple model
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Full Training Loop")
    print("=" * 80)

    # Setup device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Simple model for demonstration
    class SimpleSegmentationModel(nn.Module):
        """Simple CNN for tumor segmentation"""
        def __init__(self, num_modalities=4):
            super().__init__()
            self.conv1 = nn.Conv2d(num_modalities, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.sigmoid(self.conv3(x))
            return x

    # Initialize model
    model = SimpleSegmentationModel(num_modalities=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training loop
    num_epochs = 2  # Use 2 for demo, increase for real training
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, (images, masks, metadata) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.float().to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / num_batches
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for images, masks, metadata in val_loader:
                images = images.to(device)
                masks = masks.float().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                num_val_batches += 1

        if num_val_batches > 0:
            avg_val_loss = val_loss / num_val_batches
            print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")


def example_augmentation_effects():
    """
    Example: Visualize effects of data augmentation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Data Augmentation Effects")
    print("=" * 80)

    # Create data module
    data_module = BraTSDataModule(
        root_dir=config.DATASET_PATH,
        batch_size=4,
        num_workers=0,
        use_3d=False,
    )

    train_loader = data_module.get_train_loader()

    # Get a batch
    for images, masks, metadata in train_loader:
        print(f"Augmented batch shape: {images.shape}")
        print(f"Image intensity range: [{images.min():.4f}, {images.max():.4f}]")
        print(f"Image mean: {images.mean():.4f}, std: {images.std():.4f}")

        # Show sample from first modality
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                if idx < len(images):
                    im = images[idx, 0].numpy()  # First modality (T1)
                    axes[i, j].imshow(im, cmap='gray')
                    axes[i, j].set_title(f"T1 - {metadata[idx]['patient_id']}")
                    axes[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "augmentation_examples.png", dpi=100)
        print(f"Saved augmentation examples to {config.OUTPUT_DIR / 'augmentation_examples.png'}")
        plt.close()
        break


def example_3d_dataset():
    """
    Example: Using 3D dataset for full volume processing
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: 3D Dataset Usage")
    print("=" * 80)

    data_module = BraTSDataModule(
        root_dir=config.DATASET_PATH,
        batch_size=2,
        num_workers=2,
        use_3d=True,
    )

    train_loader = data_module.get_train_loader()

    for batch_idx, (images, masks, metadata) in enumerate(train_loader):
        print(f"\n3D Batch {batch_idx}:")
        print(f"  Image shape: {images.shape}")  # Expected: [B, C, D, H, W]
        print(f"  Mask shape: {masks.shape}")    # Expected: [B, 1, D, H, W]
        print(f"  Modalities: {images.shape[1]}")
        print(f"  Depth x Height x Width: {images.shape[2]} x {images.shape[3]} x {images.shape[4]}")

        # Show middle slice
        batch_size = images.shape[0]
        fig, axes = plt.subplots(batch_size, 4, figsize=(12, 3 * batch_size))

        for b in range(batch_size):
            middle_depth = images.shape[2] // 2
            for c in range(4):
                ax = axes[b, c] if batch_size > 1 else axes[c]
                im = images[b, c, middle_depth].numpy()
                ax.imshow(im, cmap='gray')
                ax.set_title(f"Modality {c}")
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "3d_dataset_examples.png", dpi=100)
        print(f"Saved 3D examples to {config.OUTPUT_DIR / '3d_dataset_examples.png'}")
        plt.close()
        break


def example_batch_statistics():
    """
    Example: Compute dataset statistics
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Dataset Statistics")
    print("=" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
    )

    # Compute statistics
    all_images = []
    all_masks = []

    print("Computing dataset statistics...")
    for images, masks, _ in train_loader:
        all_images.append(images)
        all_masks.append(masks)

    all_images = torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    print(f"\nDataset Statistics:")
    print(f"  Total images: {all_images.shape[0]}")
    print(f"  Image shape: {all_images.shape}")
    print(f"  Image dtype: {all_images.dtype}")
    print(f"  Image min: {all_images.min():.4f}")
    print(f"  Image max: {all_images.max():.4f}")
    print(f"  Image mean: {all_images.mean():.4f}")
    print(f"  Image std: {all_images.std():.4f}")

    print(f"\n  Mask unique values: {torch.unique(all_masks)}")
    print(f"  Tumor pixels: {(all_masks > 0).sum().item()}")
    print(f"  Background pixels: {(all_masks == 0).sum().item()}")
    print(f"  Tumor ratio: {(all_masks > 0).float().mean():.4f}")

    # Per-modality statistics
    print(f"\nPer-modality statistics:")
    for c in range(all_images.shape[1]):
        modality_data = all_images[:, c]
        print(f"  Modality {c} - Mean: {modality_data.mean():.4f}, Std: {modality_data.std():.4f}")


if __name__ == "__main__":
    # Run examples
    # Uncomment the examples you want to run

    # example_basic_usage()
    # example_augmentation_effects()
    # example_3d_dataset()
    # example_batch_statistics()
    # example_full_training_loop()

    print("=" * 80)
    print("BraTS 2023 Preprocessing Pipeline - Examples")
    print("=" * 80)
    print("\nUncomment the example functions in main to run them.")
    print("\nAvailable examples:")
    print("  1. example_basic_usage() - Basic data loading")
    print("  2. example_augmentation_effects() - Visualize augmentations")
    print("  3. example_3d_dataset() - Use 3D volumes")
    print("  4. example_batch_statistics() - Compute dataset statistics")
    print("  5. example_full_training_loop() - Complete training pipeline")
    print("\nUpdate config.DATASET_PATH before running examples!")
