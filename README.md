"""
BraTS 2023 Preprocessing Pipeline - README
Complete documentation and setup guide
"""

# BraTS 2023 MRI Preprocessing Pipeline

A production-ready, research-quality PyTorch implementation of a complete MRI preprocessing pipeline for the BraTS 2023 (Brain Tumor Segmentation) challenge.

## Features

### ✅ Multi-Modal MRI Support
- **T1**: T1-weighted imaging
- **T1ce**: T1-weighted with contrast enhancement
- **T2**: T2-weighted imaging
- **FLAIR**: Fluid Attenuation Inversion Recovery

### ✅ Preprocessing
- **Intensity Normalization**
  - Z-score normalization
  - Min-Max normalization
  - Robust normalization (IQR-based)
  - Automatic outlier clipping
  
- **Spatial Resizing**
  - Bilinear interpolation
  - Maintains aspect ratio
  - Target: 224×224 pixels

### ✅ Data Augmentation
- Random rotation (±15°)
- Horizontal & vertical flipping
- Elastic deformation with Gaussian displacement
- Gaussian noise injection
- Brightness & contrast adjustment

### ✅ Dataset Management
- Automatic train/validation/test splitting
- Support for 2D slices and 3D volumes
- Efficient batch loading
- Metadata tracking

### ✅ Production Ready
- Comprehensive error handling
- Type hints throughout
- Extensive logging
- Example training scripts
- Visualization utilities

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/sharath0908/brats-preprocessing-pipeline.git
cd brats-preprocessing-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Organize your BraTS 2023 dataset as follows:

```
/path/to/brats2023/
├── BraTS-GLI-00002-000/
│   ├── BraTS-GLI-00002-000_t1.nii.gz
│   ├── BraTS-GLI-00002-000_t1ce.nii.gz
│   ├── BraTS-GLI-00002-000_t2.nii.gz
│   ├── BraTS-GLI-00002-000_flair.nii.gz
│   └── BraTS-GLI-00002-000_seg.nii.gz
├── BraTS-GLI-00003-000/
│   ├── BraTS-GLI-00003-000_t1.nii.gz
│   ├── BraTS-GLI-00003-000_t1ce.nii.gz
│   ├── BraTS-GLI-00003-000_t2.nii.gz
│   ├── BraTS-GLI-00003-000_flair.nii.gz
│   └── BraTS-GLI-00003-000_seg.nii.gz
└── ...
```

## Quick Start

### 1. Update Configuration

Edit `config.py`:

```python
DATASET_PATH = "/path/to/brats2023"  # Update with your dataset path
BATCH_SIZE = 8
NUM_WORKERS = 4
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
```

### 2. Basic Usage

```python
from dataloader import create_dataloaders
import config

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_path=config.DATASET_PATH,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
)

# Iterate through batches
for images, masks, metadata in train_loader:
    print(f"Image shape: {images.shape}")  # [B, C, H, W]
    print(f"Mask shape: {masks.shape}")    # [B, 1, H, W]
    # Your training code here
    break
```

### 3. Advanced Usage with Data Module

```python
from dataloader import BraTSDataModule

# Create data module with 70/15/15 split
data_module = BraTSDataModule(
    root_dir="/path/to/brats2023",
    batch_size=8,
    num_workers=4,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    random_seed=42,
    use_3d=False,  # Set to True for 3D volumes
)

# Get dataloaders
train_loader = data_module.get_train_loader()
val_loader = data_module.get_val_loader()
test_loader = data_module.get_test_loader()
```

### 4. Custom Preprocessing

```python
from preprocessing import IntensityNormalizer, ImageResizer, DataAugmentor
import numpy as np

# Initialize components
normalizer = IntensityNormalizer(method="zscore")
resizer = ImageResizer(target_shape=(224, 224))

# Load image (H, W, D)
image = np.random.randn(240, 240, 155).astype(np.float32)
mask = np.random.randint(0, 2, (240, 240, 155))

# Normalize
normalized = normalizer.normalize(image, mask)

# Resize
resized = resizer.resize(normalized)
```

## Tensor Shapes

### Input/Output Format

| Component | Shape | Description |
|-----------|-------|-------------|
| **Image Batch** | `[B, 4, 224, 224]` | B=batch size, 4 modalities, 224×224 pixels |
| **Mask Batch** | `[B, 1, 224, 224]` | Binary segmentation mask |
| **Single Image** | `[4, 224, 224]` | Single sample without batch dimension |
| **Single Mask** | `[1, 224, 224]` | Single mask without batch dimension |

### Data Types

| Tensor | dtype | Range |
|--------|-------|-------|
| Images | `float32` | Normalized: mean≈0, std≈1 |
| Masks | `long` | 0 (background), 1 (tumor) |

## Configuration

Key parameters in `config.py`:

```python
# Dataset
DATASET_PATH = "/path/to/brats2023"
MODALITIES = ["t1", "t1ce", "t2", "flair"]
NUM_MODALITIES = 4

# Image processing
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
NORMALIZATION_TYPE = "zscore"

# Augmentation
AUGMENTATION_CONFIG = {
    "rotation_degrees": 15,
    "flip_probability": 0.5,
    "elastic_deformation": {...},
    "gaussian_noise": {...},
}

# Training
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
```

## Examples

Run example scripts to understand the pipeline:

```bash
# Edit examples.py and uncomment desired example
python examples.py
```

### Available Examples

1. **Basic DataLoader Usage**
   ```python
   example_basic_usage()
   ```
   - Load and inspect batches
   - Print tensor shapes and statistics

2. **Augmentation Effects**
   ```python
   example_augmentation_effects()
   ```
   - Visualize augmented samples
   - Save visualization to disk

3. **3D Dataset Usage**
   ```python
   example_3d_dataset()
   ```
   - Process full 3D volumes
   - Handle (B, C, D, H, W) tensors

4. **Dataset Statistics**
   ```python
   example_batch_statistics()
   ```
   - Compute intensity statistics
   - Analyze tumor distribution

5. **Full Training Loop**
   ```python
   example_full_training_loop()
   ```
   - Complete training pipeline
   - Simple CNN baseline model

## Module Documentation

### `config.py`
Central configuration for all parameters. Update before running.

### `preprocessing.py`
Core preprocessing components:
- `IntensityNormalizer`: Multiple normalization strategies
- `ImageResizer`: Bilinear interpolation resizing
- `DataAugmentor`: Comprehensive augmentation techniques

### `dataset.py`
PyTorch Dataset implementations:
- `BraTSDataset`: 2D slice-based dataset
- `BraTSDatasetFull3D`: Full 3D volume dataset

### `dataloader.py`
DataLoader management:
- `BraTSDataModule`: Unified interface for all splits
- `create_dataloaders()`: Convenience function
- `visualize_batch()`: Batch visualization utility

### `examples.py`
Complete usage examples and training scripts.

## Performance Tips

### Memory Optimization
- Reduce batch size if OOM errors
- Use 2D dataset for smaller GPU memory
- Set `num_workers=0` for debugging

### Speed Optimization
- Increase `num_workers` for faster loading (4-8 recommended)
- Set `pin_memory=True` for GPU acceleration
- Use 3D dataset for fewer epoch iterations

### Training Tips
- Start with small dataset subset for debugging
- Use validation split to monitor overfitting
- Save checkpoints regularly
- Log metrics for analysis

## Troubleshooting

### FileNotFoundError
- Check dataset path in `config.py`
- Verify file naming matches expected format
- Ensure all modalities present for each patient

### Out of Memory (OOM)
- Reduce batch size in `config.py`
- Use 2D dataset instead of 3D
- Decrease number of workers

### Slow Data Loading
- Increase `num_workers` in config
- Verify `pin_memory=True`
- Check disk I/O performance

## Citation

If you use this pipeline, please cite:

```bibtex
@article{brats2023,
  title={The BRATS 2023 Challenge},
  journal={arXiv},
  year={2023}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

- BraTS Challenge organizers
- Medical Imaging community
- PyTorch and MONAI teams
