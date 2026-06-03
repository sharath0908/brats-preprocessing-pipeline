"""
BraTS 2023 Preprocessing Pipeline Configuration
Contains all hyperparameters and settings for data preprocessing
"""

import os
from pathlib import Path

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET_PATH = "/path/to/brats2023"  # Update with your dataset path
MODALITIES = ["t1", "t1ce", "t2", "flair"]  # Input MRI modalities
NUM_MODALITIES = len(MODALITIES)
NUM_CLASSES = 3  # Background, Tumor, Edema (adjust based on your annotation)

# ============================================================================
# Image Configuration
# ============================================================================
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
ORIGINAL_DEPTH = 155  # BraTS typically has 155 slices

# ============================================================================
# Normalization Configuration
# ============================================================================
NORMALIZATION_TYPE = "zscore"  # Options: "zscore", "minmax", "robust"
CLIP_VALUES = True  # Clip intensity values to remove outliers
CLIP_PERCENTILE = (0.5, 99.5)  # Lower and upper percentile for clipping

# ============================================================================
# Data Split Configuration
# ============================================================================
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ============================================================================
# Data Augmentation Configuration
# ============================================================================
AUGMENTATION_CONFIG = {
    "rotation_degrees": 15,  # Random rotation range in degrees
    "flip_probability": 0.5,  # Probability of horizontal/vertical flip
    "elastic_deformation": {
        "sigma": 5,  # Standard deviation of Gaussian kernel
        "alpha": 30,  # Maximum displacement
        "probability": 0.3,
    },
    "gaussian_noise": {
        "mean": 0.0,
        "std": 0.01,  # Noise standard deviation relative to image intensity
        "probability": 0.3,
    },
    "brightness_contrast": {
        "brightness_delta": 0.2,
        "contrast_delta": 0.2,
        "probability": 0.3,
    },
}

# ============================================================================
# DataLoader Configuration
# ============================================================================
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE_TRAIN = True

# ============================================================================
# Training Configuration
# ============================================================================
DEVICE = "cuda"  # Options: "cuda", "cpu"
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# ============================================================================
# Output Configuration
# ============================================================================
OUTPUT_DIR = Path("./output")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
