"""
BraTS 2023 MRI Preprocessing Module
Handles intensity normalization, resizing, and data augmentation
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class IntensityNormalizer:
    """
    Normalize MRI image intensity using multiple strategies
    """

    def __init__(self, method: str = "zscore", clip_values: bool = True, 
                 clip_percentile: Tuple[float, float] = (0.5, 99.5)):
        """
        Args:
            method (str): Normalization method - 'zscore', 'minmax', or 'robust'
            clip_values (bool): Whether to clip extreme values before normalization
            clip_percentile (tuple): Percentiles for clipping (lower, upper)
        """
        self.method = method
        self.clip_values = clip_values
        self.clip_percentile = clip_percentile

    def normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize image intensity
        
        Args:
            image (np.ndarray): Input MRI image
            mask (np.ndarray, optional): Brain mask to normalize only within brain region
            
        Returns:
            np.ndarray: Normalized image
        """
        # Get valid region for statistics calculation
        if mask is not None:
            valid_voxels = image[mask > 0]
        else:
            valid_voxels = image[image > 0]  # Exclude background zeros

        if len(valid_voxels) == 0:
            return image

        # Clip extreme values
        if self.clip_values:
            lower, upper = np.percentile(valid_voxels, self.clip_percentile)
            image = np.clip(image, lower, upper)
            if mask is not None:
                valid_voxels = image[mask > 0]
            else:
                valid_voxels = image[image > 0]

        # Apply normalization method
        if self.method == "zscore":
            return self._zscore_normalize(image, valid_voxels, mask)
        elif self.method == "minmax":
            return self._minmax_normalize(image, valid_voxels, mask)
        elif self.method == "robust":
            return self._robust_normalize(image, valid_voxels, mask)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def _zscore_normalize(self, image: np.ndarray, valid_voxels: np.ndarray, 
                         mask: Optional[np.ndarray]) -> np.ndarray:
        """Z-score normalization: (x - mean) / std"""
        mean = np.mean(valid_voxels)
        std = np.std(valid_voxels)
        
        if std < 1e-8:
            return image
            
        normalized = (image - mean) / std
        return normalized

    def _minmax_normalize(self, image: np.ndarray, valid_voxels: np.ndarray, 
                         mask: Optional[np.ndarray]) -> np.ndarray:
        """Min-Max normalization: (x - min) / (max - min)"""
        min_val = np.min(valid_voxels)
        max_val = np.max(valid_voxels)
        
        if max_val - min_val < 1e-8:
            return image
            
        normalized = (image - min_val) / (max_val - min_val)
        return normalized

    def _robust_normalize(self, image: np.ndarray, valid_voxels: np.ndarray, 
                         mask: Optional[np.ndarray]) -> np.ndarray:
        """Robust normalization using IQR: (x - median) / IQR"""
        q1 = np.percentile(valid_voxels, 25)
        q3 = np.percentile(valid_voxels, 75)
        median = np.percentile(valid_voxels, 50)
        iqr = q3 - q1
        
        if iqr < 1e-8:
            return image
            
        normalized = (image - median) / iqr
        return normalized


class ImageResizer:
    """
    Resize 3D MRI volumes to target dimensions
    """

    def __init__(self, target_shape: Tuple[int, int] = (224, 224)):
        """
        Args:
            target_shape (tuple): Target height and width (H, W)
        """
        self.target_shape = target_shape

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions using bilinear interpolation
        
        Args:
            image (np.ndarray): Input image of shape (D, H, W) or (H, W)
            
        Returns:
            np.ndarray: Resized image
        """
        # Convert to tensor for interpolation
        if image.ndim == 3:  # (D, H, W)
            # Process each slice independently
            tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, D, H, W)
            resized = F.interpolate(
                tensor, 
                size=self.target_shape, 
                mode='bilinear', 
                align_corners=False
            )
            return resized.squeeze(0).numpy()
        elif image.ndim == 2:  # (H, W)
            tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            resized = F.interpolate(
                tensor, 
                size=self.target_shape, 
                mode='bilinear', 
                align_corners=False
            )
            return resized.squeeze(0).squeeze(0).numpy()
        else:
            raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


class DataAugmentor:
    """
    Apply various augmentation techniques to MRI images
    """

    def __init__(self, config: dict, p_augment: float = 0.5):
        """
        Args:
            config (dict): Augmentation configuration dictionary
            p_augment (float): Probability of applying augmentation
        """
        self.config = config
        self.p_augment = p_augment

    def augment(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to image and mask
        
        Args:
            image (torch.Tensor): Input image of shape (C, H, W)
            mask (torch.Tensor): Segmentation mask of shape (1, H, W)
            
        Returns:
            tuple: Augmented (image, mask)
        """
        if np.random.rand() > self.p_augment:
            return image, mask

        # Random rotation
        if np.random.rand() < self.config["flip_probability"]:
            image, mask = self._random_rotation(image, mask)

        # Horizontal flip
        if np.random.rand() < self.config["flip_probability"]:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        # Vertical flip
        if np.random.rand() < self.config["flip_probability"]:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        # Elastic deformation
        if np.random.rand() < self.config["elastic_deformation"]["probability"]:
            image, mask = self._elastic_deformation(image, mask)

        # Gaussian noise
        if np.random.rand() < self.config["gaussian_noise"]["probability"]:
            image = self._add_gaussian_noise(image)

        # Brightness/Contrast
        if np.random.rand() < self.config["brightness_contrast"]["probability"]:
            image = self._adjust_brightness_contrast(image)

        return image, mask

    def _random_rotation(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random rotation in 2D plane
        """
        angle = np.random.uniform(-self.config["rotation_degrees"], self.config["rotation_degrees"])
        angle_rad = np.radians(angle)

        # Create rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = torch.tensor(
            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=image.dtype
        )

        # Apply rotation to image
        image_np = image.numpy()
        mask_np = mask.numpy()

        rotated_image = np.zeros_like(image_np)
        rotated_mask = np.zeros_like(mask_np)

        for c in range(image_np.shape[0]):
            rotated_image[c] = ndimage.rotate(image_np[c], angle, reshape=False, order=1)

        for c in range(mask_np.shape[0]):
            rotated_mask[c] = ndimage.rotate(mask_np[c], angle, reshape=False, order=0)

        return torch.from_numpy(rotated_image).float(), torch.from_numpy(rotated_mask).long()

    def _elastic_deformation(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Elastic deformation using Gaussian displacement fields
        """
        sigma = self.config["elastic_deformation"]["sigma"]
        alpha = self.config["elastic_deformation"]["alpha"]

        h, w = image.shape[-2:]

        # Generate random displacement fields
        dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
        dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        x_deformed = np.clip(x + dx, 0, h - 1)
        y_deformed = np.clip(y + dy, 0, w - 1)

        # Apply deformation to image
        image_np = image.numpy()
        deformed_image = np.zeros_like(image_np)
        for c in range(image_np.shape[0]):
            deformed_image[c] = map_coordinates(image_np[c], [x_deformed, y_deformed], order=1)

        # Apply deformation to mask
        mask_np = mask.numpy()
        deformed_mask = np.zeros_like(mask_np)
        for c in range(mask_np.shape[0]):
            deformed_mask[c] = map_coordinates(mask_np[c], [x_deformed, y_deformed], order=0)

        return torch.from_numpy(deformed_image).float(), torch.from_numpy(deformed_mask).long()

    def _add_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to image
        """
        noise_config = self.config["gaussian_noise"]
        mean = noise_config["mean"]
        std = noise_config["std"]

        noise = torch.randn_like(image) * std + mean
        return image + noise

    def _adjust_brightness_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adjust brightness and contrast
        """
        bc_config = self.config["brightness_contrast"]
        
        brightness_delta = np.random.uniform(
            -bc_config["brightness_delta"], 
            bc_config["brightness_delta"]
        )
        contrast_delta = np.random.uniform(
            1 - bc_config["contrast_delta"], 
            1 + bc_config["contrast_delta"]
        )

        # Adjust brightness and contrast
        return image * contrast_delta + brightness_delta
