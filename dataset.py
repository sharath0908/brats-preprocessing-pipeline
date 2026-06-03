"""
BraTS 2023 Dataset Class
Handles loading, preprocessing, and augmentation of multi-modal MRI data
"""

import os
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings

from preprocessing import IntensityNormalizer, ImageResizer, DataAugmentor
import config

warnings.filterwarnings("ignore")


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2023 multi-modal MRI data
    
    Expected directory structure:
    dataset_root/
    ├── patient_001/
    │   ├── patient_001_t1.nii.gz
    │   ├── patient_001_t1ce.nii.gz
    │   ├── patient_001_t2.nii.gz
    │   ├── patient_001_flair.nii.gz
    │   └── patient_001_seg.nii.gz
    ├── patient_002/
    ...
    """

    def __init__(
        self,
        root_dir: str,
        modalities: List[str] = None,
        target_shape: Tuple[int, int] = (224, 224),
        normalization_method: str = "zscore",
        augmentation_config: Dict = None,
        augment: bool = False,
        split: str = "train",
    ):
        """
        Args:
            root_dir (str): Path to dataset root directory
            modalities (list): List of MRI modalities to load (e.g., ['t1', 't1ce', 't2', 'flair'])
            target_shape (tuple): Target spatial dimensions (H, W)
            normalization_method (str): Normalization method ('zscore', 'minmax', 'robust')
            augmentation_config (dict): Data augmentation configuration
            augment (bool): Whether to apply augmentation
            split (str): Dataset split ('train', 'val', 'test')
        """
        self.root_dir = Path(root_dir)
        self.modalities = modalities or config.MODALITIES
        self.target_shape = target_shape
        self.normalization_method = normalization_method
        self.augment = augment
        self.split = split

        # Initialize preprocessing components
        self.normalizer = IntensityNormalizer(method=normalization_method)
        self.resizer = ImageResizer(target_shape=target_shape)
        
        augmentation_config = augmentation_config or config.AUGMENTATION_CONFIG
        self.augmentor = DataAugmentor(augmentation_config, p_augment=0.8)

        # Get list of patient directories
        self.patient_dirs = sorted([
            d for d in self.root_dir.iterdir() 
            if d.is_dir() and (d / f"{d.name}_t1.nii.gz").exists()
        ])

        if len(self.patient_dirs) == 0:
            raise ValueError(f"No patient data found in {root_dir}")

        print(f"Found {len(self.patient_dirs)} patients in {split} split")

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Load and preprocess a single patient's data
        
        Args:
            idx (int): Patient index
            
        Returns:
            tuple: (image_tensor, mask_tensor, metadata)
                - image_tensor: Shape (C, H, W) where C = number of modalities
                - mask_tensor: Shape (1, H, W)
                - metadata: Dict with patient info
        """
        patient_dir = self.patient_dirs[idx]
        patient_id = patient_dir.name

        try:
            # Load multi-modal images
            images = self._load_modalities(patient_dir, patient_id)
            
            # Load segmentation mask
            mask = self._load_segmentation(patient_dir, patient_id)
            
            # Preprocess images (normalize and resize)
            images = self._preprocess_images(images, mask)
            mask = self._preprocess_mask(mask)
            
            # Convert to tensors
            image_tensor = torch.from_numpy(images).float()  # (C, H, W)
            mask_tensor = torch.from_numpy(mask).long()      # (1, H, W)
            
            # Apply augmentation if training
            if self.augment and self.split == "train":
                image_tensor, mask_tensor = self.augmentor.augment(image_tensor, mask_tensor)
            
            # Metadata
            metadata = {
                "patient_id": patient_id,
                "modalities": self.modalities,
                "original_shape": images.shape,
                "split": self.split,
            }
            
            return image_tensor, mask_tensor, metadata

        except Exception as e:
            print(f"Error loading patient {patient_id}: {str(e)}")
            raise

    def _load_modalities(self, patient_dir: Path, patient_id: str) -> np.ndarray:
        """
        Load all modalities for a patient
        
        Args:
            patient_dir (Path): Patient directory path
            patient_id (str): Patient identifier
            
        Returns:
            np.ndarray: Stacked modalities of shape (C, H, W, D) where C = num modalities
        """
        modality_data = []
        
        for modality in self.modalities:
            # Construct file path
            filename = f"{patient_id}_{modality}.nii.gz"
            filepath = patient_dir / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"Missing {modality} for patient {patient_id}")
            
            # Load NIfTI file
            nifti_img = nib.load(str(filepath))
            data = np.array(nifti_img.dataobj, dtype=np.float32)
            
            # Ensure correct orientation (transpose if needed)
            if data.ndim == 3:
                # Standard BraTS orientation: (H, W, D)
                # Transpose to (D, H, W) for consistency
                data = np.transpose(data, (2, 0, 1))
            
            modality_data.append(data)
        
        # Stack modalities: (C, D, H, W)
        stacked = np.stack(modality_data, axis=0)
        return stacked

    def _load_segmentation(self, patient_dir: Path, patient_id: str) -> np.ndarray:
        """
        Load segmentation mask for a patient
        
        Args:
            patient_dir (Path): Patient directory path
            patient_id (str): Patient identifier
            
        Returns:
            np.ndarray: Segmentation mask of shape (D, H, W)
        """
        seg_filename = f"{patient_id}_seg.nii.gz"
        seg_filepath = patient_dir / seg_filename
        
        if not seg_filepath.exists():
            # Return zeros mask if segmentation not available
            print(f"Warning: Segmentation not found for {patient_id}, using zeros")
            return np.zeros((155, 240, 240), dtype=np.uint8)
        
        # Load NIfTI file
        nifti_seg = nib.load(str(seg_filepath))
        seg_data = np.array(nifti_seg.dataobj, dtype=np.uint8)
        
        # Transpose to (D, H, W)
        if seg_data.ndim == 3:
            seg_data = np.transpose(seg_data, (2, 0, 1))
        
        return seg_data

    def _preprocess_images(self, images: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess images: normalize and resize
        
        Args:
            images (np.ndarray): Stacked modalities of shape (C, D, H, W)
            mask (np.ndarray): Segmentation mask of shape (D, H, W)
            
        Returns:
            np.ndarray: Preprocessed images of shape (C, H, W)
        """
        processed = []
        
        for c in range(images.shape[0]):
            # Get modality data (D, H, W)
            modality_data = images[c]
            
            # Normalize intensity
            modality_data = self.normalizer.normalize(modality_data, mask)
            
            # For 3D data, we'll use middle slice(s) or max intensity projection
            # Option 1: Use middle slice
            middle_slice = modality_data.shape[0] // 2
            processed_2d = modality_data[middle_slice]
            
            # Resize to target shape
            processed_2d = self.resizer.resize(processed_2d)
            
            processed.append(processed_2d)
        
        # Stack channels: (C, H, W)
        return np.stack(processed, axis=0)

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess segmentation mask
        
        Args:
            mask (np.ndarray): Segmentation mask of shape (D, H, W)
            
        Returns:
            np.ndarray: Processed mask of shape (1, H, W)
        """
        # Get middle slice
        middle_slice = mask.shape[0] // 2
        processed_mask = mask[middle_slice]
        
        # Convert multi-class to binary (tumor vs background)
        # BraTS labels: 0=background, 1=necrotic core, 2=edema, 4=enhancing tumor
        binary_mask = (processed_mask > 0).astype(np.uint8)
        
        # Resize to target shape
        processed_mask = self.resizer.resize(binary_mask.astype(np.float32))
        processed_mask = (processed_mask > 0.5).astype(np.uint8)
        
        # Add channel dimension: (1, H, W)
        return np.expand_dims(processed_mask, axis=0)


class BraTSDatasetFull3D(Dataset):
    """
    BraTS Dataset that returns full 3D volumes instead of 2D slices
    Useful for 3D models
    """

    def __init__(
        self,
        root_dir: str,
        modalities: List[str] = None,
        target_shape: Tuple[int, int, int] = (155, 224, 224),
        normalization_method: str = "zscore",
        augmentation_config: Dict = None,
        augment: bool = False,
        split: str = "train",
    ):
        """
        Args:
            root_dir (str): Path to dataset root directory
            modalities (list): List of MRI modalities
            target_shape (tuple): Target spatial dimensions (D, H, W)
            normalization_method (str): Normalization method
            augmentation_config (dict): Data augmentation configuration
            augment (bool): Whether to apply augmentation
            split (str): Dataset split
        """
        self.root_dir = Path(root_dir)
        self.modalities = modalities or config.MODALITIES
        self.target_shape = target_shape
        self.normalization_method = normalization_method
        self.augment = augment
        self.split = split

        self.normalizer = IntensityNormalizer(method=normalization_method)
        self.resizer = ImageResizer(target_shape=(target_shape[1], target_shape[2]))

        self.patient_dirs = sorted([
            d for d in self.root_dir.iterdir() 
            if d.is_dir() and (d / f"{d.name}_t1.nii.gz").exists()
        ])

        if len(self.patient_dirs) == 0:
            raise ValueError(f"No patient data found in {root_dir}")

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Return full 3D volume"""
        patient_dir = self.patient_dirs[idx]
        patient_id = patient_dir.name

        try:
            # Load multi-modal 3D images
            images = []
            for modality in self.modalities:
                filepath = patient_dir / f"{patient_id}_{modality}.nii.gz"
                if not filepath.exists():
                    raise FileNotFoundError(f"Missing {modality}")
                
                nifti_img = nib.load(str(filepath))
                data = np.array(nifti_img.dataobj, dtype=np.float32)
                data = np.transpose(data, (2, 0, 1))  # (D, H, W)
                
                # Normalize
                data = self.normalizer.normalize(data)
                
                # Resize
                resized = np.zeros((data.shape[0], self.target_shape[1], self.target_shape[2]))
                for d in range(data.shape[0]):
                    resized[d] = self.resizer.resize(data[d])
                
                images.append(resized)
            
            images = np.stack(images, axis=0)  # (C, D, H, W)
            
            # Load mask
            seg_filepath = patient_dir / f"{patient_id}_seg.nii.gz"
            if seg_filepath.exists():
                mask = nib.load(str(seg_filepath))
                mask = np.array(mask.dataobj, dtype=np.uint8)
                mask = np.transpose(mask, (2, 0, 1))  # (D, H, W)
                mask = (mask > 0).astype(np.uint8)
                
                # Resize mask
                resized_mask = np.zeros((mask.shape[0], self.target_shape[1], self.target_shape[2]))
                for d in range(mask.shape[0]):
                    resized_mask[d] = self.resizer.resize(mask[d].astype(np.float32))
                mask = (resized_mask > 0.5).astype(np.uint8)
            else:
                mask = np.zeros((images.shape[1], self.target_shape[1], self.target_shape[2]), dtype=np.uint8)
            
            image_tensor = torch.from_numpy(images).float()  # (C, D, H, W)
            mask_tensor = torch.from_numpy(np.expand_dims(mask, 0)).long()  # (1, D, H, W)
            
            metadata = {
                "patient_id": patient_id,
                "modalities": self.modalities,
                "split": self.split,
            }
            
            return image_tensor, mask_tensor, metadata

        except Exception as e:
            print(f"Error loading patient {patient_id}: {str(e)}")
            raise
