"""
Dataset loader for skin lesion segmentation.

This module implements the data loading and preprocessing pipeline described
in Section 3.1 (Image Preprocessing) of the paper:
- CLAHE applied to L-channel of LAB color space (clip_limit=2.0, tile_grid_size=(8,8))
- Data augmentation: horizontal/vertical flip, shift-scale-rotate, brightness/contrast,
  grid distortion, elastic transform, coarse dropout
- ImageNet normalization
"""
import os
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, train_test_split


class SkinLesionDataset(Dataset):
    """
    Skin Lesion Dataset for binary segmentation.
    
    Preprocessing pipeline (Section 3.1):
    1. Resize to target size (192×192)
    2. Apply CLAHE to L-channel of LAB color space
    3. Data augmentation (training only)
    4. ImageNet normalization
    """

    def __init__(self, image_paths, mask_paths, img_size, mean, std, is_train=False,
                 clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            img_size: Target image size (height, width)
            mean: Normalization mean (ImageNet)
            std: Normalization std (ImageNet)
            is_train: Whether to apply data augmentation
            clahe_clip_limit: CLAHE clip limit (default: 2.0)
            clahe_tile_grid_size: CLAHE tile grid size (default: (8, 8))
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train
        self.img_size = img_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.transforms = self.setup_transforms(mean=mean, std=std)

    def __len__(self):
        return len(self.image_paths)

    def setup_transforms(self, mean, std):
        """
        Setup augmentation transforms.
        
        Training augmentations (Section 3.1):
        - Horizontal flip (p=0.5)
        - Vertical flip (p=0.5)
        - Shift-scale-rotate (scale±12%, rotate±15°, shift±12%, p=0.5)
        - Random brightness/contrast (p=0.5)
        - Grid distortion (p=0.5)
        - Elastic transform (α=1.0, σ=50, p=0.5)
        - Coarse dropout (p=0.5)
        
        All images:
        - ImageNet normalization
        """
        transforms = []

        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    scale_limit=0.12, 
                    rotate_limit=15, 
                    shift_limit=0.12, 
                    p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.GridDistortion(p=0.5),
                A.ElasticTransform(
                    alpha=1.0, 
                    sigma=50, 
                    alpha_affine=50, 
                    p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=self.img_size[1] // 20,
                    max_width=self.img_size[0] // 20,
                    min_holes=5,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5
                )
            ])

        transforms.extend([
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(always_apply=True),
        ])

        return A.Compose(transforms)

    def apply_clahe_lab(self, image):
        """
        Apply CLAHE to the L-channel of LAB color space.
        
        As described in Section 3.1 of the paper: CLAHE is applied to the
        L-channel of the LAB color space to enhance local contrast while
        avoiding noise amplification.
        
        Parameters:
            clip_limit: 2.0 (controls contrast limiting)
            tile_grid_size: (8, 8) (grid for histogram equalization)
        
        Args:
            image: RGB image (numpy array)
        
        Returns:
            Contrast-enhanced RGB image
        """
        # Convert RGB to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=self.clahe_tile_grid_size
        )
        l_channel = clahe.apply(l_channel)

        # Merge and convert back to RGB
        lab = cv2.merge([l_channel, a_channel, b_channel])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image

    def load_file(self, file_path, depth=0):
        """
        Load and preprocess image or mask file.
        
        Args:
            file_path: Path to file
            depth: cv2.IMREAD_COLOR for images, cv2.IMREAD_GRAYSCALE for masks
        
        Returns:
            Preprocessed image/mask
        """
        file = cv2.imread(file_path, depth)

        if depth == cv2.IMREAD_COLOR:
            file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

        resized_file = cv2.resize(
            file,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Apply CLAHE on L-channel of LAB color space (for color images only)
        if depth == cv2.IMREAD_COLOR:
            resized_file = self.apply_clahe_lab(resized_file)

        # Binarize masks
        if depth == cv2.IMREAD_GRAYSCALE:
            _, resized_file = cv2.threshold(resized_file, 127, 255, cv2.THRESH_BINARY)

        return resized_file

    def __getitem__(self, index):
        image = self.load_file(self.image_paths[index], depth=cv2.IMREAD_COLOR)
        mask = self.load_file(self.mask_paths[index], depth=cv2.IMREAD_GRAYSCALE)

        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"].to(torch.long)
        mask = (mask > 0).to(torch.long)

        return image, mask


def get_all_image_paths(config):
    """
    Collect all image and mask paths from both train and valid directories.
    
    Used for 5-fold cross-validation where we combine all available images
    and re-split them into folds.
    
    Args:
        config: Configuration object with TRAIN_IMAGES, TRAIN_MASKS, etc.
    
    Returns:
        Tuple of (image_paths, mask_paths) as numpy arrays
    """
    train_imgs = sorted(glob(config.TRAIN_IMAGES))
    train_msks = sorted(glob(config.TRAIN_MASKS))
    valid_imgs = sorted(glob(config.VALID_IMAGES))
    valid_msks = sorted(glob(config.VALID_MASKS))

    all_imgs = train_imgs + valid_imgs
    all_msks = train_msks + valid_msks

    print(f"Total images collected: {len(all_imgs)}")
    return np.array(all_imgs), np.array(all_msks)


def get_kfold_dataloaders(config, fold_idx, train_indices, test_indices, all_imgs, all_msks):
    """
    Create train, validation, and test dataloaders for a single fold.
    
    Protocol (Section 4):
    - Training portion (4 folds) is further split: 90% train, 10% validation
    - Validation set is used for early stopping and LR scheduling
    - Test set (1 held-out fold) is used only for final evaluation
    
    Args:
        config: Configuration object
        fold_idx: Current fold index
        train_indices: Indices for training (4 folds)
        test_indices: Indices for testing (1 held-out fold)
        all_imgs: All image paths
        all_msks: All mask paths
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_imgs = all_imgs[train_indices]
    train_msks = all_msks[train_indices]
    test_imgs = all_imgs[test_indices]
    test_msks = all_msks[test_indices]

    # Split training portion into train and validation for model selection
    # VAL_SPLIT = 0.1 means 10% for validation
    tr_imgs, val_imgs, tr_msks, val_msks = train_test_split(
        train_imgs, train_msks,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        shuffle=True
    )

    print(f"\nFold {fold_idx + 1}/{config.N_FOLDS}")
    print(f"  Train: {len(tr_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    # Get CLAHE parameters from config if available
    clahe_clip = getattr(config, 'CLAHE_CLIP_LIMIT', 2.0)
    clahe_grid = getattr(config, 'CLAHE_TILE_GRID_SIZE', (8, 8))

    train_dataset = SkinLesionDataset(
        image_paths=tr_imgs.tolist(),
        mask_paths=tr_msks.tolist(),
        img_size=config.IMAGE_SIZE,
        mean=config.MEAN,
        std=config.STD,
        is_train=True,
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid_size=clahe_grid
    )

    val_dataset = SkinLesionDataset(
        image_paths=val_imgs.tolist(),
        mask_paths=val_msks.tolist(),
        img_size=config.IMAGE_SIZE,
        mean=config.MEAN,
        std=config.STD,
        is_train=False,
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid_size=clahe_grid
    )

    test_dataset = SkinLesionDataset(
        image_paths=test_imgs.tolist(),
        mask_paths=test_msks.tolist(),
        img_size=config.IMAGE_SIZE,
        mean=config.MEAN,
        std=config.STD,
        is_train=False,
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid_size=clahe_grid
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_dataloaders(config):
    """
    Create train and validation dataloaders (single split, backward compatible).
    
    This function is provided for backward compatibility with non-CV training.
    For 5-fold cross-validation, use get_kfold_dataloaders instead.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (train_loader, valid_loader)
    """
    train_imgs = sorted(glob(config.TRAIN_IMAGES))
    train_msks = sorted(glob(config.TRAIN_MASKS))
    valid_imgs = sorted(glob(config.VALID_IMAGES))
    valid_msks = sorted(glob(config.VALID_MASKS))

    print(f"Found {len(train_imgs)} training images")
    print(f"Found {len(valid_imgs)} validation images")

    # Get CLAHE parameters from config if available
    clahe_clip = getattr(config, 'CLAHE_CLIP_LIMIT', 2.0)
    clahe_grid = getattr(config, 'CLAHE_TILE_GRID_SIZE', (8, 8))

    train_dataset = SkinLesionDataset(
        image_paths=train_imgs,
        mask_paths=train_msks,
        img_size=config.IMAGE_SIZE,
        mean=config.MEAN,
        std=config.STD,
        is_train=True,
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid_size=clahe_grid
    )

    valid_dataset = SkinLesionDataset(
        image_paths=valid_imgs,
        mask_paths=valid_msks,
        img_size=config.IMAGE_SIZE,
        mean=config.MEAN,
        std=config.STD,
        is_train=False,
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid_size=clahe_grid
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, valid_loader