"""
Dataset preparation script for ISIC 2018, PH2, and HAM10000 datasets
"""
import os
import shutil
import zipfile
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import numpy as np


class DatasetPreparation:
    """Base class for dataset preparation"""

    def __init__(self, dataset_name, raw_data_path, output_path, train_ratio=0.8):
        self.dataset_name = dataset_name
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.train_ratio = train_ratio

        # Create output directories
        self.train_images_dir = self.output_path / "train" / "images"
        self.train_masks_dir = self.output_path / "train" / "masks"
        self.valid_images_dir = self.output_path / "valid" / "images"
        self.valid_masks_dir = self.output_path / "valid" / "masks"

        for dir_path in [self.train_images_dir, self.train_masks_dir,
                         self.valid_images_dir, self.valid_masks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_mask(self, mask_path):
        """Process and binarize mask"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        # Binarize mask
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary_mask

    def copy_files(self, image_mask_pairs, destination_type='train'):
        """Copy image-mask pairs to destination"""
        if destination_type == 'train':
            img_dir = self.train_images_dir
            mask_dir = self.train_masks_dir
        else:
            img_dir = self.valid_images_dir
            mask_dir = self.valid_masks_dir

        print(f"Copying {len(image_mask_pairs)} {destination_type} samples...")
        for img_path, mask_path in tqdm(image_mask_pairs):
            # Copy image
            img_name = Path(img_path).name
            if not img_name.endswith('.png'):
                img_name = Path(img_path).stem + '.png'

            # Read and save image
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(img_dir / img_name), img)

            # Process and save mask
            mask = self.process_mask(mask_path)
            if mask is not None:
                mask_name = Path(mask_path).stem + '.png'
                cv2.imwrite(str(mask_dir / mask_name), mask)


class ISIC2018Preparation(DatasetPreparation):
    """Prepare ISIC 2018 dataset"""

    def prepare(self):
        """
        Expected structure for ISIC 2018:
        ISIC2018/
        ├── ISIC2018_Task1-2_Training_Input/
        │   └── *.jpg
        └── ISIC2018_Task1_Training_GroundTruth/
            └── *_segmentation.png
        """
        print(f"\n{'='*60}")
        print(f"Preparing ISIC 2018 Dataset")
        print(f"{'='*60}")

        # Find image and mask directories
        image_dir = self.raw_data_path / "ISIC2018_Task1-2_Training_Input"
        mask_dir = self.raw_data_path / "ISIC2018_Task1_Training_GroundTruth"

        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            print("Please download ISIC 2018 dataset from:")
            print("https://challenge.isic-archive.com/data/#2018")
            return False

        if not mask_dir.exists():
            print(f"❌ Mask directory not found: {mask_dir}")
            return False

        # Get image files
        image_files = sorted(list(image_dir.glob("*.jpg")))
        print(f"Found {len(image_files)} images")

        # Match images with masks
        image_mask_pairs = []
        for img_path in image_files:
            img_id = img_path.stem
            mask_path = mask_dir / f"{img_id}_segmentation.png"

            if mask_path.exists():
                image_mask_pairs.append((str(img_path), str(mask_path)))

        print(f"Found {len(image_mask_pairs)} image-mask pairs")

        if len(image_mask_pairs) == 0:
            print("❌ No valid image-mask pairs found!")
            return False

        # Split into train and validation
        train_pairs, valid_pairs = train_test_split(
            image_mask_pairs,
            train_size=self.train_ratio,
            random_state=42,
            shuffle=True
        )

        print(f"Train samples: {len(train_pairs)}")
        print(f"Valid samples: {len(valid_pairs)}")

        # Copy files
        self.copy_files(train_pairs, 'train')
        self.copy_files(valid_pairs, 'valid')

        print(f"✓ ISIC 2018 dataset prepared successfully!")
        print(f"Output directory: {self.output_path}")
        return True


class PH2Preparation(DatasetPreparation):
    """Prepare PH2 dataset"""

    def prepare(self):
        """
        Expected structure for PH2:
        PH2Dataset/
        └── images/
            ├── IMD*
            │   ├── IMD*_Dermoscopic_Image/*.bmp
            │   └── IMD*_lesion/*.bmp
        """
        print(f"\n{'='*60}")
        print(f"Preparing PH2 Dataset")
        print(f"{'='*60}")

        images_dir = self.raw_data_path / "images"

        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            print("Please download PH2 dataset from:")
            print("https://www.kaggle.com/datasets/athina123/ph2dataset")
            return False

        # Find all image folders
        image_folders = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        print(f"Found {len(image_folders)} image folders")
        

        image_mask_pairs = []

        for folder in image_folders:
            folder_name = folder.name

            # Find dermoscopic image
            dermoscopic_dir = folder / f"{folder_name}_Dermoscopic_Image"
            lesion_dir = folder / f"{folder_name}_lesion"

            if dermoscopic_dir.exists() and lesion_dir.exists():
                # Get image file
                img_files = list(dermoscopic_dir.glob("*.bmp"))
                mask_files = list(lesion_dir.glob("*_lesion.bmp"))

                if img_files and mask_files:
                    image_mask_pairs.append((str(img_files[0]), str(mask_files[0])))

        print(f"Found {len(image_mask_pairs)} image-mask pairs")

        if len(image_mask_pairs) == 0:
            print("❌ No valid image-mask pairs found!")
            return False

        # Split into train and validation
        train_pairs, valid_pairs = train_test_split(
            image_mask_pairs,
            train_size=self.train_ratio,
            random_state=42,
            shuffle=True
        )

        print(f"Train samples: {len(train_pairs)}")
        print(f"Valid samples: {len(valid_pairs)}")

        # Copy files
        self.copy_files(train_pairs, 'train')
        self.copy_files(valid_pairs, 'valid')

        print(f"✓ PH2 dataset prepared successfully!")
        print(f"Output directory: {self.output_path}")
        return True


class HAM10000Preparation(DatasetPreparation):
    """Prepare HAM10000 dataset"""

    def prepare(self):
        """
        Expected structure for HAM10000:
        HAM10000/
        ├── images/
        │   └── *.jpg
        └── masks/
            └── *.png
        """
        print(f"\n{'='*60}")
        print(f"Preparing HAM10000 Dataset")
        print(f"{'='*60}")

        images_dir = self.raw_data_path / "images"
        masks_dir = self.raw_data_path / "masks"

        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            print("Please download HAM10000 dataset from:")
            print("https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification")
            return False

        if not masks_dir.exists():
            print(f"❌ Masks directory not found: {masks_dir}")
            return False

        # Get all images
        image_files = sorted(list(images_dir.glob("*.jpg")))
        print(f"Found {len(image_files)} images")

        # Match with masks
        image_mask_pairs = []
        for img_path in image_files:
            img_id = img_path.stem
            mask_path = masks_dir / f"{img_id}.png"

            if mask_path.exists():
                image_mask_pairs.append((str(img_path), str(mask_path)))

        print(f"Found {len(image_mask_pairs)} image-mask pairs")

        if len(image_mask_pairs) == 0:
            print("❌ No valid image-mask pairs found!")
            return False

        # Split into train and validation
        train_pairs, valid_pairs = train_test_split(
            image_mask_pairs,
            train_size=self.train_ratio,
            random_state=42,
            shuffle=True
        )

        print(f"Train samples: {len(train_pairs)}")
        print(f"Valid samples: {len(valid_pairs)}")

        # Copy files
        self.copy_files(train_pairs, 'train')
        self.copy_files(valid_pairs, 'valid')

        print(f"✓ HAM10000 dataset prepared successfully!")
        print(f"Output directory: {self.output_path}")
        return True


def prepare_dataset(dataset_name, raw_data_path, output_path, train_ratio=0.8):
    """
    Prepare a specific dataset

    Args:
        dataset_name: 'isic2018', 'ph2', or 'ham10000'
        raw_data_path: Path to raw dataset
        output_path: Path to save processed dataset
        train_ratio: Ratio of training samples (default: 0.8)
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'isic2018':
        preparer = ISIC2018Preparation(dataset_name, raw_data_path, output_path, train_ratio)
    elif dataset_name == 'ph2':
        preparer = PH2Preparation(dataset_name, raw_data_path, output_path, train_ratio)
    elif dataset_name == 'ham10000':
        preparer = HAM10000Preparation(dataset_name, raw_data_path, output_path, train_ratio)
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        print("Available datasets: isic2018, ph2, ham10000")
        return False

    return preparer.prepare()


def main():
    """Main function with menu"""
    print("\n" + "="*60)
    print("Dataset Preparation for MRLSANet")
    print("="*60)

    print("\nAvailable datasets:")
    print("1. ISIC 2018")
    print("2. PH2")
    print("3. HAM10000")
    print("4. All datasets")

    choice = input("\nSelect dataset (1-4): ").strip()

    # Get paths
    print("\nEnter paths:")
    if choice == '4':
        print("You'll be prompted for each dataset separately")
    else:
        raw_path = input("Raw data path: ").strip()
        output_path = input("Output path (default: ../datasets/<dataset_name>): ").strip()
        if not output_path:
            # Will be set based on dataset choice
            output_path = None

        train_ratio = input("Train ratio (default: 0.8): ").strip()
        train_ratio = float(train_ratio) if train_ratio else 0.8

    # Process based on choice
    if choice == '1':
        if not output_path:
            output_path = os.path.join("..", "datasets", "ISIC2018")
        prepare_dataset('isic2018', raw_path, output_path, train_ratio)
    elif choice == '2':
        if not output_path:
            output_path = os.path.join("..", "datasets", "PH2")
        prepare_dataset('ph2', raw_path, output_path, train_ratio)
    elif choice == '3':
        if not output_path:
            output_path = os.path.join("..", "datasets", "HAM10000")
        prepare_dataset('ham10000', raw_path, output_path, train_ratio)
    elif choice == '4':
        datasets = [
            ('isic2018', 'ISIC 2018', 'ISIC2018'),
            ('ph2', 'PH2', 'PH2'),
            ('ham10000', 'HAM10000', 'HAM10000')
        ]

        for dataset_id, dataset_display, folder_name in datasets:
            print(f"\n{'='*60}")
            print(f"Preparing {dataset_display}")
            print(f"{'='*60}")

            raw_path = input(f"Raw data path for {dataset_display}: ").strip()
            output_path = input(f"Output path (default: ../datasets/{folder_name}): ").strip()
            if not output_path:
                output_path = os.path.join("..", "datasets", folder_name)

            prepare_dataset(dataset_id, raw_path, output_path, 0.8)
    else:
        print("Invalid choice!")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    # Example usage:
    # python prepare_datasets.py

    # Or programmatically:
    # prepare_dataset('isic2018', 'path/to/raw/ISIC2018', 'preprocessed_isic2018')
    # prepare_dataset('ph2', 'path/to/raw/PH2Dataset', 'preprocessed_ph2')
    # prepare_dataset('ham10000', 'path/to/raw/HAM10000', 'preprocessed_ham10000')

    main()
