"""
Configuration file for MRLSANet training.

This configuration file contains all hyperparameters used in the experiments
as reported in the paper. Key parameters:
- Loss: Tversky (α=0.5, β=0.5) + BCE with equal weighting
- Input resolution: 192×192
- 5-fold cross-validation with 10% validation split within each fold
- Random seed: 42 for reproducibility

Reference: Section 4 (Experiments and Results) of the paper.
"""
import os


class Config:
    """
    Configuration class for MRLSA-Net training.
    
    All hyperparameters are explicitly documented for reproducibility
    as requested by reviewers (Comment 11).
    """
    
    # ==================== Dataset Configuration ====================
    DATASET_NAME = "ISIC2018"
    DATASET_PATH = os.path.join(os.getcwd(), "..", "datasets", DATASET_NAME)

    # Paths for single train/valid split (backward compatibility)
    TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images", "*.png")
    TRAIN_MASKS = os.path.join(DATASET_PATH, "train", "masks", "*.png")
    VALID_IMAGES = os.path.join(DATASET_PATH, "valid", "images", "*.png")
    VALID_MASKS = os.path.join(DATASET_PATH, "valid", "masks", "*.png")

    # ==================== Model Configuration ====================
    NUM_CLASSES = 1  # Binary segmentation (lesion vs background)
    IN_CHANNELS = 3  # RGB input
    BACKBONE = "timm-efficientnet-b5"  # Pre-trained encoder (noisy-student weights)

    # ==================== Image Configuration ====================
    # Input resolution: 192×192 (Section 3.1)
    # Justification: Balance between computational cost and segmentation quality
    IMAGE_SIZE = (192, 192)
    
    # ImageNet normalization statistics
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    
    # CLAHE preprocessing parameters (Section 3.1)
    # Applied to L-channel of LAB color space
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)

    # ==================== Training Configuration ====================
    BATCH_SIZE = 16
    NUM_EPOCHS = 150
    LEARNING_RATE = 1e-4  # Initial learning rate for Adam optimizer
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0  # Set to 0 for Windows compatibility; increase on Linux

    # ==================== Cross-validation Configuration ====================
    # 5-fold cross-validation (Section 4)
    N_FOLDS = 5
    # Validation split: 10% of training portion for early stopping/LR scheduling
    VAL_SPLIT = 0.1

    # ==================== Early Stopping Configuration ====================
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs

    # ==================== Loss Function Configuration ====================
    # Tversky loss parameters (Section 3.4, Equation 18)
    # α = β = 0.5: Balanced penalization of FP and FN (Dice-equivalent behavior)
    TVERSKY_ALPHA = 0.5  # Weight for false positives
    TVERSKY_BETA = 0.5   # Weight for false negatives
    
    # BCE loss weight (equal weighting with Tversky)
    BCE_WEIGHT = 1.0

    # ==================== Inference Configuration ====================
    # Fixed threshold for binary prediction (Section 4.2)
    INFERENCE_THRESHOLD = 0.5

    # ==================== Checkpoint Configuration ====================
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"

    # ==================== Device Configuration ====================
    DEVICE = "cuda"

    # ==================== Reproducibility ====================
    RANDOM_SEED = 42  # Fixed seed for all random operations
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("\n" + "="*60)
        print("MRLSA-Net Configuration")
        print("="*60)
        print(f"\nDataset: {cls.DATASET_NAME}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Backbone: {cls.BACKBONE}")
        print(f"\nTraining:")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Max Epochs: {cls.NUM_EPOCHS}")
        print(f"  Early Stopping Patience: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"\nLoss Parameters:")
        print(f"  Tversky α: {cls.TVERSKY_ALPHA}")
        print(f"  Tversky β: {cls.TVERSKY_BETA}")
        print(f"\nCross-validation:")
        print(f"  N Folds: {cls.N_FOLDS}")
        print(f"  Validation Split: {cls.VAL_SPLIT}")
        print(f"\nPreprocessing:")
        print(f"  CLAHE Clip Limit: {cls.CLAHE_CLIP_LIMIT}")
        print(f"  CLAHE Tile Grid: {cls.CLAHE_TILE_GRID_SIZE}")
        print(f"\nReproducibility:")
        print(f"  Random Seed: {cls.RANDOM_SEED}")
        print("="*60 + "\n")


# Dataset-specific configurations
class ISIC2018Config(Config):
    """Configuration for ISIC 2018 dataset."""
    DATASET_NAME = "ISIC2018"
    DATASET_PATH = os.path.join(os.getcwd(), "..", "datasets", "ISIC2018")
    TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images", "*.png")
    TRAIN_MASKS = os.path.join(DATASET_PATH, "train", "masks", "*.png")
    VALID_IMAGES = os.path.join(DATASET_PATH, "valid", "images", "*.png")
    VALID_MASKS = os.path.join(DATASET_PATH, "valid", "masks", "*.png")


class PH2Config(Config):
    """Configuration for PH2 dataset."""
    DATASET_NAME = "PH2"
    DATASET_PATH = os.path.join(os.getcwd(), "..", "datasets", "PH2")
    TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images", "*.png")
    TRAIN_MASKS = os.path.join(DATASET_PATH, "train", "masks", "*.png")
    VALID_IMAGES = os.path.join(DATASET_PATH, "valid", "images", "*.png")
    VALID_MASKS = os.path.join(DATASET_PATH, "valid", "masks", "*.png")


class HAM10000Config(Config):
    """
    Configuration for HAM10000 dataset.
    
    Note: HAM10000 is primarily a classification dataset. Segmentation masks
    are obtained from third-party sources (Kaggle dataset [38] in the paper).
    """
    DATASET_NAME = "HAM10000"
    DATASET_PATH = os.path.join(os.getcwd(), "..", "datasets", "HAM10000")
    TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images", "*.png")
    TRAIN_MASKS = os.path.join(DATASET_PATH, "train", "masks", "*.png")
    VALID_IMAGES = os.path.join(DATASET_PATH, "valid", "images", "*.png")
    VALID_MASKS = os.path.join(DATASET_PATH, "valid", "masks", "*.png")