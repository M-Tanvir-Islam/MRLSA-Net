"""
Training script for MRLSANet with 5-fold cross-validation.

This script implements the training protocol described in Section 4 of the paper:
- 5-fold cross-validation on ISIC-2018
- Within each fold: 4 folds for training (with 10% validation split), 1 fold for testing
- Early stopping with patience=10 based on validation Dice score
- Learning rate scheduling with ReduceLROnPlateau

Results are reported as mean ± standard deviation across all folds.
"""
import os
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

from config import Config
from models import get_model
from dataset import get_all_image_paths, get_kfold_dataloaders
from utils import tversky_bce_loss, MetricTracker, save_checkpoint, set_seed, print_model_summary


def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """
    Train model for one epoch.
    
    Args:
        model: MRLSA-Net model
        train_loader: Training data loader
        optimizer: Adam optimizer
        criterion: Combined Tversky + BCE loss
        device: Training device
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        Dictionary with training metrics (loss, dice, iou)
    """
    model.train()
    metrics = MetricTracker(device=device)

    pbar = tqdm(train_loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.unsqueeze(1).float().to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                predictions = model(images, mode="seg")
                loss = criterion(predictions, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images, mode="seg")
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()

        metrics.update(loss, predictions.detach(), masks.detach())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return metrics.compute()


def evaluate(model, data_loader, criterion, device, desc="Validation"):
    """
    Evaluate model on validation/test data.
    
    Args:
        model: MRLSA-Net model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Evaluation device
        desc: Description for progress bar
    
    Returns:
        Dictionary with evaluation metrics (loss, dice, iou)
    """
    model.eval()
    metrics = MetricTracker(device=device)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)

            predictions = model(images, mode="seg")
            loss = criterion(predictions, masks)

            metrics.update(loss, predictions, masks)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return metrics.compute()


def train_single_fold(config, fold_idx, train_loader, val_loader, test_loader, device):
    """
    Train and evaluate a single fold.
    
    Training protocol (Section 4):
    - Adam optimizer with lr=1e-4
    - ReduceLROnPlateau scheduler (patience=5, factor=0.1)
    - Early stopping (patience=10) based on validation Dice
    - Best model checkpoint saved per fold
    
    Args:
        config: Configuration object
        fold_idx: Current fold index (0-4)
        train_loader: Training data loader
        val_loader: Validation data loader (for early stopping)
        test_loader: Test data loader (held-out fold)
        device: Training device
    
    Returns:
        Dictionary with test metrics for this fold
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{config.N_FOLDS}")
    print(f"{'='*60}")

    fold_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"fold_{fold_idx + 1}")
    os.makedirs(fold_checkpoint_dir, exist_ok=True)

    # Initialize model
    model = get_model(config).to(device)
    print_model_summary(model)

    # Optimizer: Adam with lr=1e-4, weight_decay=1e-4
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Loss function: Tversky (α=0.5, β=0.5) + BCE
    criterion = lambda pred, tgt: tversky_bce_loss(
        pred, tgt, alpha=config.TVERSKY_ALPHA, beta=config.TVERSKY_BETA
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.1,
        verbose=True
    )

    # Mixed precision training (if CUDA)
    scaler = GradScaler() if config.DEVICE == "cuda" else None

    best_dice = 0.0
    best_model_path = None
    epochs_without_improvement = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 50)

        # Training
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validation (for early stopping and LR scheduling)
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Validation")

        # Update learning rate based on validation loss
        scheduler.step(val_metrics['loss'])

        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f} | "
              f"IoU: {train_metrics['iou']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f}")

        # Save best model based on validation Dice
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            epochs_without_improvement = 0
            best_model_path = os.path.join(
                fold_checkpoint_dir,
                f"best_model_dice_{best_dice:.4f}.pth"
            )
            save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path, config)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model and evaluate on held-out test fold
    print(f"\nEvaluating best model on held-out test fold...")
    if best_model_path is not None:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device, desc="Testing")

    print(f"\nFold {fold_idx + 1} Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f} | "
          f"Dice: {test_metrics['dice']:.4f} | "
          f"IoU: {test_metrics['iou']:.4f}")

    return test_metrics


def train_kfold(config):
    """
    5-fold cross-validation training.
    
    Protocol (Section 4):
    - Dataset split into 5 folds using KFold with shuffle=True, random_state=42
    - In each iteration: 4 folds for training, 1 fold for testing
    - Within training portion: 10% used for validation (early stopping)
    - Results reported as mean ± std across all 5 folds
    
    Args:
        config: Configuration object
    
    Returns:
        List of test metrics dictionaries for each fold
    """
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print configuration
    config.print_config()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print("Loading all image paths...")
    all_imgs, all_msks = get_all_image_paths(config)

    # 5-fold cross-validation with fixed random state for reproducibility
    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    fold_results = []

    for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(all_imgs)):
        train_loader, val_loader, test_loader = get_kfold_dataloaders(
            config, fold_idx, train_indices, test_indices, all_imgs, all_msks
        )

        test_metrics = train_single_fold(
            config, fold_idx, train_loader, val_loader, test_loader, device
        )
        fold_results.append(test_metrics)

    # Aggregate results across all folds
    print(f"\n{'='*60}")
    print(f"5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")

    all_dice = [r['dice'] for r in fold_results]
    all_iou = [r['iou'] for r in fold_results]
    all_loss = [r['loss'] for r in fold_results]

    for i, r in enumerate(fold_results):
        print(f"Fold {i + 1}: Dice={r['dice']:.4f} | IoU={r['iou']:.4f} | Loss={r['loss']:.4f}")

    print(f"\nMean ± Std:")
    print(f"  Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"  IoU:  {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print(f"  Loss: {np.mean(all_loss):.4f} ± {np.std(all_loss):.4f}")

    # Save aggregated results
    results_path = os.path.join(config.CHECKPOINT_DIR, "kfold_results.txt")
    with open(results_path, 'w') as f:
        f.write("5-FOLD CROSS-VALIDATION RESULTS\n")
        f.write("="*50 + "\n\n")
        for i, r in enumerate(fold_results):
            f.write(f"Fold {i + 1}: Dice={r['dice']:.4f} | IoU={r['iou']:.4f} | Loss={r['loss']:.4f}\n")
        f.write(f"\nMean ± Std:\n")
        f.write(f"  Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}\n")
        f.write(f"  IoU:  {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}\n")
        f.write(f"  Loss: {np.mean(all_loss):.4f} ± {np.std(all_loss):.4f}\n")
    print(f"\nResults saved to: {results_path}")

    return fold_results


if __name__ == "__main__":
    config = Config()
    train_kfold(config)