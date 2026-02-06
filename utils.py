"""
Utility functions for training and evaluation
"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Dice, JaccardIndex


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


def tversky_bce_loss(predictions, targets, smooth=1e-8, alpha=0.5, beta=0.5):
    """
    Combined Tversky Loss and Binary Cross-Entropy Loss.
    
    As described in Section 3.4 of the paper: The Tversky loss extends the Dice loss
    by introducing tunable parameters that control the relative penalties assigned
    to false positives and false negatives.
    
    Default parameters: alpha=0.5, beta=0.5 (balanced FP/FN penalization, Dice-equivalent)
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth binary masks
        smooth: Smoothing factor for numerical stability (default: 1e-8)
        alpha: Weight for false positives (default: 0.5)
        beta: Weight for false negatives (default: 0.5)

    Returns:
        Combined loss value (Tversky + BCE)
    """
    # Ensure targets are float
    targets = targets.float()

    # Get probabilities
    probs = torch.sigmoid(predictions)

    # Calculate statistics
    dims = (0, 2, 3)
    TP = (probs * targets).sum(dims)
    FP = ((1 - targets) * probs).sum(dims)
    FN = (targets * (1 - probs)).sum(dims)

    # Tversky loss (Equation 18 in the paper)
    tversky_score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    tversky_loss = 1.0 - tversky_score.mean()

    # Binary cross-entropy loss (Equation 19 in the paper)
    bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='mean')

    # Combined loss (Equation 17 in the paper)
    return tversky_loss + bce_loss


class MetricTracker:
    """
    Track training and validation metrics.
    
    Tracks: Loss, Dice Coefficient (DC), and Jaccard Index (JI/IoU).
    Uses fixed threshold of 0.5 for binarization as described in Section 4.2.
    """

    def __init__(self, device='cuda', threshold=0.5):
        """
        Initialize metric tracker.
        
        Args:
            device: Device for metric computation
            threshold: Threshold for binary prediction (default: 0.5)
        """
        self.device = device
        self.threshold = threshold
        self.dice = Dice(num_classes=None, threshold=threshold, average="micro").to(device)
        self.iou = JaccardIndex(num_classes=1, threshold=threshold, task="binary").to(device)
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.loss_sum = 0.0
        self.count = 0
        self.dice.reset()
        self.iou.reset()

    def update(self, loss, predictions, targets):
        """Update metrics with batch results"""
        self.loss_sum += loss.item()
        self.count += 1

        # Convert predictions to probabilities
        probs = torch.sigmoid(predictions)

        self.dice.update(probs, targets)
        self.iou.update(probs, targets)

    def compute(self):
        """Compute average metrics"""
        avg_loss = self.loss_sum / self.count if self.count > 0 else 0.0
        dice_score = self.dice.compute().item()
        iou_score = self.iou.compute().item()

        return {
            'loss': avg_loss,
            'dice': dice_score,
            'iou': iou_score
        }


def save_checkpoint(model, optimizer, epoch, metrics, filepath, config=None):
    """
    Save model checkpoint with all necessary information for reproducibility.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        config: Optional config object for saving hyperparameters
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save config if provided
    if config is not None:
        checkpoint['config'] = {
            'image_size': config.IMAGE_SIZE,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'tversky_alpha': config.TVERSKY_ALPHA,
            'tversky_beta': config.TVERSKY_BETA,
            'backbone': config.BACKBONE
        }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, metrics


def denormalize(tensor, mean, std):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std + mean


def print_model_summary(model):
    """Print model parameter count and structure summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print(f"Model Summary")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"{'='*50}\n")
    
    return total_params, trainable_params