"""
Inference script for MRLSANet
"""
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from config import Config
from models import get_model
from utils import denormalize


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    model = get_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    return model


def apply_clahe_lab(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to the L-channel of LAB color space.

    Matches the preprocessing applied during training (Section 3.1 of the paper).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image


def preprocess_image(image_path, img_size, mean, std):
    """Load and preprocess a single image"""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    # Resize
    image = cv2.resize(image, (img_size[1], img_size[0]))

    # Apply CLAHE on L-channel of LAB color space (matching training preprocessing)
    image = apply_clahe_lab(image)

    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std

    # To tensor
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    return image, original_size


def predict(model, image_tensor, device):
    """Run inference on a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor, mode="seg")
        prediction = torch.sigmoid(prediction)

    return prediction


def visualize_results(original_img, pred_mask, save_path=None):
    """Visualize original image and predicted mask"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    # Overlay
    overlay = original_img.copy()
    mask_colored = np.zeros_like(original_img)
    mask_colored[:, :, 0] = pred_mask * 255  # Red channel
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to: {save_path}")

    plt.show()


def inference_single_image(model, image_path, config, device, save_dir=None):
    """Run inference on a single image"""
    # Preprocess
    image_tensor, original_size = preprocess_image(
        image_path, config.IMAGE_SIZE, config.MEAN, config.STD
    )

    # Predict
    prediction = predict(model, image_tensor, device)

    # Post-process
    pred_mask = prediction[0, 0].cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]))
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Load original image for visualization
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Visualize
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f"result_{img_name}")

    visualize_results(original_img, pred_mask, save_path)

    return pred_mask


def inference_batch(model, image_dir, config, device, save_dir="results"):
    """Run inference on all images in a directory"""
    image_paths = glob(os.path.join(image_dir, "*.png")) + \
                  glob(os.path.join(image_dir, "*.jpg"))

    print(f"Found {len(image_paths)} images")

    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        inference_single_image(model, img_path, config, device, save_dir)


def main():
    """Main inference function"""
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model_dice_*.pth")
    checkpoints = glob(checkpoint_path)

    if not checkpoints:
        print("No checkpoint found! Please train the model first.")
        return

    # Use the latest checkpoint
    checkpoint_path = sorted(checkpoints)[-1]
    model = load_model(checkpoint_path, config, device)

    # Example usage - single image
    # inference_single_image(model, "path/to/image.png", config, device, "results")

    # Example usage - batch processing
    valid_images_dir = os.path.join(config.DATASET_PATH, "valid", "images")
    if os.path.exists(valid_images_dir):
        # Process first 5 validation images
        image_paths = glob(os.path.join(valid_images_dir, "*.png"))[:5]
        for img_path in image_paths:
            inference_single_image(model, img_path, config, device, "results")
    else:
        print(f"Validation directory not found: {valid_images_dir}")


if __name__ == "__main__":
    main()