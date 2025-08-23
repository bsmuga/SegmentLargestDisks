"""Simple example script to generate and visualize disk dataset."""

import argparse
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import DisksDataset
from src.confusion_matrix import ConfusionMatrix


def visualize_samples(dataset: DisksDataset, num_samples: int = 4) -> None:
    """Visualize samples from the dataset.
    
    Parameters
    ----------
    dataset : DisksDataset
        Dataset to visualize
    num_samples : int
        Number of samples to show
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        image, segmentation = dataset[i]
        
        # Remove batch dimension for visualization
        image = image.squeeze()
        
        # Plot image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Image')
        axes[i, 0].axis('off')
        
        # Plot segmentation
        im = axes[i, 1].imshow(segmentation, cmap='tab10', vmin=0, vmax=dataset.labeled_disks)
        axes[i, 1].set_title(f'Segmentation (top {dataset.labeled_disks} disks)')
        axes[i, 1].axis('off')
        
        # Plot overlay
        axes[i, 2].imshow(image, cmap='gray', alpha=0.7)
        masked_seg = np.ma.masked_where(segmentation == 0, segmentation)
        axes[i, 2].imshow(masked_seg, cmap='tab10', alpha=0.5, vmin=0, vmax=dataset.labeled_disks)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def demonstrate_confusion_matrix() -> None:
    """Demonstrate the confusion matrix calculation."""
    print("\n=== Confusion Matrix Demo ===")
    
    # Create a simple example
    cm = ConfusionMatrix(iou_threshold=0.5, num_classes=4)
    
    # Create synthetic predictions and targets
    target = np.zeros((100, 100), dtype=np.int64)
    target[20:40, 20:40] = 1  # Class 1
    target[60:80, 60:80] = 2  # Class 2
    
    # Perfect prediction for class 1, missed class 2, false positive class 3
    preds = np.zeros((100, 100), dtype=np.int64)
    preds[20:40, 20:40] = 1  # Correct class 1
    preds[70:90, 70:90] = 3  # False positive class 3
    
    result = cm(preds, target)
    
    print("\nConfusion Matrix Results:")
    print("Format: [TP, FP, FN, TN] for each class")
    for i in range(4):
        print(f"Class {i}: TP={result[i,0]:.0f}, FP={result[i,1]:.0f}, "
              f"FN={result[i,2]:.0f}, TN={result[i,3]:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Generate and visualize disk dataset")
    parser.add_argument("--image-size", type=int, nargs=2, default=[200, 150],
                        help="Image size (width, height)")
    parser.add_argument("--max-disks", type=int, default=10,
                        help="Maximum number of disks per image")
    parser.add_argument("--labeled-disks", type=int, default=3,
                        help="Number of largest disks to label")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--demo-confusion", action="store_true",
                        help="Demonstrate confusion matrix calculation")
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = DisksDataset(
        image_size=tuple(args.image_size),
        disk_max_num=args.max_disks,
        labeled_disks=args.labeled_disks,
        items=args.num_samples,
        seed=args.seed
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Image size: {args.image_size}")
    print(f"Max disks: {args.max_disks}")
    print(f"Labeled disks: {args.labeled_disks}")
    
    # Visualize samples
    visualize_samples(dataset, args.num_samples)
    
    # Demo confusion matrix if requested
    if args.demo_confusion:
        demonstrate_confusion_matrix()


if __name__ == "__main__":
    main()