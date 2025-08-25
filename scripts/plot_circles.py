import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import DisksDataset


def plot_disks(size: tuple[int, int], num_points: int, seed: int = None) -> plt.Figure:
    """Generate and plot disks using optimized rendering"""
    # Create random generator
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    # Generate disks using the static method
    disks = DisksDataset.generate_disks(size, num_points, rng)
    
    values = [1] * len(disks)
    image = DisksDataset.disks2img(size, disks, values)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(image, cmap='viridis')
    ax.set_title(f'Generated {len(disks)} Non-overlapping Disks')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    fig.colorbar(cax, label='Disk presence')
    return fig


def plot_segmentation_example(size: tuple[int, int] = (300, 200), 
                            max_disks: int = 8, 
                            labeled_disks: int = 3,
                            seed: int = 42) -> plt.Figure:
    """Plot example of disk segmentation with labeled largest disks"""
    dataset = DisksDataset(size, max_disks, labeled_disks, 1, seed=seed)
    image, segmentation = dataset[0]
    
    image_np = image.squeeze()
    seg_np = segmentation
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title('Generated Image')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(seg_np, cmap='tab10', vmin=0, vmax=labeled_disks)
    axes[1].set_title(f'Segmentation (0=bg, 1-{labeled_disks}=largest)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], ticks=list(range(labeled_disks + 1)))
    
    axes[2].imshow(image_np, cmap='gray', alpha=0.7)
    masked_seg = np.ma.masked_where(seg_np == 0, seg_np)
    axes[2].imshow(masked_seg, cmap='tab10', alpha=0.6, vmin=0, vmax=labeled_disks)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_reproducibility_comparison(size: tuple[int, int] = (200, 150),
                                  max_disks: int = 6,
                                  labeled_disks: int = 3) -> plt.Figure:
    """Compare reproducibility with and without seed"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # With seed - should be identical
    seed = 42
    ds1 = DisksDataset(size, max_disks, labeled_disks, 1, seed=seed)
    ds2 = DisksDataset(size, max_disks, labeled_disks, 1, seed=seed)
    
    img1, seg1 = ds1[0]
    img2, seg2 = ds2[0]
    
    axes[0, 0].imshow(img1.squeeze(), cmap='gray')
    axes[0, 0].set_title('With seed=42 (Run 1)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2.squeeze(), cmap='gray')
    axes[0, 1].set_title('With seed=42 (Run 2)')
    axes[0, 1].axis('off')
    
    identical = np.array_equal(img1, img2)
    axes[0, 2].text(0.5, 0.5, f'Identical: {identical}', 
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if identical else "lightcoral"))
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Reproducibility Check')
    
    ds3 = DisksDataset(size, max_disks, labeled_disks, 1)
    ds4 = DisksDataset(size, max_disks, labeled_disks, 1)
    
    img3, seg3 = ds3[0]
    img4, seg4 = ds4[0]
    
    axes[1, 0].imshow(img3.squeeze(), cmap='gray')
    axes[1, 0].set_title('Without seed (Run 1)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img4.squeeze(), cmap='gray')
    axes[1, 1].set_title('Without seed (Run 2)')
    axes[1, 1].axis('off')
    
    different = not np.array_equal(img3, img4)
    axes[1, 2].text(0.5, 0.5, f'Different: {different}', 
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if different else "lightcoral"))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Randomness Check')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import os
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Generate all visualizations
    print("Generating disk visualization...")
    fig1 = plot_disks((300, 200), 8)
    fig1.savefig('images/disk_generation.png', dpi=300, bbox_inches='tight')
    
    print("Generating segmentation example...")
    fig2 = plot_segmentation_example()
    fig2.savefig('images/segmentation_example.png', dpi=300, bbox_inches='tight')
    
    print("Generating reproducibility comparison...")
    fig3 = plot_reproducibility_comparison()
    fig3.savefig('images/reproducibility_comparison.png', dpi=300, bbox_inches='tight')
    
    print("All visualizations saved to images/ directory!")
    
    # Show the plots
    plt.show()
