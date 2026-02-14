"""Vision Transformer (ViT) segmentation model implemented with Flax NNX.

Memory-safe for 4GB VRAM (~2.3M params):
- Patch size: 16x16 → 64 tokens from 128x128 input
- Embed dim: 192, Heads: 4, Layers: 4, MLP ratio: 4
- Decoder: 4x ConvTranspose stages (8→16→32→64→128)
"""

import jax.numpy as jnp
from flax import nnx


class PatchEmbed(nnx.Module):
    """Convert image into a sequence of patch embeddings using a strided convolution."""

    def __init__(self, patch_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Conv(
            1, embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x):
        # x: (B, H, W, 1) → (B, H/P, W/P, embed_dim)
        x = self.proj(x)
        B, H, W, C = x.shape
        # Flatten spatial → sequence: (B, num_patches, embed_dim)
        return x.reshape(B, H * W, C)


class TransformerBlock(nnx.Module):
    """Standard pre-norm Transformer block: LayerNorm → MHSA → residual → LayerNorm → MLP → residual."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        mlp_hidden = embed_dim * mlp_ratio
        self.mlp_dense1 = nnx.Linear(embed_dim, mlp_hidden, rngs=rngs)
        self.mlp_dense2 = nnx.Linear(mlp_hidden, embed_dim, rngs=rngs)

    def __call__(self, x):
        # Self-attention with pre-norm
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        # MLP with pre-norm
        h = self.norm2(x)
        h = nnx.gelu(self.mlp_dense1(h))
        h = self.mlp_dense2(h)
        x = x + h
        return x


class ViT(nnx.Module):
    """Vision Transformer for semantic segmentation.

    Encoder: patch embedding + positional embedding + N transformer blocks.
    Decoder: reshape to spatial grid + 4x ConvTranspose upsample stages.

    Input:  (B, H, W, 1)  — H, W must be divisible by patch_size
    Output: (B, H, W, num_classes) logits
    """

    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        patch_size = 16
        embed_dim = 192
        num_heads = 4
        num_layers = 4
        mlp_ratio = 4

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size, embed_dim, rngs=rngs)

        # Learnable positional embedding (max 256 patches = 256x256 input with patch_size=16)
        self.pos_embed = nnx.Param(
            jnp.zeros((1, 256, embed_dim)),
        )

        # Transformer encoder
        self.blocks = nnx.List([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, rngs=rngs)
            for _ in range(num_layers)
        ])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)

        # Convolutional decoder: upsample from (H/16, W/16) back to (H, W)
        # 4 stages, each doubles spatial resolution
        # Stage 1: embed_dim → 128
        self.up1 = nnx.ConvTranspose(embed_dim, 128, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.up1_bn = nnx.BatchNorm(128, rngs=rngs)
        # Stage 2: 128 → 64
        self.up2 = nnx.ConvTranspose(128, 64, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.up2_bn = nnx.BatchNorm(64, rngs=rngs)
        # Stage 3: 64 → 32
        self.up3 = nnx.ConvTranspose(64, 32, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.up3_bn = nnx.BatchNorm(32, rngs=rngs)
        # Stage 4: 32 → 16
        self.up4 = nnx.ConvTranspose(32, 16, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.up4_bn = nnx.BatchNorm(16, rngs=rngs)

        # Segmentation head
        self.head = nnx.Conv(16, num_classes, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x):
        B, H, W, _ = x.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        num_patches = grid_h * grid_w

        # Patch embed → (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Add positional embedding (slice to actual number of patches)
        x = x + self.pos_embed[...][:, :num_patches, :]

        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Reshape to spatial grid: (B, grid_h, grid_w, embed_dim)
        x = x.reshape(B, grid_h, grid_w, self.embed_dim)

        # Decoder: 4x upsample (each stage doubles resolution)
        x = nnx.relu(self.up1_bn(self.up1(x)))
        x = nnx.relu(self.up2_bn(self.up2(x)))
        x = nnx.relu(self.up3_bn(self.up3(x)))
        x = nnx.relu(self.up4_bn(self.up4(x)))

        # Head
        return self.head(x)
