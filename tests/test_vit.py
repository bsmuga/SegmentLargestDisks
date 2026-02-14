"""Tests for ViT segmentation model architecture."""

import jax.numpy as jnp
import pytest
from flax import nnx

from models.vit import PatchEmbed, TransformerBlock, ViT

RNGS = nnx.Rngs(0)


class TestPatchEmbed:
    def test_output_shape(self):
        pe = PatchEmbed(patch_size=16, embed_dim=192, rngs=RNGS)
        x = jnp.zeros((2, 128, 128, 1))
        out = pe(x)
        # 128/16 = 8 → 8*8 = 64 patches
        assert out.shape == (2, 64, 192)

    def test_different_input_sizes(self):
        pe = PatchEmbed(patch_size=16, embed_dim=192, rngs=RNGS)
        for size, expected_patches in [(64, 16), (128, 64), (256, 256)]:
            x = jnp.zeros((1, size, size, 1))
            out = pe(x)
            assert out.shape == (1, expected_patches, 192)


class TestTransformerBlock:
    def test_output_shape_preserved(self):
        block = TransformerBlock(embed_dim=192, num_heads=4, mlp_ratio=4, rngs=RNGS)
        x = jnp.zeros((2, 64, 192))
        out = block(x)
        assert out.shape == (2, 64, 192)

    def test_different_sequence_lengths(self):
        block = TransformerBlock(embed_dim=192, num_heads=4, mlp_ratio=4, rngs=RNGS)
        for seq_len in [16, 64, 256]:
            x = jnp.zeros((1, seq_len, 192))
            out = block(x)
            assert out.shape == (1, seq_len, 192)


class TestViT:
    def test_output_shape_default(self):
        model = ViT(num_classes=6, rngs=RNGS)
        x = jnp.zeros((1, 128, 128, 1))
        out = model(x)
        assert out.shape == (1, 128, 128, 6)

    def test_output_shape_different_num_classes(self):
        for nc in [2, 3, 10]:
            model = ViT(num_classes=nc, rngs=RNGS)
            x = jnp.zeros((1, 128, 128, 1))
            assert model(x).shape == (1, 128, 128, nc)

    def test_batch_dimension(self):
        model = ViT(num_classes=6, rngs=RNGS)
        for bs in [1, 4]:
            x = jnp.zeros((bs, 128, 128, 1))
            assert model(x).shape[0] == bs

    def test_output_is_logits_not_probabilities(self):
        """Logits should span negative and positive values, not be in [0,1]."""
        model = ViT(num_classes=6, rngs=RNGS)
        x = jnp.ones((1, 128, 128, 1))
        out = model(x)
        assert float(jnp.min(out)) < 0.0 or float(jnp.max(out)) > 1.0

    def test_different_spatial_sizes(self):
        """ViT should work with any spatial size divisible by 16."""
        model = ViT(num_classes=6, rngs=RNGS)
        for size in [64, 128, 256]:
            x = jnp.zeros((1, size, size, 1))
            out = model(x)
            assert out.shape == (1, size, size, 6)

    def test_train_eval_modes(self):
        """BatchNorm should behave differently in train vs eval mode."""
        model = ViT(num_classes=6, rngs=RNGS)
        x = jnp.ones((2, 128, 128, 1))

        model.train()
        out_train = model(x)

        model.eval()
        out_eval = model(x)

        assert not jnp.allclose(out_train, out_eval)

    def test_deterministic_forward(self):
        """Same input should produce same output in eval mode."""
        model = ViT(num_classes=6, rngs=RNGS)
        model.eval()
        x = jnp.ones((1, 128, 128, 1))
        out1 = model(x)
        out2 = model(x)
        assert jnp.allclose(out1, out2)

    def test_constructor_contract_matches_unet(self):
        """ViT and UNet should have the same constructor signature."""
        from models import UNet
        unet = UNet(num_classes=6, rngs=RNGS)
        vit = ViT(num_classes=6, rngs=RNGS)
        # Both should accept the same input shape and produce same output shape
        x = jnp.zeros((1, 128, 128, 1))
        assert unet(x).shape == vit(x).shape
