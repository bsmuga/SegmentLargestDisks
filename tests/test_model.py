"""Tests for UNet model architecture."""

import jax.numpy as jnp
import pytest
from flax import nnx

from model import DoubleConv, DecoderBlock, EncoderBlock, UNet

RNGS = nnx.Rngs(0)


class TestDoubleConv:
    def test_output_shape(self):
        block = DoubleConv(1, 32, rngs=RNGS)
        x = jnp.zeros((2, 64, 64, 1))
        out = block(x)
        assert out.shape == (2, 64, 64, 32)

    def test_preserves_spatial_dims(self):
        block = DoubleConv(16, 64, rngs=RNGS)
        x = jnp.ones((1, 128, 128, 16))
        out = block(x)
        assert out.shape[1:3] == (128, 128)

    def test_different_channel_sizes(self):
        for in_c, out_c in [(1, 32), (32, 64), (64, 128), (256, 512)]:
            block = DoubleConv(in_c, out_c, rngs=RNGS)
            x = jnp.zeros((1, 16, 16, in_c))
            assert block(x).shape == (1, 16, 16, out_c)


class TestEncoderBlock:
    def test_output_shape_halved(self):
        block = EncoderBlock(1, 32, rngs=RNGS)
        x = jnp.zeros((2, 64, 64, 1))
        pooled, skip = block(x)
        assert pooled.shape == (2, 32, 32, 32)
        assert skip.shape == (2, 64, 64, 32)

    def test_skip_has_full_resolution(self):
        block = EncoderBlock(32, 64, rngs=RNGS)
        x = jnp.zeros((1, 128, 128, 32))
        pooled, skip = block(x)
        assert skip.shape[1:3] == x.shape[1:3]
        assert pooled.shape[1] == x.shape[1] // 2
        assert pooled.shape[2] == x.shape[2] // 2


class TestDecoderBlock:
    def test_output_shape_doubled(self):
        block = DecoderBlock(512, 256, rngs=RNGS)
        x = jnp.zeros((2, 16, 16, 512))
        skip = jnp.zeros((2, 32, 32, 256))
        out = block(x, skip)
        assert out.shape == (2, 32, 32, 256)

    def test_all_decoder_levels(self):
        configs = [(512, 256, 16), (256, 128, 32), (128, 64, 64), (64, 32, 128)]
        for in_c, out_c, spatial in configs:
            block = DecoderBlock(in_c, out_c, rngs=RNGS)
            x = jnp.zeros((1, spatial, spatial, in_c))
            skip = jnp.zeros((1, spatial * 2, spatial * 2, out_c))
            out = block(x, skip)
            assert out.shape == (1, spatial * 2, spatial * 2, out_c)


class TestUNet:
    def test_output_shape_default(self):
        model = UNet(num_classes=6, rngs=RNGS)
        x = jnp.zeros((1, 256, 256, 1))
        out = model(x)
        assert out.shape == (1, 256, 256, 6)

    def test_output_shape_different_num_classes(self):
        for nc in [2, 3, 10]:
            model = UNet(num_classes=nc, rngs=RNGS)
            x = jnp.zeros((1, 64, 64, 1))
            assert model(x).shape == (1, 64, 64, nc)

    def test_batch_dimension(self):
        model = UNet(num_classes=6, rngs=RNGS)
        for bs in [1, 4]:
            x = jnp.zeros((bs, 64, 64, 1))
            assert model(x).shape[0] == bs

    def test_output_is_logits_not_probabilities(self):
        """Logits should span negative and positive values, not be in [0,1]."""
        model = UNet(num_classes=6, rngs=RNGS)
        x = jnp.ones((1, 64, 64, 1))
        out = model(x)
        assert float(jnp.min(out)) < 0.0 or float(jnp.max(out)) > 1.0

    def test_different_spatial_sizes(self):
        """UNet should work with any spatial size divisible by 16."""
        model = UNet(num_classes=6, rngs=RNGS)
        for size in [64, 128, 256]:
            x = jnp.zeros((1, size, size, 1))
            out = model(x)
            assert out.shape == (1, size, size, 6)

    def test_train_eval_modes(self):
        """BatchNorm should behave differently in train vs eval mode."""
        model = UNet(num_classes=6, rngs=RNGS)
        x = jnp.ones((2, 64, 64, 1))

        model.train()
        out_train = model(x)

        model.eval()
        out_eval = model(x)

        # Outputs differ because BatchNorm uses batch stats in train, running stats in eval
        assert not jnp.allclose(out_train, out_eval)

    def test_deterministic_forward(self):
        """Same input should produce same output in eval mode."""
        model = UNet(num_classes=6, rngs=RNGS)
        model.eval()
        x = jnp.ones((1, 64, 64, 1))
        out1 = model(x)
        out2 = model(x)
        assert jnp.allclose(out1, out2)

    def test_has_expected_submodules(self):
        model = UNet(num_classes=6, rngs=RNGS)
        assert isinstance(model.enc1, EncoderBlock)
        assert isinstance(model.enc4, EncoderBlock)
        assert isinstance(model.bottleneck, DoubleConv)
        assert isinstance(model.dec1, DecoderBlock)
        assert isinstance(model.dec4, DecoderBlock)
        assert isinstance(model.head, nnx.Conv)
