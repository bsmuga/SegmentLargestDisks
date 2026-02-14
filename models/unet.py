"""UNet segmentation model implemented with Flax NNX."""

from flax import nnx
import jax.numpy as jnp


class DoubleConv(nnx.Module):
    """Two consecutive Conv → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = nnx.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nnx.Module):
    """DoubleConv followed by 2×2 MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.double_conv = DoubleConv(in_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        features = self.double_conv(x)
        pooled = nnx.max_pool(features, window_shape=(2, 2), strides=(2, 2))
        return pooled, features


class DecoderBlock(nnx.Module):
    """ConvTranspose upsample → concat skip → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.up_conv = nnx.ConvTranspose(
            in_channels, out_channels, kernel_size=(2, 2), strides=(2, 2), rngs=rngs,
        )
        # After concat: out_channels (from up) + out_channels (from skip)
        self.double_conv = DoubleConv(out_channels * 2, out_channels, rngs=rngs)

    def __call__(self, x, skip):
        x = self.up_conv(x)
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.double_conv(x)
        return x


class UNet(nnx.Module):
    """Standard UNet for semantic segmentation.

    Channel progression: 32 → 64 → 128 → 256 → 512 (bottleneck) → 256 → 128 → 64 → 32.
    Input:  (B, H, W, 1)
    Output: (B, H, W, num_classes) logits
    """

    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        channels = [32, 64, 128, 256]

        # Encoder
        self.enc1 = EncoderBlock(1, channels[0], rngs=rngs)
        self.enc2 = EncoderBlock(channels[0], channels[1], rngs=rngs)
        self.enc3 = EncoderBlock(channels[1], channels[2], rngs=rngs)
        self.enc4 = EncoderBlock(channels[2], channels[3], rngs=rngs)

        # Bottleneck
        self.bottleneck = DoubleConv(channels[3], 512, rngs=rngs)

        # Decoder
        self.dec4 = DecoderBlock(512, channels[3], rngs=rngs)
        self.dec3 = DecoderBlock(channels[3], channels[2], rngs=rngs)
        self.dec2 = DecoderBlock(channels[2], channels[1], rngs=rngs)
        self.dec1 = DecoderBlock(channels[1], channels[0], rngs=rngs)

        # Segmentation head
        self.head = nnx.Conv(channels[0], num_classes, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x):
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Head
        return self.head(x)
