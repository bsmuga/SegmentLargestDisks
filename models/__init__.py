"""Models package — registry and factory for segmentation architectures."""

from models.unet import DoubleConv, DecoderBlock, EncoderBlock, UNet
from models.vit import PatchEmbed, TransformerBlock, ViT

MODELS: dict[str, type] = {
    "unet": UNet,
    "vit": ViT,
}


def create_model(name: str, *, num_classes: int, rngs):
    """Instantiate a model by short name.

    Parameters
    ----------
    name : str
        One of the keys in ``MODELS`` (e.g. ``"unet"``, ``"vit"``).
    num_classes : int
        Number of output segmentation classes.
    rngs : nnx.Rngs
        Flax NNX RNG container.

    Returns
    -------
    nnx.Module
    """
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from {sorted(MODELS)}")
    return MODELS[key](num_classes=num_classes, rngs=rngs)
