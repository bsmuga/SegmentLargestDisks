import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


class SegmentationModule(L.LightningModule):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        encoder_weights: str | None,
        in_channels: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name, encoder_weights, in_channels, classes
        )
        self.classes = classes

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image, segmentation = batch
        segmentation = torch.permute(
            F.one_hot(segmentation, self.classes), (0, 3, 1, 2)
        )
        loss = F.mse_loss(self.model(image), segmentation)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image, segmentation = batch
        segmentation = torch.permute(
            F.one_hot(segmentation, self.classes), (0, 3, 1, 2)
        ).to(float)
        val_loss = F.mse_loss(self.model(image), segmentation)
        self.log("val_loss", val_loss)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image, segmentation = batch
        segmentation = torch.permute(
            F.one_hot(segmentation, self.classes), (0, 3, 1, 2)
        )
        test_loss = F.mse_loss(self.model(image), segmentation)
        self.log("test_loss", test_loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
