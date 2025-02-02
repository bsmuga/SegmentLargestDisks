import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchmetrics import Dice


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
        self.train_dice = Dice(num_classes=classes)
        self.val_dice = Dice(num_classes=classes)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image, segmentation = batch
        preds = self.model(image)

        loss = F.mse_loss(
            preds,
            torch.permute(F.one_hot(segmentation, self.classes), (0, 3, 1, 2)).to(
                torch.float32
            ),
        )
        self.log("train_loss", loss)

        with torch.no_grad():
            self.train_dice(torch.argmax(preds, dim=1), segmentation)
            self.log("train_dice", self.train_dice, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image, segmentation = batch
        preds = self.model(image)

        val_loss = F.mse_loss(
            preds,
            torch.permute(F.one_hot(segmentation, self.classes), (0, 3, 1, 2)).to(
                torch.float32
            ),
        )
        self.log("val_loss", val_loss)

        with torch.no_grad():
            self.val_dice(torch.argmax(preds, dim=1), segmentation)
            self.log("val_dice", self.val_dice, on_step=False, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image, segmentation = batch
        test_loss = F.mse_loss(
            self.model(image),
            torch.permute(F.one_hot(segmentation, self.classes), (0, 3, 1, 2)).to(
                torch.float32
            ),
        )
        self.log("test_loss", test_loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
