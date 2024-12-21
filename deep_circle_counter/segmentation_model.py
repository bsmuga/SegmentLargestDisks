import torch
import torch.nn.functional as F
import lightning as L


class SegmentationModel(L.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = torch.nn.Identity()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        image, segmentation = batch
        loss = F.mse_loss(self.model(image), segmentation)
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        image, segmentation = batch
        val_loss = F.mse_loss(self.model(image), segmentation)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        test_loss = F.mse_loss(self.model(x), x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
