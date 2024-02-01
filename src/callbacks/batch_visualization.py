from typing import List

from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchvision.utils import make_grid

from src.transforms import inv_trans


class VisualizeBatch(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.batch = None

    def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Tensor,
            batch: List[Tensor],
            batch_idx: int,
    ) -> None:
        if trainer.is_last_batch:
            self.batch = batch[0]

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            batch_visualizations = [inv_trans(img) for img in self.batch]

            self._visualization(trainer, 'Batch preview', batch_visualizations)

    @staticmethod
    def _visualization(trainer: Trainer, name: str, image_list: List) -> None:
        grid = make_grid(image_list, normalize=False)
        trainer.logger.experiment.add_image(
            name,
            img_tensor=grid,
            global_step=trainer.current_epoch,
        )
