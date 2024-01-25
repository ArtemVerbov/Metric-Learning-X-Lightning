from typing import List
import random
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchvision.utils import make_grid

from src.transforms import inv_trans


class VisualizeTriplets(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.list_of_triplets = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: List[Tensor],
        batch_idx: int,
    ) -> None:

        random_index = random.randint(0, len(outputs['triplets'][0]))
        anchor_index = outputs['triplets'][0][random_index].item()
        positive_index = outputs['triplets'][1][random_index].item()
        negative_index = outputs['triplets'][2][random_index].item()

        anchors = batch[0][anchor_index]
        positive = batch[0][positive_index]
        negative = batch[0][negative_index]

        self.list_of_triplets.extend([anchors, positive, negative])

    def on_train_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            visualizations = [inv_trans(img) for img in self.list_of_triplets]
            grid = make_grid(visualizations, normalize=False, nrow=3)
            self.list_of_triplets.clear()
            trainer.logger.experiment.add_image(
                'Triplets preview. (Anchor, Positive, Negative)',
                img_tensor=grid,
                global_step=trainer.current_epoch,
            )
