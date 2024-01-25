from typing import TYPE_CHECKING, Dict, List, Optional

import timm
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric

from src.config import ModelConfig
from src.metrics import get_metrics

if TYPE_CHECKING:
    from pytorch_metric_learning.losses import BaseMetricLossFunction
    from pytorch_metric_learning.miners import BaseMiner
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


class MetricLearningModule(LightningModule):  # noqa: WPS214
    def __init__(
        self,
        model_cfg: ModelConfig,
        optimizer: 'Optimizer',
        loss: 'BaseMetricLossFunction',
        mining_func: 'BaseMiner',
        scheduler: Optional['LRScheduler'] = None,
    ):
        super().__init__()
        self.loss = loss
        self.mining_func = mining_func

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model_cfg = model_cfg
        self.metrics = get_metrics(
            include=self.model_cfg.metrics_include,
            k=self.model_cfg.knn,
        )

        self.model = timm.create_model(
            model_name=self.model_cfg.model,
            pretrained=self.model_cfg.pretrained,
            num_classes=self.model_cfg.embedding_size,
        )

        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()
        self._precision_at_one_val = MeanMetric()
        self._NMI_val = MeanMetric()
        self._precision_at_one_test = MeanMetric()
        self._NMI_test = MeanMetric()

        self.save_hyperparameters(ignore=['loss', 'mining_func'])

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: List[Tensor]) -> Dict:  # noqa: WPS210
        images, labels = batch
        embeddings = self.forward(images)
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss(embeddings, labels, indices_tuple)

        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': embeddings, 'labels': labels, 'triplets': indices_tuple}

    def validation_step(self, batch: List[Tensor], batch_index: int) -> None:  # noqa: WPS210
        images, labels = batch
        embeddings = self.forward(images)
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss(embeddings, labels, indices_tuple)

        self._valid_loss(loss)

        metrics = self.metrics.get_accuracy(
            embeddings,
            labels,
            reference=embeddings,
            reference_labels=labels,
            ref_includes_query=True,
        )

        self._NMI_val(metrics['NMI'])
        self._precision_at_one_val(metrics['precision_at_1'])

        self.log('val_loss', loss, on_step=False, prog_bar=False, logger=True)

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        images, labels = batch
        embeddings = self.forward(images)

        metrics = self.metrics.get_accuracy(
            embeddings,
            labels,
            reference=embeddings,
            reference_labels=labels,
            ref_includes_query=True,
        )
        self._NMI_test(metrics['NMI'])
        self._precision_at_one_test(metrics['precision_at_1'])

        return embeddings

    def on_train_epoch_end(self) -> None:
        self.log('mean_train_loss', self._train_loss, on_step=False, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:

        self.log('mean_valid_loss', self._valid_loss, on_step=False, prog_bar=True, on_epoch=True)
        self.log_dict(
            dictionary={
                'precision_at_1_val': self._precision_at_one_val,
                'NMI_val': self._NMI_val,
            },
            prog_bar=True,
            on_epoch=True,
        )

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            dictionary={
                'precision_at_1_test': self._precision_at_one_test,
                'NMI_test': self._NMI_test,
            },
            prog_bar=True,
            on_epoch=True,
        )

    # noinspection PyCallingNonCallable
    def configure_optimizers(self) -> Dict:
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler:
            scheduler = self.scheduler(optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': self.model_cfg.interval,
                    'frequency': self.model_cfg.optimizer_frequency,
                    'monitor': self.model_cfg.monitor,
                },
            }
        return {'optimizer': optimizer}
