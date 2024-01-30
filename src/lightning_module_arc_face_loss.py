from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import timm
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor
from torchmetrics import MeanMetric

from src.config import ModelConfig
from src.metrics import get_metrics

if TYPE_CHECKING:
    from pytorch_metric_learning.losses import BaseMetricLossFunction
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


class MetricLearningModule(LightningModule):  # noqa: WPS214
    def __init__(
        self,
        classes: Dict[str, int],
        model_cfg: ModelConfig,
        optimizer: 'Optimizer',
        loss: 'BaseMetricLossFunction',
        scheduler: Optional['LRScheduler'] = None,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.automatic_optimization = False

        self.loss = loss(
            num_classes=len(classes),
            embedding_size=self.model_cfg.embedding_size
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

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

        self.save_hyperparameters(ignore=['loss'])

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: List[Tensor]) -> Dict:  # noqa: WPS210

        optimizer, loss_optimizer = self.optimizers()
        optimizer.zero_grad()
        loss_optimizer.zero_grad()

        images, labels = batch
        embeddings = self.forward(images)
        loss = self.loss(embeddings, labels)

        self.manual_backward(loss)
        optimizer.step()
        loss_optimizer.step()

        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss, 'loss_optimizer': loss_optimizer, 'preds': embeddings, 'labels': labels}

    def validation_step(self, batch: List[Tensor], batch_index: int) -> None:  # noqa: WPS210
        with torch.no_grad():
            images, labels = batch
            embeddings = self.forward(images)
            loss = self.loss(embeddings, labels)

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

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Dict:
        with torch.no_grad():
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

        return {'embeddings': embeddings, 'labels': labels}

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
        self._scheduler_step()

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
    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        loss_optimizer = self.optimizer(params=self.loss.parameters())
        if self.scheduler:
            scheduler = self.scheduler(optimizer)

            return (
                {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': self.model_cfg.interval,
                        'frequency': self.model_cfg.optimizer_frequency,
                        'monitor': self.model_cfg.monitor,
                    },
                },
                {'optimizer': loss_optimizer}
            )
        return optimizer, loss_optimizer

    def _scheduler_step(self):
        if (self.trainer.current_epoch + 1) % self.model_cfg.optimizer_frequency == 0:
            if isinstance(self.lr_schedulers(), ReduceLROnPlateau):
                self.lr_schedulers().step(self.trainer.callback_metrics[self.model_cfg.monitor])
            elif not isinstance(self.lr_schedulers(), ReduceLROnPlateau):
                self.lr_schedulers().step()
