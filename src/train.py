from dataclasses import asdict
from typing import TYPE_CHECKING

import hydra
import lightning
from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.callbacks.debug import VisualizeTriplets
from src.callbacks.embedding_visualization_callback import EmbeddingLogging
from src.config import ExperimentConfig
from src.constants import CONFIG_PATH
from src.datamodule import DataModule
from src.lightning_module import MetricLearningModule

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from pytorch_metric_learning.losses import BaseMetricLossFunction
    from pytorch_metric_learning.miners import BaseMiner
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


# noinspection PyDataclass
@hydra.main(config_path=str(CONFIG_PATH), config_name='conf', version_base='1.2')
def train(cfg: 'DictConfig'):  # noqa: WPS210

    experiment_config: ExperimentConfig = hydra.utils.instantiate(cfg.experiment_config)
    opt: 'Optimizer' = hydra.utils.instantiate(cfg.lightning_module.optimizer)
    scheduler: 'LRScheduler' = hydra.utils.instantiate(cfg.lightning_module.scheduler)
    loss: 'BaseMetricLossFunction' = hydra.utils.instantiate(cfg.lightning_module.loss)
    mining_func: 'BaseMiner' = hydra.utils.instantiate(cfg.lightning_module.mining_function)

    lightning.seed_everything(0)
    datamodule = DataModule(cfg=experiment_config.data_config)

    if experiment_config.project_config.track_in_clearml:
        Task.force_requirements_env_freeze()
        task = Task.init(
            project_name=experiment_config.project_config.project_name,
            task_name=experiment_config.project_config.experiment_name,
            # If `output_uri=True` uses default ClearML output URI,
            # can use string value to specify custom storage URI like S3.
            output_uri=True,
        )
        # Stores yaml config as a dictionary in clearml
        task.connect(asdict(experiment_config))
        task.connect_configuration(datamodule.transforms.get_train_transforms(), name='transformations')

    model = MetricLearningModule(
        experiment_config.model_config,
        optimizer=opt,
        scheduler=scheduler,
        loss=loss,
        mining_func=mining_func,
    )

    lr_logger = LearningRateMonitor(logging_interval='epoch')
    visualize = VisualizeTriplets(every_n_epochs=3)
    embedding_logger = EmbeddingLogging(datamodule.class_to_idx)
    early_stopping = EarlyStopping(monitor='precision_at_1_val', mode='max', patience=5)
    check_points = ModelCheckpoint(monitor='precision_at_1_val', mode='max', verbose=True)

    trainer = Trainer(
        **asdict(experiment_config.trainer_config),
        callbacks=[
            lr_logger,
            visualize,
            embedding_logger,
            early_stopping,
            check_points,
        ],
        overfit_batches=10,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path='best',
    )


if __name__ == '__main__':
    train()
    from pytorch_metric_learning.losses import ArcFaceLoss, TripletMarginLoss