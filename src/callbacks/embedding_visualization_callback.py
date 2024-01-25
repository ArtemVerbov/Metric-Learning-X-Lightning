from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning import Callback, LightningModule, Trainer
from sklearn.manifold import TSNE
from torch import Tensor


class EmbeddingLogging(Callback):
    def __init__(self, class_to_idx: Dict[str, int]):
        super().__init__()
        self.labels = {index: name.split('_')[0] for name, index in class_to_idx.items()}
        self.predicts: List[Tensor] = []
        self.targets: List[Tensor] = []

    def on_test_batch_end(  # noqa: WPS211
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: List[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.predicts.append(outputs)
        self.targets.append(batch[1])

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        targets = torch.cat(self.targets, dim=0)
        predicts = torch.cat(self.predicts, dim=0)
        self._create_scatter(predicts, targets)

        trainer.logger.experiment.add_figure(
            'Projection of high denominational embedding space to two detentions',
            figure=plt.gcf(),
            global_step=trainer.current_epoch,
        )

    def _create_scatter(self, predicts: Tensor, targets: Tensor):
        labels = 'labels'
        tsne = TSNE(
            init='pca',
            perplexity=len(self.labels),
            metric='cosine',
        ).fit_transform(np.array(predicts))

        df = pd.DataFrame(tsne, columns=['x', 'y'])
        df[labels] = targets
        df[labels] = df[labels].map(self.labels)
        scatter = sns.scatterplot(data=df, x='x', y='y', hue=labels, palette='deep')
        scatter.legend(bbox_to_anchor=(1.04, 1.0), fancybox=True)
        plt.tight_layout()
