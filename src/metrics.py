from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
)


def get_metrics(**kwargs) -> AccuracyCalculator:
    return AccuracyCalculator(**kwargs)
