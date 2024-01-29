# Metric-learning template 

```
All experiments are tracked with clearml: https://clear.ml/docs/latest/docs/

Environment created with poetry: https://python-poetry.org/docs/

To install dependeces: poetry install
To run training: poetry run python3.11 -m src.train   
```

1. Setup ClearML: clearml-init

2. Migrate dataset to ClearML: make migrate_dataset

## Dataset information
Dataset used: Stanford Online Products

More info: https://paperswithcode.com/dataset/stanford-online-products

## All training and test metrics traced with clearml:
https://app.clear.ml/projects/34447ffe1ce24bd4a9701d8ca7e12cfc/experiments/98b62b3cbb4a4fa39eebbc85ea39adf0/output/execution

## Test data projection of high dimension embedding space to two dimensions

![alt text](https://github.com/ArtemVerbov/Image-Segmentation-X-Lightning/blob/main/media/masksData.png?raw=true)
