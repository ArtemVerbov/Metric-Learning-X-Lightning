# Metric-learning template 

```
All experiments are tracked with clearml: https://clear.ml/docs/latest/docs/

Environment created with poetry: https://python-poetry.org/docs/

To install dependeces: poetry install

There are two options to run training in this repository:

1. Defualt with TripletMarginLoss and TripletMarginMiner: poetry run python3.11 -m src.train 
2. With ArcFaceLoss: poetry run python3.11 -m src.train lightning_module=ArcFaceLoss
```

1. Setup ClearML: clearml-init

2. Migrate dataset to ClearML: make migrate_dataset

## Dataset information
Dataset used: Stanford Online Products

More info: https://paperswithcode.com/dataset/stanford-online-products

## Metrics traced with clearml:
1. Training and test with [TripletMarginLoss](https://app.clear.ml/projects/34447ffe1ce24bd4a9701d8ca7e12cfc/experiments/98b62b3cbb4a4fa39eebbc85ea39adf0/output/execution)
2. Training and test with [ArcFaceLoss](https://app.clear.ml/projects/34447ffe1ce24bd4a9701d8ca7e12cfc/experiments/c7b74a9bf4474e768cde966c81d4f165/output/execution)

## Test data projection of high dimension embedding space to two dimensions

### With TripletLoss:
![alt text](https://github.com/ArtemVerbov/Metric-Learning-X-Lightning/blob/main/media/Triplet_Loss_Embeddings.jpeg?raw=true)

### With ArcFaceLoss:
![alt text](https://github.com/ArtemVerbov/Metric-Learning-X-Lightning/blob/arc-face-loss/media/ArcFacle_Loss_Embeddings.jpeg?raw=true)
