# ML Project with DVC, MLflow, and MinIO

This project demonstrates a complete ML workflow:
1. Fetching data from MinIO
2. Preprocessing
3. Training a model
4. Logging with MLflow
5. Managing workflow with DVC

## Setup
```
pip install -r requirements.txt
dvc init
mlflow ui
```

## Running Pipeline
```
dvc repro
```
