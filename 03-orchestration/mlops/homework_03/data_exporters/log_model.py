import mlflow
import mlflow.sklearn

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def log_model(dv, lr):
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment("nyc-taxi-experiment")

    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "LRmodel")
        mlflow


