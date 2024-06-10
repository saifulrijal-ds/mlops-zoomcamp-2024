from mage_ai.io.file import FileIO
import joblib

import mlflow
import mlflow.sklearn

from typing import Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def save_and_log_artifacts(
    data: Tuple[DictVectorizer, LinearRegression]
    ) -> None:

    dv, lr = data

    dv_filepath = "./dv.joblib"

    joblib.dump(dv, dv_filepath)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi-mage")

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(lr, "LinRegModel")
        mlflow.log_artifact(dv_filepath)