from typing import List, Optional

import polars as pl

CATEGORICAL_FEATURES = ['PULocationID', 'DOLocationID']
NUMERICAL_FEATURES = []
TARGET = ['duration']


def select_features(df: pl.DataFrame, features: Optional[List[str]] = None) -> pl.DataFrame:
    columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + TARGET
    if features:
        columns += features

    return df.select(columns)