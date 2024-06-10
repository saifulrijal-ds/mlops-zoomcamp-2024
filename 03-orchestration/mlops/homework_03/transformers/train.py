from sklearnex import patch_sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

import polars as pl
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def train(
    df: pl.DataFrame
):
    df = df.to_pandas()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    dv = DictVectorizer()
    X_dict = X.to_dict('records')
    X_dict = dv.fit_transform(X_dict)

    lr = LinearRegression()
    lr.fit(X_dict, y)

    print(lr.intercept_)

    return dv, lr
