import polars as pl

from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_selector import select_features

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(
    df: pl.DataFrame
) -> pl.DataFrame:
    df = clean(df)
    df = select_features(df)

    return df