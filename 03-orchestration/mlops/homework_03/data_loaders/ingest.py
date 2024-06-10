import requests
from io import BytesIO
from typing import List

import polars as pl

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pl.DataFrame:
    dfs: List[pl.DataFrame] = []

    # Ingest March 2023 Yellow taxi trips data
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    response = requests.get(
        url
    )

    if response.status_code != 200:
        raise Exception(response.text)

    df = pl.read_parquet(BytesIO(response.content))
    dfs.append(df)

    return pl.concat(dfs)