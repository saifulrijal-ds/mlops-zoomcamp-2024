import polars as pl


def clean(
    df: pl.DataFrame,
    include_extreme_durations: bool = False,
) -> pl.DataFrame:
    # Calculate the trip duration in minutes
    df = df.with_columns(
        duration=(pl.col('tpep_dropoff_datetime') - pl.col('tpep_pickup_datetime')).dt.total_seconds() / 60
    )

    if not include_extreme_durations:
        # Filter out trips that are less than 1 minute or more than 60 minutes
        df = df.filter((pl.col('duration') >= 1) & (pl.col('duration') <= 60))

    # Convert location IDs to string to treat them as categorical features
    df = df.with_columns(
        pl.col('PULocationID').cast(pl.Utf8).alias('PULocationID'),
        pl.col('DOLocationID').cast(pl.Utf8).alias('DOLocationID')
    )

    return df