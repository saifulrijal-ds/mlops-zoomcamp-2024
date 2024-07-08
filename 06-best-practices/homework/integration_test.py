import os
import logging
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def create_sample_data() -> pd.DataFrame:
    data = [
        (None, None, datetime(2021, 1, 1, 1, 1), datetime(2021, 1, 1, 1, 10)),
        (1, 1, datetime(2021, 1, 1, 1, 2), datetime(2021, 1, 1, 1, 10)),
        (1, None, datetime(2021, 1, 1, 1, 2, 0), datetime(2021, 1, 1, 1, 2, 59)),
        (3, 4, datetime(2021, 1, 1, 1, 2, 0), datetime(2021, 1, 1, 2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    return pd.DataFrame(data, columns=columns)

def main() -> None:
    load_dotenv()
    s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")

    sample_data = create_sample_data()
    year, month = 2023, 1
    storage_options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }
    }

    input_file = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    logger.info(f"Uploading sample data to {input_file}")
    sample_data.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=storage_options
    )

    logger.info("Running prediction job")
    os.system(f"python batch.py {year} {month}")

    output_file = f"s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
    logger.info(f"Reading output from {output_file}")
    df_output = pd.read_parquet(
        output_file,
        storage_options=storage_options
    )

    total_duration = df_output['predicted_duration'].sum()
    logger.info(f"Total predicted duration: {total_duration}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
