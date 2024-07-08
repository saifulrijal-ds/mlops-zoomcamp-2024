import batch
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    expected_output = [
    {'PULocationID': '-1', 'DOLocationID': '-1', 'tpep_pickup_datetime': '2023-01-01 01:01:00', 'tpep_dropoff_datetime': '2023-01-01 01:10:00', 'duration': 9.0},
    {'PULocationID': '1', 'DOLocationID': '1', 'tpep_pickup_datetime': '2023-01-01 01:02:00', 'tpep_dropoff_datetime': '2023-01-01 01:10:00', 'duration': 8.0},
    ]

    expected_output_df = pd.DataFrame(expected_output, )
    expected_output_df['tpep_dropoff_datetime'] = pd.to_datetime(expected_output_df['tpep_dropoff_datetime'])
    expected_output_df['tpep_pickup_datetime'] = pd.to_datetime(expected_output_df['tpep_pickup_datetime']) 

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    input_df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    output_df = batch.prepare_data(input_df, categorical)

    # pd.testing.assert_frame_equal(output_df, expected_output_df)
    assert output_df.to_dict() == expected_output_df.to_dict()
