import datetime
import time
import random
import logging
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric, 
    DatasetDriftMetric, 
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
    ColumnMissingValuesMetric,
    ColumnSummaryMetric
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists evidently_metrics;
create table evidently_metrics (
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns int,
    share_missing_values float,
    fare_amount_quantile_50 float,
    fare_amount_min float,
    fare_amount_max float,
    fare_amount_mean float,
    fare_amount_median float,
    fare_amount_std float,
    fare_amount_missing_values int
)
"""

reference_data = pd.read_parquet('data/green_tripdata_2022-01_reference_data.parquet')

with open('models/hgb_reg.bin', 'rb') as f_in:
    model = joblib.load(f_in)

raw_data = pd.read_parquet('data/green_tripdata_2024-03.parquet')

begin = datetime.datetime(2024, 3, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name='prediction'),
        ColumnQuantileMetric(column_name='fare_amount', quantile=0.5),
        ColumnSummaryMetric(column_name='fare_amount'),
        ColumnMissingValuesMetric(column_name='fare_amount'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ]
)

def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=monitoring", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=monitoring") as conn:
            conn.execute(create_table_statement)

def calculate_metrics_postgresql(curr, i):
    current_data = raw_data[(raw_data['lpep_pickup_datetime'] >= (begin + datetime.timedelta(i))) 
                            & (raw_data['lpep_pickup_datetime'] < (begin + datetime.timedelta(i + 1)))]
    
    current_data['prediction'] = model.predict(current_data[num_features + cat_features])
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][4]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][5]['result']['current']['share_of_missing_values']
    fare_amount_quantile_50 = result['metrics'][1]['result']['current']['value']
    fare_amount_min = result['metrics'][2]['result']['current_characteristics']['min']
    fare_amount_max = result['metrics'][2]['result']['current_characteristics']['max']
    fare_amount_mean = result['metrics'][2]['result']['current_characteristics']['mean']
    fare_amount_median = result['metrics'][2]['result']['current_characteristics']['p50']
    fare_amount_std = result['metrics'][2]['result']['current_characteristics']['std']
    fare_amount_missing_values = result['metrics'][3]['result']['current']['number_of_missing_values']

    curr.execute("""
                 INSERT INTO evidently_metrics(timestamp,
                 prediction_drift, num_drifted_columns, share_missing_values,
                 fare_amount_quantile_50, fare_amount_min, fare_amount_max, fare_amount_mean, fare_amount_median,
                 fare_amount_std, fare_amount_missing_values) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                 """,
                 (begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values,
                  fare_amount_quantile_50, fare_amount_min, fare_amount_max, fare_amount_mean, fare_amount_median,
                  fare_amount_std, fare_amount_missing_values))

def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=monitoring", autocommit=True) as conn:
        for i in range(0, 30):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)

                new_send = datetime.datetime.now()
                seconds_elapsed = (new_send - last_send).total_seconds()
                if seconds_elapsed < SEND_TIMEOUT:
                    time.sleep(SEND_TIMEOUT - seconds_elapsed)
                while last_send < new_send:
                    last_send = last_send + datetime.timedelta(seconds=SEND_TIMEOUT)
                    logging.info(f"Sent {i + 1} messages")

if __name__ == "__main__":
    batch_monitoring_backfill()