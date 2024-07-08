## Homework

In this homework, we'll take the ride duration prediction model that we deployed in batch mode in homework 4 and improve the reliability of our code with unit and integration tests.

You'll find the starter code in the [homework](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/cohorts/2024/06-best-practices/homework) directory.

## Q1. Refactoring

Before we can start covering our code with tests, we need to refactor it. We'll start by getting rid of all the global variables.

- Let's create a function `main` with two parameters: `year` and `month`.
- Move all the code (except `read_data`) inside `main`
- Make `categorical` a parameter for `read_data` and pass it inside `main`

Now we need to create the "main" block from which we'll invoke
the main function. How does the `if` statement that we use for
this looks like?

```python
#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    # output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

**if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)**
```

Hint: after refactoring, check that the code still works. Just run it e.g. for March 2023 and see if it finishes successfully.

To make it easier to run it, you can write results to your local
filesystem. E.g. here:

```python
output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'

```

## Q2. Installing pytest

- [x]  Now we need to install `pytest`:

```bash
pipenv install --dev pytest

```

Next, create a folder `tests` and create two files. One will be
the file with tests. We can name it `test_batch.py`.

What should be the other file?

**`__init__.py`**

Hint: to be able to test `batch.py`, we need to be able to
import it. Without this other file, we won't be able to do it.

## Q3. Writing first unit tes

Now let's cover our code with unit tests.

We'll start with the pre-processing logic inside `read_data`.

It's difficult to test right now because first reads
the file and then performs some transformations. We need to split this
code into two parts: reading (I/O) and transformation.

So let's create a function `prepare_data` that takes in a dataframe
(and some other parameters too) and applies some transformation to it.

(That's basically the entire `read_data` function after reading
the parquet file)

Now create a test and use this as input:

```python
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

```

Where `dt` is a helper function:

```python
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

```

Define the expected output and use the assert to make sure
that the actual dataframe matches the expected one.

Tip: When you compare two Pandas DataFrames, the result is also a DataFrame.
The same is true for Pandas Series. Also, a DataFrame could be turned into a list of dictionaries.

How many rows should be there in the expected dataframe?

- 1
- **2**
- 3
- 4

```python
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

```

## Q4. Mocking S3 with Localstack

Now let's prepare for an integration test. In our script, we
write data to S3. So we'll use Localstack to mimic S3.

First, let's run Localstack with Docker compose. Let's create a
`docker-compose.yaml` file with just one service: localstack. Inside
localstack, we're only interested in running S3.

Start the service and test it by creating a bucket where we'll
keep the output. Let's call it "nyc-duration".

With AWS CLI, this is how we create a bucket:

```bash
aws s3 mb s3://nyc-duration

```

Then we need to check that the bucket was successfully created. With AWS, this is how we typically do it:

```bash
aws s3 ls

```

In both cases we should adjust commands for localstack. What option do we need to use for such purposes?

- `-backend-store-uri`
- `-profile`
- **`-endpoint-url`**
- `-version`
- [x]  Set up or configure AWS credentials (create new profile called `localstack` )
    
    ```bash
    aws configure --profile localstack
    
    # AWS Access Key ID [None]: test
    # AWS Secret Access Key [None]: test
    # Default region name [None]: us-east-1
    # Default output format [None]: 
    ```
    
- [x]  let’s modify a little bit `/home/ubuntu/.aws/config`
    
    ```
    [profile localstack]
    region = us-east-1
    endpoint_url = http://localhost:4566
    ```
    
- [x]  create bucket
    
    ```bash
    aws s3 mb s3://nyc-duration --profile localstack
    ```
    
    or (if endpoint is not define in `config` file)
    
    ```bash
    aws s3 mb s3://nyc-duration --profile localstack --endpoint-url http://localhost:4566 
    ```
    
- [x]  list bucket
    
    ```bash
    aws s3 ls --profile localstack
    ```
    

## Make input and output paths configurable

Right now the input and output paths are hardcoded, but we want
to change it for the tests.

One of the possible ways would be to specify `INPUT_FILE_PATTERN` and `OUTPUT_FILE_PATTERN` via the env
variables. Let's do that:

```bash
export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"

```

And this is how we can read them:

```python
def get_input_path(year, month):
    default_input_pattern = '<https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet>'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    # rest of the main function ...

```

## Reading from Localstack S3 with Pandas

So far we've been reading parquet files from S3 with using
pandas `read_parquet`. But this way we read it from the
actual S3 service. Now we need to replace it with our localstack
one.

For that, we need to specify the endpoint url:

```python
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

df = pd.read_parquet('s3://bucket/file.parquet', storage_options=options)

```

Let's modify our `read_data` function:

- check if `S3_ENDPOINT_URL` is set, and if it is, use it for reading
- otherwise use the usual way

## Q5. Creating test data

Now let's create `integration_test.py`

We'll use the dataframe we created in Q3 (the dataframe for the unit test)
and save it to S3. You don't need to do anything else: just create a dataframe
and save it.

We will pretend that this is data for **January 2023**.

Run the `integration_test.py` script. After that, use AWS CLI to verify that the
file was created.

Use this snipped for saving the file:

```python
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

```

What's the size of the file?

- **3620**
- 23620
- 43620
- 63620

Note: it's important to use the code from the snippet for saving
the file. Otherwise the size may be different depending on the OS,
engine and compression. Even if you use this exact snippet, the size
of your dataframe may still be a bit off. Just select the closest option.

```python
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "")
options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

year = 2023
month = 1
input_file = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

```

![Untitled](images/s3_localstack_input_file.png)

## Q6. Finish the integration test

We can read from our localstack s3, but we also need to write to it.

Create a function `save_data` which works similarly to `read_data`,
but we use it for saving a dataframe.

Let's run the `batch.py` script for January 2023 (the fake data
we created in Q5).

We can do that from our integration test in Python: we can use
`os.system` for doing that (there are other options too).

Now it saves the result to localstack.

The only thing we need to do now is to read this data and
verify the result is correct.

What's the sum of predicted durations for the test dataframe?

- 13.08
- **36.28**
- 69.28
- 81.08

![Untitled1](images/s3_localstack_batch.png)

## Running the test (ungraded)

The rest is ready, but we need to write a shell script for doing
that.

Let's do that!

## Submit the results

- Submit your results here: https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw6
- It's possible that your answers won't match exactly. If it's the case, select the closest one.