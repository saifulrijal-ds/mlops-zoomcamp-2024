import pickle
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO)

def load_model(model_path: Path) -> Tuple[Any, Any]:
    """
    Load tge model and data vectorizer from the given path.

    Args:
        model_path (Path): Path to the saved model file.

    Returns:
        Tuple[Any, Any]: The data vectorizer and the model.
    """
    try:
        with model_path.open('rb') as f_in:
            dv, model = pickle.load(f_in)
        return dv, model
    except Exception as e:
        logging.error(f"Error in load_model: {e}")
        raise

# Load the data to predict
def read_dataframe(file_path: str, categorical_features: List[str]) -> pd.DataFrame:
    """
    Read the data from the given file path and preprocess it with Polars DataFrame.
    
    Args:
        file_path (Path): Path to the input data file.
        categorical_features (List[str]): Lits of catregorical feature names.
    
    Returns:
        pd.DataFrame: The proprocessed data as a Pandas DataFrame.
    """
    try:
        df = pl.scan_parquet(str(file_path))

        df = preprocess_data(df, categorical_features)

        return df.collect().to_pandas()
    except Exception as e:
        logging.error(f"Error in read_dataframe: {e}")
        raise

def preprocess_data(df: pl.LazyFrame, categorical_features: List[str]) -> pl.LazyFrame:
    """
    Preprocess the input data.

    Args:
        df (pl.LazyFrame): The input data as a Polars LazyFrame (scan parquet).
        categorical_features (List[str]): List of categorical feature names.

    Returns:
        pl.LazyFrame: The preprocessed data as a Polars LazyFrame.
    """
    try:
        df = df.with_columns(
            duration=(pl.col('tpep_dropoff_datetime') - pl.col('tpep_pickup_datetime')).dt.total_seconds() / 60
        ).filter(
            (pl.col('duration') >= 1) & (pl.col('duration') <= 60)
        ).with_columns(
            pl.col('PULocationID').fill_null(value=-1).cast(pl.Int16).cast(pl.Utf8).alias('PULocationID'),
            pl.col('DOLocationID').fill_null(value=-1).cast(pl.Int16).cast(pl.Utf8).alias('DOLocationID')
        )

        df = df.select(categorical_features + ['duration'])

        return df
    except Exception as e:
        logging.error(f"Error in preprocess_data: {e}")
        raise

def prepare_dictionaries(df: pd.DataFrame, categorical_features: List[str]) -> List[Dict]:
    """
    Prepare the input data as a list of dictionaries for the data vectorizer.

    Args:
        df (pd.DataFrame): The input data as a Pandas DataFrame.
        categorical_features (List[str]): List of categorical feature names.

    Returns:
        List[Dict]: The input data as a list of dictionaries.
    """
    try:
        return df[categorical_features].to_dict(orient='records')
    except Exception as e:
        logging.error(f"Error in prepare_dictionaries: {e}")
        raise

def apply_model(model_path: Path, input_file: str, output_file: Path) -> None:
    """
    Apply the model to the input data and save the predictions to the output file.

    Args:
        model_path (Path): Path to the saved model file.
        input_file (Path): Path to the input data file.
        output_file (Path): Path to the output file for saving predictions.
    """
    try:
        categorical_features = ['PULocationID', 'DOLocationID']
        df = read_dataframe(input_file, categorical_features)
        dicts = prepare_dictionaries(df, categorical_features)

        dv, model = load_model(model_path)

        X = dv.transform(dicts)
        y_pred = model.predict(X)

        df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
        
        df_result = pd.DataFrame()
        df_result['ride_id'] = df['ride_id']
        df_result['predicted_duration'] = y_pred

        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )

        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)

        logging.info(f"Mean of the prediction is: {mean_pred}")
        logging.info(f"Standard deviation of the prediction is: {std_pred}")
        logging.info(f"Predictions saved in {output_file}")

        return mean_pred, std_pred
    except Exception as e:
        logging.error(f"Error in apply_model: {e}")
        raise

def run(taxi_type: str, year: int, month: int) -> None:
    """
    Run the prediction pipeline for the given taxi type, year, and month.

    Args:
        taxi_type (str): The type of taxi data (e.g., 'yellow').
        year (int): The year of the data.
        month (int): The month of the data.
    """
    try:
        input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
        output_file = Path(f"outputs/{taxi_type}_tripdata_{year:04d}-{month:02d}_prediction.parquet")
        model_path = Path("model.bin")    

        return apply_model(model_path, input_file, output_file)
    except Exception as e:
        logging.error(f"Error in run: {e}")
        raise

if __name__ == "__main__":
    import sys

    try:
        taxi_type = sys.argv[1]
        year = int(sys.argv[2])
        month = int(sys.argv[3])
        
        mean, std = run(taxi_type, year, month)
        print(f"Mean prediction: {mean}")
        print(f"Std deviation of prediction: {std}")
    except Exception as e:
        logging.error(f"Error in main: {e}")
