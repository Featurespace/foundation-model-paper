import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from clickhouse_driver import Client
from loguru import logger
from sklearn.preprocessing import OneHotEncoder

from .configs.dataset import DatasetSettings
from .features import load_features

EPSILON = 1e-6


def load_datasets(
    config: DatasetSettings,
    cache: Optional[Path] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Loads train & test datasets.
    Featurised transactions for train and test datasets should be stored
    in separate tables: `train_txn_table` and `test_txn_table`.
    Labels (targets) for both datasets should be stored in a single table
    `targets_table` containing `customerId` and `label` columns.
    """
    if cache is not None and cache.exists():
        logger.info(f"Loading data from cache at {cache}")
        with open(cache, "rb") as file:
            (train_y, train_X), (test_y, test_X) = pickle.load(file)
    else:
        logger.info("Creating dataset from clickhouse tables")
        train_features, test_features = load_features(config)

        targets = load_targets(config.targets_table)
        train_dataset = train_features.merge(targets, on="customerId")
        test_dataset = test_features.merge(targets, on="customerId")
        train_dataset, test_dataset = train_dataset.drop("customerId", axis=1), test_dataset.drop("customerId", axis=1)

        train_y, test_y = train_dataset.pop("label").to_numpy(), test_dataset.pop("label").to_numpy()

        train_X, test_X = preprocess_data(train_dataset, test_dataset)
        logger.info(f"Loaded train dataset contains {len(train_X)} entities")
        logger.info(f"Loaded test dataset contains {len(test_X)} entities")

        if cache is not None:
            logger.info(f"Saving dataset to cache at {cache}")
            with open(cache, "wb") as file:
                pickle.dump(((train_y, train_X), (test_y, test_X)), file)

    return (train_y, train_X), (test_y, test_X)


def load_targets(table: str) -> pd.DataFrame:
    """Loads the table with sequence labels.
    The table should contain the following columns:
        - customerId
        - label
    It can additionally contain other sequence-level features
    that can be used for modelling.
    """
    logger.info(f"Loading targets from {table}")
    client = Client("localhost")
    return client.query_dataframe(f"SELECT * FROM {table}")


def preprocess_data(train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """This function applies the following preprocessing steps:
    * log-transform and normalize numerical features;
    * one-hot-encode categorical features;
    * normalize embeddings.
    """
    logger.info("Preprocessing data")
    train_ds = train_ds.fillna(0.0)
    test_ds = test_ds.fillna(0.0)

    numerical_columns = set(
        col for col, dtype in train_ds.dtypes.items() if np.issubdtype(dtype, np.number) and "embedding" not in col
    )
    categorical_columns = set(col for col, dtype in train_ds.dtypes.items() if not np.issubdtype(dtype, np.number))
    embedding_columns = set(col for col, dtype in train_ds.dtypes.items() if "embedding" in col)

    train_numerical = train_ds[list(numerical_columns)].to_numpy()
    test_numerical = test_ds[list(numerical_columns)].to_numpy()
    train_categorical = train_ds[list(categorical_columns)].to_numpy()
    test_categorical = test_ds[list(categorical_columns)].to_numpy()
    train_embeddings = train_ds[list(embedding_columns)].to_numpy()
    test_embeddings = test_ds[list(embedding_columns)].to_numpy()

    if len(numerical_columns) > 0:
        train_numerical, test_numerical = log_transform(train_numerical), log_transform(test_numerical)
        train_numerical, test_numerical = normalize_features(train_ds=train_numerical, test_ds=test_numerical)

    if len(categorical_columns) > 0:
        train_categorical, test_categorical = encode_categorical_features(
            train_ds=train_categorical, test_ds=test_categorical
        )

    if len(embedding_columns) > 0:
        train_embeddings, test_embeddings = normalize_features(train_ds=train_embeddings, test_ds=test_embeddings)

    return (
        np.concatenate(
            [array for array in [train_numerical, train_categorical, train_embeddings] if array.shape[1] > 0], axis=1
        ),
        np.concatenate(
            [array for array in [test_numerical, test_categorical, test_embeddings] if array.shape[1] > 0], axis=1
        ),
    )


def normalize_features(train_ds: np.ndarray, test_ds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_ds, axis=0)
    std = np.std(train_ds, axis=0)

    normalized_train_ds = (train_ds - mean) / (std + EPSILON)
    normalized_test_ds = (test_ds - mean) / (std + EPSILON)

    return normalized_train_ds, normalized_test_ds


def log_transform(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log(np.abs(x) + 1.0)


def encode_categorical_features(train_ds: np.ndarray, test_ds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(train_ds)

    encoded_train = encoder.transform(train_ds).toarray()
    encoded_test = encoder.transform(test_ds).toarray()

    return encoded_train, encoded_test
