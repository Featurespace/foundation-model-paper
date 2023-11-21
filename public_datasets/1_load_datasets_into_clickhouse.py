""" This module loads public datasets into ClickHouse tables that
can be used with the `benchmarker` library for training and evaluating
models.
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from clickhouse_driver import Client
from loguru import logger

logging_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <level>{message}</level>"
logger.remove()
logger.add(sys.stderr, format=logging_format, level="INFO")


class PandasTypes(Enum):
    STR = "str"
    FLOAT = "float"
    INT = "int32"
    BIN = "int8"
    DATE = "datetime64[ns]"


class ClickhouseTypes(Enum):
    STR = "String"
    FLOAT = "Float32"
    INT = "UInt32"
    BIN = "UInt8"
    DATE = "DateTime"


pandas_type_to_clickhouse = {
    PandasTypes.STR: ClickhouseTypes.STR,
    PandasTypes.FLOAT: ClickhouseTypes.FLOAT,
    PandasTypes.INT: ClickhouseTypes.INT,
    PandasTypes.BIN: ClickhouseTypes.BIN,
    PandasTypes.DATE: ClickhouseTypes.DATE,
}


def to_clickhouse_table_schema(schema_dict: Dict[str, PandasTypes]) -> str:
    field_specs = [
        f"`{field_name}` {pandas_type_to_clickhouse[type_name].value}" for field_name, type_name in schema_dict.items()
    ]
    schema_string = ", ".join(field_specs)
    return f"({schema_string})"


@dataclass
class ClickhouseTableSettings:
    txn_table_schema: Dict[str, PandasTypes]
    targets_table_schema: Dict[str, PandasTypes]

    "Columns used for ordering data within ClickHouse tables"
    txn_tables_order_by_columns: List[str]
    targets_table_order_by_columns: List[str]

    database: str
    targets_table_name: str
    labelled_txn_table_name: str
    unlabelled_txn_table_name: Optional[str] = None


def load_data_into_clickhouse(
    settings: ClickhouseTableSettings,
    targets: pd.DataFrame,
    labelled_txn: pd.DataFrame,
    unlabelled_txn: Optional[pd.DataFrame] = None,
    replace_if_exists: bool = True,
):
    """Load transactions and labels into clickhouse tables

    It creates three tables with labelled transactions, unlabelled
    transactions, and sequence labels from provided pandas dataframes.
    The column names in those dataframes have to match exactly the
    schema dictionaries provided in `settings`.
    """

    logger.info("Starting data importation")
    client = Client("localhost", settings={"use_numpy": True})
    client.execute(f"CREATE DATABASE IF NOT EXISTS {settings.database}")

    txn_table_schema = to_clickhouse_table_schema(settings.txn_table_schema)
    targets_table_schema = to_clickhouse_table_schema(settings.targets_table_schema)

    labelled_txn = labelled_txn[list(settings.txn_table_schema.keys())]
    targets = targets[list(settings.targets_table_schema.keys())]

    labelled_txn = labelled_txn.astype({key: val.value for key, val in settings.txn_table_schema.items()})
    targets = targets.astype({key: val.value for key, val in settings.targets_table_schema.items()})

    if unlabelled_txn is not None:
        unlabelled_txn = unlabelled_txn[list(settings.txn_table_schema.keys())]
        unlabelled_txn = unlabelled_txn.astype({key: val.value for key, val in settings.txn_table_schema.items()})

    if replace_if_exists:
        client.execute(f"DROP TABLE IF EXISTS {settings.database}.{settings.labelled_txn_table_name}")
        client.execute(f"DROP TABLE IF EXISTS {settings.database}.{settings.targets_table_name}")
        if unlabelled_txn is not None:
            client.execute(f"DROP TABLE IF EXISTS {settings.database}.{settings.unlabelled_txn_table_name}")

    # --- Create targets table ---

    client.execute(
        f"""
        CREATE TABLE {settings.database}.{settings.targets_table_name}
        {targets_table_schema}
        ENGINE = MergeTree()
        ORDER BY ({", ".join(settings.targets_table_order_by_columns)})
    """
    )
    client.insert_dataframe(f"INSERT INTO {settings.database}.{settings.targets_table_name} VALUES", targets)
    logger.info("Finished importing targets")

    # --- Create labelled txn table ---

    client.execute(
        f"""
        CREATE TABLE {settings.database}.{settings.labelled_txn_table_name}
        {txn_table_schema}
        ENGINE = MergeTree()
        ORDER BY ({", ".join(settings.txn_tables_order_by_columns)})
    """
    )
    client.insert_dataframe(f"INSERT INTO {settings.database}.{settings.labelled_txn_table_name} VALUES", labelled_txn)
    logger.info("Finished importing labelled data")

    if unlabelled_txn is not None:
        # --- Create unlabelled txn table ---

        client.execute(
            f"""
            CREATE TABLE {settings.database}.{settings.unlabelled_txn_table_name}
            {txn_table_schema}
            ENGINE = MergeTree()
            ORDER BY ({", ".join(settings.txn_tables_order_by_columns)})
        """
        )
        client.insert_dataframe(
            f"INSERT INTO {settings.database}.{settings.unlabelled_txn_table_name} VALUES", unlabelled_txn
        )
        logger.info("Finished importing unlabelled data")


def load_retail_for_expenditure():
    purchases = pd.read_csv("retail/purchases.csv")
    logger.info("Loaded transactions")
    products = pd.read_csv("retail/products.csv")
    logger.info("Loaded products info")

    purchases = purchases.merge(products, on="product_id")

    column_names_map = {
        "client_id": "customerId",
        "trn_sum_from_iss": "amt",
    }
    purchases = purchases.rename(column_names_map, axis=1)

    purchases["transactionTime"] = pd.to_datetime(purchases["transaction_datetime"], format="%Y-%m-%d %H:%M:%S")
    purchases.sort_values(["customerId", "transactionTime"], inplace=True)
    purchases.reset_index(inplace=True)
    purchases["hour_diff"] = (
        purchases[["customerId", "transactionTime"]]
        .groupby(["customerId"])
        .diff()["transactionTime"]
        .dt.total_seconds()
        .fillna(0.0)
        / 3600.0
    )
    purchases["transaction_number"] = purchases.groupby(["customerId"]).cumcount() + 1
    # Make sure there is at least 1s different between two consecutive transactions
    # to ensure a correct order of transactions in the ClickHouse table.
    purchases["transactionTime"] += pd.to_timedelta(purchases["transaction_number"], unit="s")
    purchases["transactionId"] = (
        purchases["customerId"].astype(str).str.cat(purchases["transaction_number"].astype(str))
    )

    purchases_for_training = purchases[purchases["transactionTime"] < "2019-02-18 00:00:00"].reset_index()
    purchases_for_labelling = purchases[purchases["transactionTime"] >= "2019-02-18 00:00:00"].reset_index()

    # Compute expenditure during the `labelling` period
    targets = (
        purchases_for_labelling[["customerId", "amt"]].groupby("customerId", sort=False)["amt"].sum().reset_index()
    )
    targets = targets.rename({"amt": "label"}, axis=1)

    clickhouse_tables_settings = ClickhouseTableSettings(
        database="NPPR_PAPER",
        labelled_txn_table_name="retail_expenditure_labelled_txn",
        targets_table_name="retail_expenditure_targets",
        txn_table_schema={
            "customerId": PandasTypes.STR,
            "transactionId": PandasTypes.STR,
            "transaction_number": PandasTypes.INT,
            "transactionTime": PandasTypes.DATE,
            "amt": PandasTypes.FLOAT,
            "product_quantity": PandasTypes.FLOAT,
            "segment_id": PandasTypes.STR,
            "level_1": PandasTypes.STR,
            "level_2": PandasTypes.STR,
            "level_3": PandasTypes.STR,
            "is_own_trademark": PandasTypes.BIN,
            "is_alcohol": PandasTypes.BIN,
            "hour_diff": PandasTypes.FLOAT,
        },
        targets_table_schema={
            "customerId": PandasTypes.STR,
            "label": PandasTypes.FLOAT,
        },
        txn_tables_order_by_columns=["customerId", "transactionTime"],
        targets_table_order_by_columns=["customerId"],
    )
    load_data_into_clickhouse(settings=clickhouse_tables_settings, labelled_txn=purchases_for_training, targets=targets)


def load_rosbank_for_churn():
    labelled_txn = pd.read_csv("rosbank/train.csv")
    logger.info("Loaded labelled data")
    unlabelled_txn = pd.read_csv("rosbank/test.csv")
    logger.info("Loaded unlabelled data")

    column_names_map = {
        "cl_id": "customerId",
        "MCC": "mcc",
        "amount": "amt",
        "target_flag": "label",
    }
    labelled_txn = labelled_txn.rename(column_names_map, axis=1)
    unlabelled_txn = unlabelled_txn.rename(column_names_map, axis=1)

    # Each transaction in a history of transactions for a given `customerId`
    # has the same `label` attached, so in the line below we extract the
    # label from only the first transaction.
    targets = labelled_txn[["customerId", "label"]].groupby("customerId").first().reset_index()

    for dataframe in (labelled_txn, unlabelled_txn):
        dataframe["transactionTime"] = pd.to_datetime(dataframe["TRDATETIME"], format="%d%b%y:%H:%M:%S")
        dataframe.sort_values(["customerId", "transactionTime"], inplace=True)
        dataframe.reset_index(inplace=True)
        dataframe["hour_diff"] = (
            dataframe[["customerId", "transactionTime"]]
            .groupby(["customerId"])
            .diff()["transactionTime"]
            .dt.total_seconds()
            .fillna(0.0)
            / 3600.0
        )
        dataframe["transaction_number"] = dataframe.groupby(["customerId"]).cumcount() + 1
        # Make sure there is at least 1s different between two consecutive transactions
        # to ensure a correct order of transactions in the ClickHouse table.
        dataframe["transactionTime"] += pd.to_timedelta(dataframe["transaction_number"], unit="s")
        dataframe["transactionId"] = (
            dataframe["customerId"].astype(str).str.cat(dataframe["transaction_number"].astype(str))
        )

    clickhouse_tables_settings = ClickhouseTableSettings(
        database="NPPR_PAPER",
        labelled_txn_table_name="rosbank_labelled_txn",
        unlabelled_txn_table_name="rosbank_unlabelled_txn",
        targets_table_name="rosbank_targets",
        txn_table_schema={
            "customerId": PandasTypes.STR,
            "transactionId": PandasTypes.STR,
            "transaction_number": PandasTypes.INT,
            "transactionTime": PandasTypes.DATE,
            "amt": PandasTypes.FLOAT,
            "mcc": PandasTypes.STR,
            "channel_type": PandasTypes.STR,
            "currency": PandasTypes.STR,
            "trx_category": PandasTypes.STR,
            "hour_diff": PandasTypes.FLOAT,
        },
        targets_table_schema={
            "customerId": PandasTypes.STR,
            "label": PandasTypes.INT,
        },
        txn_tables_order_by_columns=["customerId", "transactionTime"],
        targets_table_order_by_columns=["customerId"],
    )
    load_data_into_clickhouse(
        settings=clickhouse_tables_settings, labelled_txn=labelled_txn, unlabelled_txn=unlabelled_txn, targets=targets
    )


def create_datetime_sber_data(df):
    """In data from SberBank, the date feature is simply a day number
    of the transaction. We parse it into a format that can be converted
    into a datetime column.
    """
    year, day_of_year = divmod(int(df.trans_date), 366)
    padded_day_of_year = f"00{day_of_year + 1}"[-3:]
    return f"{2017 + year}:{padded_day_of_year}"


def load_sber():
    labelled_txn = pd.read_csv("age_group_prediction/transactions_train.csv")
    logger.info("Loaded labelled data")
    targets = pd.read_csv("age_group_prediction/train_target.csv")
    logger.info("Loaded targets")
    unlabelled_txn = pd.read_csv("age_group_prediction/transactions_test.csv")
    logger.info("Loaded unlabelled data")

    column_names_map = {
        "client_id": "customerId",
        "small_group": "mcc",
        "amount_rur": "amt",
        "bins": "label",
    }

    labelled_txn = labelled_txn.rename(column_names_map, axis=1)
    unlabelled_txn = unlabelled_txn.rename(column_names_map, axis=1)
    targets = targets.rename(column_names_map, axis=1)

    for dataframe in (labelled_txn, unlabelled_txn):
        dataframe["transactionTime"] = pd.to_datetime(
            dataframe.apply(create_datetime_sber_data, axis=1), format="%Y:%j"
        )
        dataframe.sort_values(["customerId", "transactionTime"], inplace=True)
        dataframe.reset_index(inplace=True)
        dataframe["day_diff"] = (
            dataframe[["customerId", "transactionTime"]]
            .groupby(["customerId"])
            .diff()["transactionTime"]
            .dt.total_seconds()
            .fillna(0.0)
            / 86400.0
        )
        dataframe["transaction_number"] = dataframe.groupby(["customerId"]).cumcount() + 1
        # Make sure there is at least 1s different between two consecutive transactions
        # to ensure a correct order of transactions in the ClickHouse table.
        dataframe["transactionTime"] += pd.to_timedelta(dataframe["transaction_number"], unit="s")
        dataframe["transactionId"] = (
            dataframe["customerId"].astype(str).str.cat(dataframe["transaction_number"].astype(str))
        )

    clickhouse_tables_settings = ClickhouseTableSettings(
        database="NPPR_PAPER",
        labelled_txn_table_name="sber_labelled_txn",
        unlabelled_txn_table_name="sber_unlabelled_txn",
        targets_table_name="sber_targets",
        txn_table_schema={
            "customerId": PandasTypes.STR,
            "transactionId": PandasTypes.STR,
            "transaction_number": PandasTypes.INT,
            "transactionTime": PandasTypes.DATE,
            "day_diff": PandasTypes.FLOAT,
            "amt": PandasTypes.FLOAT,
            "mcc": PandasTypes.STR,
        },
        targets_table_schema={
            "customerId": PandasTypes.STR,
            "label": PandasTypes.INT,
        },
        txn_tables_order_by_columns=["customerId", "transactionTime"],
        targets_table_order_by_columns=["customerId"],
    )
    load_data_into_clickhouse(
        settings=clickhouse_tables_settings, labelled_txn=labelled_txn, unlabelled_txn=unlabelled_txn, targets=targets
    )


def load_data_from_parquet_partitions(path: Path):
    partitions = []
    for partition_path in path.iterdir():
        partitions.append(pd.read_parquet(partition_path))
    return pd.concat(partitions)


def load_alpha_bank():
    labelled_txn_path = Path("default_prediction_on_credit_cards/train_transactions_contest")
    targets_path = Path("default_prediction_on_credit_cards/train_target.csv")
    unlabelled_txn_path = Path("default_prediction_on_credit_cards/test_transactions_contest")

    labelled_txn = load_data_from_parquet_partitions(labelled_txn_path)
    logger.info("Loaded labelled data")
    targets = pd.read_csv(targets_path)
    logger.info("Loaded targets")
    unlabelled_txn = load_data_from_parquet_partitions(unlabelled_txn_path)
    logger.info("Loaded unlabelled data")

    column_names_map = {
        "app_id": "customerId",
        "amnt": "amt",
        "flag": "label",
        "hour": "hour_of_day",
        "weekofyear": "week_of_year",
    }
    labelled_txn = labelled_txn.drop_duplicates().rename(column_names_map, axis=1)
    unlabelled_txn = unlabelled_txn.drop_duplicates().rename(column_names_map, axis=1)
    targets = targets.rename(column_names_map, axis=1)

    for dataframe in (labelled_txn, unlabelled_txn):
        dataframe.sort_values(["customerId", "transaction_number"], inplace=True)
        dataframe.reset_index(inplace=True)
        dataframe["cum_hour_diff"] = (
            dataframe[["customerId", "hour_diff"]].replace(-1, 0).groupby(["customerId"]).cumsum()
        )
        # Create a datetime column starting from a dummy date for each first transaction in a sequence
        dataframe["transactionTime"] = pd.Timestamp("2022-01-01 00:00:00")
        dataframe["transactionTime"] += pd.to_timedelta(dataframe["cum_hour_diff"], unit="hour")
        # Make sure there is at least 1s different between two consecutive transactions
        # to ensure a correct order of transactions in the ClickHouse table.
        dataframe["transactionTime"] += pd.to_timedelta(dataframe["transaction_number"], unit="second")
        dataframe["transactionId"] = (
            dataframe["customerId"].astype(str).str.cat(dataframe["transaction_number"].astype(str))
        )

    clickhouse_tables_settings = ClickhouseTableSettings(
        database="NPPR_PAPER",
        labelled_txn_table_name="alpha_labelled_txn",
        unlabelled_txn_table_name="alpha_unlabelled_txn",
        targets_table_name="alpha_targets",
        txn_table_schema={
            "customerId": PandasTypes.STR,
            "transactionId": PandasTypes.STR,
            "transaction_number": PandasTypes.INT,
            "transactionTime": PandasTypes.DATE,
            "amt": PandasTypes.FLOAT,
            "currency": PandasTypes.STR,
            "operation_kind": PandasTypes.STR,
            "card_type": PandasTypes.STR,
            "operation_type": PandasTypes.STR,
            "operation_type_group": PandasTypes.STR,
            "ecommerce_flag": PandasTypes.BIN,
            "payment_system": PandasTypes.STR,
            "income_flag": PandasTypes.BIN,
            "mcc": PandasTypes.STR,
            "country": PandasTypes.STR,
            "city": PandasTypes.STR,
            "mcc_category": PandasTypes.STR,
            "week_of_year": PandasTypes.STR,
            "day_of_week": PandasTypes.STR,
            "hour_of_day": PandasTypes.STR,
            "days_before": PandasTypes.INT,
            "hour_diff": PandasTypes.FLOAT,
        },
        targets_table_schema={
            "customerId": PandasTypes.STR,
            "label": PandasTypes.INT,
            "product": PandasTypes.STR,
        },
        txn_tables_order_by_columns=["customerId", "transactionTime"],
        targets_table_order_by_columns=["customerId"],
    )
    load_data_into_clickhouse(
        settings=clickhouse_tables_settings, labelled_txn=labelled_txn, unlabelled_txn=unlabelled_txn, targets=targets
    )


if __name__ == "__main__":
    logger.info("--- Importing Rosbank data")
    load_rosbank_for_churn()

    logger.info("--- Importing Sber Bank data")
    load_sber()

    logger.info("--- Importing Alpha Bank data")
    load_alpha_bank()

    logger.info("--- Importing retail purchases expenditure dataset")
    load_retail_for_expenditure()
