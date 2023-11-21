""" This module generates train/test tables from labelled data which were used
for training and evaluation. It also produces a table for training an embedding
model on the concatenation of labelled training data and unlabelled data.
"""

import sys

from clickhouse_driver import Client
from loguru import logger

logging_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <level>{message}</level>"
logger.remove()
logger.add(sys.stderr, format=logging_format, level="INFO")

client = Client("localhost")

MAX_MEM_USAGE = 300000000000


unsupervised_split_query = """
    CREATE TABLE {dst_table}
    ENGINE = MergeTree()
    ORDER BY (customerId, transactionTime)
    AS WITH
        cityHash64(customerId, 'split-salt') / pow(2,64) AS p,
        multiIf(p < 0.80, 'train', p < 1.0, 'test', 'other') AS split
    SELECT *
    FROM {src_labelled_table}
    WHERE split = 'train'
    UNION ALL
    SELECT *
    FROM {src_unlabelled_table}
    ORDER BY (customerId, transactionTime)
"""


train_test_split_query = """
    CREATE TABLE {dst_table}
    ENGINE = MergeTree()
    ORDER BY (customerId, transactionTime)
    AS WITH
        cityHash64(customerId, 'split-salt') / pow(2,64) AS p,
        multiIf(p < 0.80, 'train', p < 1.0, 'test', 'other') AS split
    SELECT *
    FROM {src_labelled_table}
    WHERE split = '{split}'
    ORDER BY (customerId, transactionTime)
"""


for dataset_name in ["sber", "rosbank", "retail_expenditure", "alpha"]:
    logger.info(f"--- Creating splits for {dataset_name} dataset")
    client.execute(f"DROP TABLE IF EXISTS NPPR_PAPER.{dataset_name}_unsupervised_split")
    client.execute(f"DROP TABLE IF EXISTS NPPR_PAPER.{dataset_name}_train_split")
    client.execute(f"DROP TABLE IF EXISTS NPPR_PAPER.{dataset_name}_test_split")

    if dataset_name != "retail_expenditure":
        "There is no unlabelled data on the retail dataset"
        logger.info("Creating unsupervised split")
        client.execute(
            unsupervised_split_query.format(
                src_labelled_table=f"NPPR_PAPER.{dataset_name}_labelled_txn",
                src_unlabelled_table=f"NPPR_PAPER.{dataset_name}_unlabelled_txn",
                dst_table=f"NPPR_PAPER.{dataset_name}_unsupervised_split",
            ),
            settings={"max_memory_usage": MAX_MEM_USAGE},
        )

    logger.info("Creating train split")
    client.execute(
        train_test_split_query.format(
            src_labelled_table=f"NPPR_PAPER.{dataset_name}_labelled_txn",
            dst_table=f"NPPR_PAPER.{dataset_name}_train_split",
            split="train",
        ),
        settings={"max_memory_usage": MAX_MEM_USAGE},
    )

    logger.info("Creating test split")
    client.execute(
        train_test_split_query.format(
            src_labelled_table=f"NPPR_PAPER.{dataset_name}_labelled_txn",
            dst_table=f"NPPR_PAPER.{dataset_name}_test_split",
            split="test",
        ),
        settings={"max_memory_usage": MAX_MEM_USAGE},
    )
