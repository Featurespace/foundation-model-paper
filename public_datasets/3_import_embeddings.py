""" This module inserts embeddings from `embeddings.tar.gz` archive
(which needs to be manually unpacked) into ClickHouse database.
"""

import os

from tqdm import tqdm

os.system("clickhouse-client --query 'CREATE DATABASE IF NOT EXISTS NPPR_PAPER'")
for table in tqdm(
    [
        # ROSBANK
        "NPPR_PAPER.rosbank_train_avg_embeddings_np_ne",
        "NPPR_PAPER.rosbank_test_avg_embeddings_np_ne",
        "NPPR_PAPER.rosbank_train_embeddings_coles",
        "NPPR_PAPER.rosbank_test_embeddings_coles",
        "NPPR_PAPER.rosbank_train_embeddings_red",
        "NPPR_PAPER.rosbank_test_embeddings_red",
        "NPPR_PAPER.rosbank_train_embeddings_simcse",
        "NPPR_PAPER.rosbank_test_embeddings_simcse",
        "NPPR_PAPER.rosbank_train_avg_embeddings_np_only",
        "NPPR_PAPER.rosbank_test_avg_embeddings_np_only",
        "NPPR_PAPER.rosbank_train_avg_embeddings_ne_only",
        "NPPR_PAPER.rosbank_test_avg_embeddings_ne_only",
        "NPPR_PAPER.rosbank_train_embeddings_np_ne",
        "NPPR_PAPER.rosbank_test_embeddings_np_ne",
        "NPPR_PAPER.rosbank_train_avg_embeddings_coles",
        "NPPR_PAPER.rosbank_test_avg_embeddings_coles",
        # SBER
        "NPPR_PAPER.sber_train_avg_embeddings_np_ne_0_001",
        "NPPR_PAPER.sber_test_avg_embeddings_np_ne_0_001",
        "NPPR_PAPER.sber_train_embeddings_coles",
        "NPPR_PAPER.sber_test_embeddings_coles",
        "NPPR_PAPER.sber_train_embeddings_red",
        "NPPR_PAPER.sber_test_embeddings_red",
        "NPPR_PAPER.sber_train_embeddings_simcse",
        "NPPR_PAPER.sber_test_embeddings_simcse",
        "NPPR_PAPER.sber_train_avg_embeddings_np_only",
        "NPPR_PAPER.sber_test_avg_embeddings_np_only",
        "NPPR_PAPER.sber_train_avg_embeddings_ne_only",
        "NPPR_PAPER.sber_test_avg_embeddings_ne_only",
        "NPPR_PAPER.sber_train_embeddings_np_ne_0_001",
        "NPPR_PAPER.sber_test_embeddings_np_ne_0_001",
        "NPPR_PAPER.sber_train_avg_embeddings_coles",
        "NPPR_PAPER.sber_test_avg_embeddings_coles",
        # RETAIL EXPENDITURE
        "NPPR_PAPER.retail_expenditure_train_avg_embeddings_np_ne_0_005",
        "NPPR_PAPER.retail_expenditure_test_avg_embeddings_np_ne_0_005",
        "NPPR_PAPER.retail_expenditure_train_embeddings_coles",
        "NPPR_PAPER.retail_expenditure_test_embeddings_coles",
        "NPPR_PAPER.retail_expenditure_train_embeddings_red",
        "NPPR_PAPER.retail_expenditure_test_embeddings_red",
        "NPPR_PAPER.retail_expenditure_train_embeddings_simcse",
        "NPPR_PAPER.retail_expenditure_test_embeddings_simcse",
        "NPPR_PAPER.retail_expenditure_train_avg_embeddings_np_only",
        "NPPR_PAPER.retail_expenditure_test_avg_embeddings_np_only",
        "NPPR_PAPER.retail_expenditure_train_avg_embeddings_ne_only",
        "NPPR_PAPER.retail_expenditure_test_avg_embeddings_ne_only",
        "NPPR_PAPER.retail_expenditure_train_embeddings_np_ne_0_005",
        "NPPR_PAPER.retail_expenditure_test_embeddings_np_ne_0_005",
        "NPPR_PAPER.retail_expenditure_train_avg_embeddings_coles",
        "NPPR_PAPER.retail_expenditure_test_avg_embeddings_coles",
        # ALPHA
        "NPPR_PAPER.alpha_train_avg_embeddings_np_ne_0_001",
        "NPPR_PAPER.alpha_test_avg_embeddings_np_ne_0_001",
        "NPPR_PAPER.alpha_train_embeddings_coles",
        "NPPR_PAPER.alpha_test_embeddings_coles",
        "NPPR_PAPER.alpha_train_embeddings_red",
        "NPPR_PAPER.alpha_test_embeddings_red",
        "NPPR_PAPER.alpha_train_embeddings_simcse",
        "NPPR_PAPER.alpha_test_embeddings_simcse",
        "NPPR_PAPER.alpha_train_avg_embeddings_np_only",
        "NPPR_PAPER.alpha_test_avg_embeddings_np_only",
        "NPPR_PAPER.alpha_train_avg_embeddings_ne_only",
        "NPPR_PAPER.alpha_test_avg_embeddings_ne_only",
        "NPPR_PAPER.alpha_train_embeddings_np_ne_0_001",
        "NPPR_PAPER.alpha_test_embeddings_np_ne_0_001",
        "NPPR_PAPER.alpha_train_avg_embeddings_coles",
        "NPPR_PAPER.alpha_test_avg_embeddings_coles",
    ]
):
    os.system(f"clickhouse-client --query 'DROP TABLE IF EXISTS {table}'")
    os.system(f"clickhouse-client -mn < embeddings/{table}.create_query")
    os.system(f"cat embeddings/{table}.parquet | clickhouse-client --query 'INSERT INTO {table} FORMAT Parquet'")
