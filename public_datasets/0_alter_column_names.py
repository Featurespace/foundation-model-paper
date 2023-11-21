from clickhouse_driver import Client
from tqdm import tqdm

client = Client("localhost")

tables = [
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

recreate_table_query = """
    CREATE TABLE {table}
    ENGINE = MergeTree()
    ORDER BY (customerId, transactionTime)
    AS SELECT
        * EXCEPT (customerId, TenantId, EventId, PredictTime),
        customerId AS customerId,
        EventId AS transactionId,
        PredictTime AS transactionTime
    FROM {table}_backup
    ORDER BY customerId, transactionTime ASC
"""

for table_name in tqdm(tables):
    client.execute(f"RENAME TABLE {table_name} TO {table_name}_backup")
    client.execute(recreate_table_query.format(table=table_name))
    client.execute(f"DROP TABLE {table_name}_backup")
