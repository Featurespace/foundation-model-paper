# Extracting public datasets

To create extracts of public datasets you'll need a virtual environment with `python`, `pandas`, `loguru`, `tqdm`, and `clickhouse_driver`. Within this environment, perform the following steps:
1. Download datasets using links in `sources.txt`.
2. Unpack these datastes into the following folders within `public_datasets`:
    * `age_group_prediction` - SberBank dataset,
    * `default_prediction_on_credit_cards` - Alpha Bank dataset,
    * `rosbank` - Rosbank dataset,
    * `retail` - X5 group dataset.
3. Run `1_load_datasets_into_clickhouse.py` - this will import datasets into ClickHouse in the same format that we used for our experiments.
4. Run `2_create_train_test_split.py`.
5. Download an archive containing embeddings and unpack it in `public_datasets`. The archive will be available upon request at piotr [dot] skalski [at] featurespace [dot] co [dot] uk
6. Run `3_import_embeddings.py` - This will import embeddings from models pretrained with different algorithms used for benchmarking.
