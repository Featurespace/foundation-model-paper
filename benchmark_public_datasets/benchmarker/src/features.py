from typing import List, Tuple

import pandas as pd
from clickhouse_driver import Client
from loguru import logger

from .configs.dataset import DatasetSettings

CH_SETTINGS = {"max_memory_usage": 100000000000}


def load_features(config: DatasetSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (config.numerical_features or config.categorical_features) and config.embeddings:
        # If both features and embeddings are specified
        train_engineered_features, test_engineered_features = load_hand_engineered_features(
            train_table=config.train_txn_table,
            test_table=config.test_txn_table,
            numerical_features=config.numerical_features or [],
            categorical_features=config.categorical_features or [],
        )
        train_embeddings, test_embeddings = load_embeddings(
            train_table=config.embeddings.train_emb_table, test_table=config.embeddings.test_emb_table
        )
        train_features = train_engineered_features.merge(train_embeddings, on="customerId")
        test_features = test_engineered_features.merge(test_embeddings, on="customerId")
        return train_features, test_features
    elif config.embeddings:
        # If only embeddings are specified
        train_embeddings, test_embeddings = load_embeddings(
            train_table=config.embeddings.train_emb_table, test_table=config.embeddings.test_emb_table
        )
        return train_embeddings, test_embeddings
    else:
        # If only features are specified
        train_engineered_features, test_engineered_features = load_hand_engineered_features(
            train_table=config.train_txn_table,
            test_table=config.test_txn_table,
            numerical_features=config.numerical_features or [],
            categorical_features=config.categorical_features or [],
        )
        return train_engineered_features, test_engineered_features


def load_embeddings(train_table: str, test_table: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    client = Client("localhost")
    table_schema = client.execute(f"DESCRIBE {train_table}")
    column_names = [col_name for col_name, *_ in table_schema]
    embedding_columns = [col_name for col_name in column_names if "embedding" in col_name]
    if len(embedding_columns) == 0:
        raise ValueError(f"Table {train_table} contains no embeddings")
    logger.info(f"Extracting embeddings of dimensionality {len(embedding_columns)}")

    embeddings_expr = ", ".join(embedding_columns)
    query_blueprint = """
    SELECT
        customerId,
        {embeddings_expr}
    FROM {table}
    """

    logger.info(f"Loading training embeddings from {train_table}")
    train_query = query_blueprint.format(embeddings_expr=embeddings_expr, table=train_table)
    logger.debug(train_query)
    train_embeddings = client.query_dataframe(train_query, settings=CH_SETTINGS)
    logger.info(f"Loading test embeddings from {test_table}")
    test_query = query_blueprint.format(embeddings_expr=embeddings_expr, table=test_table)
    logger.debug(test_query)
    test_embeddings = client.query_dataframe(test_query, settings=CH_SETTINGS)

    return train_embeddings, test_embeddings


def load_hand_engineered_features(
    train_table: str, test_table: str, numerical_features: List[str], categorical_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For each `customerId` compute a set of aggregate features aggregated over
    the whole history of transactions for that `customerId`.
    """
    client = Client("localhost")
    logger.info("Loading data")

    numerical_feats_expr_list = get_numerical_features_sql_expr_list(train_table, numerical_features)
    categorical_feats_expr_list = get_categorical_features_sql_expr_list(
        train_table, categorical_features, numerical_features
    )
    features_expr_list = [*numerical_feats_expr_list, *categorical_feats_expr_list]
    features_expr = ", ".join(features_expr_list)

    query_blueprint = """
    SELECT
        customerId,
        {features_expr}
    FROM {table}
    GROUP BY customerId
    """

    logger.info(f"Loading training features from {train_table} ({len(features_expr_list)} features)")
    train_query = query_blueprint.format(features_expr=features_expr, table=train_table)
    logger.debug(train_query)
    train_features = client.query_dataframe(train_query, settings=CH_SETTINGS)

    logger.info(f"Loading test features from {test_table} ({len(features_expr_list)} features)")
    test_query = query_blueprint.format(features_expr=features_expr, table=test_table)
    logger.debug(test_query)
    test_features = client.query_dataframe(test_query, settings=CH_SETTINGS)

    return train_features, test_features


def get_numerical_features_sql_expr_list(table: str, features: List[str]) -> List[str]:
    client = Client("localhost")
    logger.info(f"Extracting features for numerical columns: {features}")

    numerical_columns = [col for col, dtype, *_ in client.execute(f"DESCRIBE {table}") if "Float" in dtype]
    if len(features) != len(set(numerical_columns).intersection(features)):
        raise ValueError(
            " Incorrect numerical column name(s) \n"
            f" - expected features: {features}\n"
            f" - numerical features in {table}: {numerical_columns}"
        )

    sql_expr_list = ["count() AS total_count"]
    for feature in features:
        sql_expr_list.extend(
            [
                f"sum({feature}) AS total_sum_{feature}",
                f"avg({feature}) AS total_mean_{feature}",
                f"varPop({feature}) AS total_var_{feature}",
                f"min({feature}) AS total_min_{feature}",
                f"max({feature}) AS total_max_{feature}",
            ]
        )
    logger.info(f"Number of features for numerical columns: {len(sql_expr_list)}")
    return sql_expr_list


def get_distinct_feature_values(table: str, feature: str, top_p: float = 0.98) -> List[str]:
    """Extract most common categories for a given categorical feature that
    together account for `top_p` fraction of transactions
    """
    client = Client("localhost")
    query = f"""
    WITH (SELECT count() FROM {table}) AS total_count
    SELECT
        {feature},
        count() / total_count AS prop
    FROM {table}
    GROUP BY {feature}
    ORDER BY prop DESC
    """
    logger.debug(query)

    top_values = []
    values_and_their_proportions = client.execute(query)
    cumulative_txn_prop = 0.0
    for value, txn_proportion in values_and_their_proportions:
        top_values.append(str(value))
        cumulative_txn_prop += txn_proportion
        if cumulative_txn_prop > top_p:
            break

    logger.info(f"Feature {feature}: {len(top_values)} top-{top_p} values (out of {len(values_and_their_proportions)})")
    return top_values


def get_categorical_features_sql_expr_list(
    table: str, features: List[str], numerical_features: str = "amt"
) -> List[str]:
    client = Client("localhost")
    logger.info(f"Extracting features for categorical columns: {features}")

    categorical_columns = [col for col, dtype, *_ in client.execute(f"DESCRIBE {table}") if "Float" not in dtype]
    if len(features) != len(set(categorical_columns).intersection(features)):
        raise ValueError(
            " Incorrect categorical column name(s) \n"
            f" - expected features: {features}\n"
            f" - categorical features in {table}: {categorical_columns}"
        )

    sql_expr_list = []
    for feature in features:
        distinct_feature_values = get_distinct_feature_values(table, feature)
        for value in distinct_feature_values:
            value = value.replace(".", "_")
            sql_expr_list.append(f"\ncountIf({feature} = '{value}') AS count_{feature}_{value}")
            sql_expr_list.extend(
                [
                    f"avgIf({num_feature_name}, {feature} = '{value}') AS mean_{num_feature_name}_{feature}_{value}"
                    for num_feature_name in numerical_features
                ]
            )
            sql_expr_list.extend(
                [
                    f"varPopIf({num_feature_name}, {feature} = '{value}') AS var_{num_feature_name}_{feature}_{value}"
                    for num_feature_name in numerical_features
                ]
            )
    logger.info(f"Number of features for categorical columns: {len(sql_expr_list)}")
    return sql_expr_list
