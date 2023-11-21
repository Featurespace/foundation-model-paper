from typing import List, Optional

from pydantic import BaseModel, root_validator


class EmbeddingSettings(BaseModel):
    train_emb_table: str
    test_emb_table: str


class DatasetSettings(BaseModel):
    """The `train_txn_table` and `test_tsn_table` tables should contain
    transactions for two disjoint sets of entities identified in `customerId`
    column. The `targets_table` column should contain entity-level labels for
    both train and test tables, having columns: `customerId`, `label`. It can
    include more columns which will be used as features.
    """

    train_txn_table: str
    test_txn_table: str
    targets_table: str
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    embeddings: Optional[EmbeddingSettings] = None

    @root_validator
    def check_there_is_at_least_one_source_of_features(cls, values):
        if (
            values.get("embeddings") is None
            and values.get("numerical_features") is None
            and values.get("categorical_features") is None
        ):
            raise ValueError("You have to provide at least one source of features for the model.")
        return values
