from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, root_validator

from .dataset import DatasetSettings
from .model import ModelSettings


class TrainSettings(BaseModel):
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    num_epochs: PositiveInt
    early_stopping: bool = False
    patience: Optional[int] = Field(default=None, ge=0)
    logging_frequency: PositiveInt = 10
    class_weights: Optional[Dict[int, float]] = None
    num_cross_val_folds: int = Field(gt=1)
    metric: Literal["accuracy", "auc", "rmse", "mae", "msle"]

    @root_validator
    def check_patience_set_if_early_stopping(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        early_stopping = values.get("early_stopping")
        patience = values.get("patience")
        if early_stopping == (patience is None):
            raise ValueError("Patience needs to be set if and only if early stopping is enabled")
        return values

    @root_validator
    def check_early_stopping_patience_is_sensible(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        num_epochs = values.get("num_epochs")
        patience = values.get("patience")
        if patience is not None and patience >= num_epochs:
            raise ValueError("Patience is greater or equal than num_epochs. Early stopping won't take effect")
        return values


class TrainConfig(BaseModel):
    dataset: DatasetSettings
    model: ModelSettings
    train: TrainSettings
