from typing import List

from pydantic import BaseModel, Field, PositiveInt


class ModelSettings(BaseModel):
    hidden_layers: List[PositiveInt]
    dropout_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    l1: float = Field(default=0.0, ge=0.0)
    l2: float = Field(default=0.0, ge=0.0)
