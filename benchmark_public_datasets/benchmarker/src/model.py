from typing import Literal, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import L1L2

from .configs.model import ModelSettings


def create_mlp(
    settings: ModelSettings,
    n_input_features: int,
    n_out_neurons: int,
    out_activation: Optional[Literal["softmax", "sigmoid"]] = None,
) -> tf.keras.Model:
    inputs = Input(shape=(n_input_features,))
    outputs = inputs
    for i, units in enumerate(settings.hidden_layers):
        outputs = Dense(
            units, activation="relu", kernel_regularizer=L1L2(l1=settings.l1, l2=settings.l2), name=f"dense_{i}"
        )(outputs)
        outputs = Dropout(settings.dropout_rate, name=f"dropout_{i}")(outputs)
    outputs = Dense(n_out_neurons, activation=out_activation, name="dense_final")(outputs)
    return tf.keras.Model(inputs, outputs, name="MLP")
