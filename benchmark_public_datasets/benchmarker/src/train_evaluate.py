import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import click
import numpy as np
import tensorflow as tf
from loguru import logger
from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import (
    AUC,
    BinaryAccuracy,
    MeanAbsoluteError,
    MeanSquaredLogarithmicError,
    RootMeanSquaredError,
    SparseCategoricalAccuracy,
)
from tensorflow.keras.optimizers import Adam

from .configs.train import TrainConfig
from .datasets import load_datasets
from .logging import LoggingCallback
from .model import create_mlp

MAX_CLASSES_FOR_CLASSIFICATION = 10


@click.command()
@click.option("--config", "config_files", type=Path, multiple=True, required=True)
@click.option("--debug", "debug", default=False, is_flag=True)
@click.option("--cache", "cache", type=Path, required=False)
@click.argument("overrides", nargs=-1)
def train_evaluate(config_files: List[str], overrides: List[str], debug: bool, cache: Optional[Path] = None):
    set_up_logger(debug)
    logger.info("Running train-evaluate function")
    conf_dict = parse_config_files(config_files, overrides)
    config = TrainConfig(**conf_dict)

    (train_y, train_X), (test_y, test_X) = load_datasets(config.dataset, cache)

    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)

    for train_idxs, test_idxs in KFold(config.train.num_cross_val_folds, shuffle=True).split(train_X):
        train_X_fold, train_y_fold = train_X[train_idxs], train_y[train_idxs]
        val_X_fold, val_y_fold = train_X[test_idxs], train_y[test_idxs]

        num_uniq_targets = len(np.unique(train_y))
        if num_uniq_targets >= MAX_CLASSES_FOR_CLASSIFICATION:  # regression
            num_output_neurons = 1
            out_activation = None
        elif num_uniq_targets > 2:  # multi-class classification
            num_output_neurons = num_uniq_targets
            out_activation = "softmax"
        else:  # binary classification
            num_output_neurons = 1
            out_activation = "sigmoid"

        model = create_mlp(
            config.model,
            n_input_features=train_X.shape[1],
            n_out_neurons=num_output_neurons,
            out_activation=out_activation,
        )
        model.compile(
            optimizer=Adam(config.train.learning_rate),
            loss=get_loss(num_uniq_targets),
            metrics=[get_metric(config.train.metric, num_output_neurons)],
        )
        model.summary(print_fn=logger.info)

        callbacks = []
        callbacks.append(LoggingCallback(print_fn=logger.info))
        if config.train.early_stopping:
            early_stopping_mode = "min" if config.train.metric in ["rmse", "mae", "msle"] else "max"
            callbacks.append(
                EarlyStopping(
                    monitor=f"val_{config.train.metric}",
                    patience=config.train.patience,
                    mode=early_stopping_mode,
                    restore_best_weights=True,
                )
            )

        model.fit(
            x=train_X_fold,
            y=train_y_fold,
            batch_size=config.train.batch_size,
            epochs=config.train.num_epochs,
            verbose=0,
            callbacks=callbacks,
            validation_data=(val_X_fold, val_y_fold),
            shuffle=True,
            class_weight=config.train.class_weights,
        )

        val_fold_metrics = model.evaluate(val_X_fold, val_y_fold, return_dict=True, verbose=0)
        for metric, value in val_fold_metrics.items():
            val_metrics[metric].append(value)
        logger.info(f"--- Cross-validation fold metrics ---\n{json.dumps(val_fold_metrics, indent=2)}")

        test_fold_metrics = model.evaluate(test_X, test_y, return_dict=True, verbose=0)
        for metric, value in test_fold_metrics.items():
            test_metrics[metric].append(value)

    avg_val_metrics = {metric: {"avg": np.mean(vals), "std": np.std(vals)} for metric, vals in val_metrics.items()}
    avg_test_metrics = {metric: {"avg": np.mean(vals), "std": np.std(vals)} for metric, vals in test_metrics.items()}

    logger.info(f"--- Cross-validation results ---\n{json.dumps(avg_val_metrics, indent=2)}")
    logger.info(f"--- Test set results ---\n{json.dumps(avg_test_metrics, indent=2)}")

    # This metric can be used in downstream hyperparameter tuning
    click.echo(avg_val_metrics[config.train.metric]["avg"])


def get_metric(metric_name: str, num_output_neurons: int) -> tf.keras.metrics.Metric:
    if metric_name == "accuracy" and num_output_neurons > 1:
        return SparseCategoricalAccuracy(name=metric_name)
    elif metric_name == "accuracy":
        return BinaryAccuracy(name=metric_name)
    elif metric_name == "auc":  # Use only with binary classification problem
        return AUC(num_thresholds=100_000, name=metric_name)
    elif metric_name == "rmse":
        return RootMeanSquaredError(name=metric_name)
    elif metric_name == "mae":
        return MeanAbsoluteError(name=metric_name)
    elif metric_name == "msle":
        return MeanSquaredLogarithmicError(name=metric_name)
    else:
        raise ValueError(f"Unknown metric `{metric_name}`")


def get_loss(num_uniq_targets: int) -> tf.keras.losses.Loss:
    if num_uniq_targets > 10:
        return tf.keras.losses.MeanSquaredLogarithmicError()
    if num_uniq_targets > 2:
        return tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        return tf.keras.losses.BinaryCrossentropy()


def parse_config_files(config_files: List[str], overrides: List[str]) -> Dict[str, Any]:
    confs = [OmegaConf.load(config_file) for config_file in config_files]
    parsed_overrides = OmegaConf.from_dotlist(list(overrides))
    conf = OmegaConf.merge(*confs, parsed_overrides)
    logger.info(f"Parsed config:\n{OmegaConf.to_yaml(conf, resolve=True)}")
    conf_dict = OmegaConf.to_container(conf, resolve=True)
    return cast(Dict[str, Any], conf_dict)


def set_up_logger(debug: bool):
    logging_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " "<level>{level}</level> | " "<level>{message}</level>"
    )
    logger.remove()
    logger.add(sys.stderr, format=logging_format, level="DEBUG" if debug else "INFO")
