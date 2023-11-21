#!/usr/bin/env bash
set -eu

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Retrieve new hyperparameters for the next tuning round
SUGGESTED_PARAMS=$( \
    optuna ask \
    --storage "$OPTUNA_STORAGE" \
    --study-name "$OPTUNA_STUDY_NAME" \
    --sampler "$OPTUNA_SAMPLER" \
    --search-space "$OPTUNA_SEARCH_SPACE" \
)
TRIAL_NUMBER=$(echo $SUGGESTED_PARAMS | jq '.number')
ARGUMENTS=$(echo $SUGGESTED_PARAMS | jq -r '.params | [to_entries[] | "\(.key)=\(.value)"] | join(" ")')

# Run the benchmarker and extract the return value which is
# the performance of the model from cross-validation
RETURN_VALUE=$( \
    benchmarker \
        --config $MODEL_TRAIN_CONFIG \
        --config $DATASET_CONFIG \
        ${CACHE:+--cache $CACHE} \
        $ARGUMENTS \
        ${EXTRA_ARGUMENTS:+$EXTRA_ARGUMENTS} \
)

# Report the performance of this set of hyperparameters
optuna tell \
    --storage "$OPTUNA_STORAGE" \
    --study-name "$OPTUNA_STUDY_NAME" \
    --trial-number "$TRIAL_NUMBER" \
    --values "$RETURN_VALUE" \
    --state complete
