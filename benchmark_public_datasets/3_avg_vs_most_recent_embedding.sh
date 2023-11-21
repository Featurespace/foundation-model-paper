#!/usr/bin/env bash

# This script reproduces results from Table 4 in the paper
# It compares the performance of models trained using average embeddings
# vs embeddings produced on the most recent transaction for each sequence.

set -eu

export CONDA_ENV="public_benchmarker"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

NUM_RUNS=50
MAX_PARALLEL_JOBS=1

SCRIPT_PATH=$(realpath ${BASH_SOURCE[0]})
export SCRIPT_DIR=$(dirname $SCRIPT_PATH)

export OPTUNA_SEARCH_SPACE="$(cat $SCRIPT_DIR/configs/optuna_search_space.yaml | yq)"
export OPTUNA_STORAGE="sqlite:///paper.db"
export OPTUNA_SAMPLER="TPESampler"

declare -A METRICS=(
    [rosbank]=auc
    [sber]=accuracy
    [retail]=accuracy
    [alpha]=auc
)

declare -A BATCH_SIZES=(
    [rosbank]=256
    [sber]=256
    [retail]=512
    [alpha]=1024
)

declare -A NP_NE_METHOD_NAME_MAPPING=(
    [rosbank]=np_ne
    [sber]=np_ne_0_001
    [retail]=np_ne_0_005
    [alpha]=np_ne_0_001
)

for DATASET in "rosbank" "sber" "retail_expenditure" "alpha"
do
    for METHOD in "np_ne" "coles"
    do
        for EMBEDDING_MODE in "avg" "most_recent"
        do
            if [[ $METHOD = "np_ne" ]]
            then
                export METHOD=${NP_NE_METHOD_NAME_MAPPING[$DATASET]}
            fi

            export OPTUNA_STUDY_NAME="embedding_mode_${DATASET}_${METHOD}"
            export CACHE=$SCRIPT_DIR/caches/embedding_mode/${DATASET}_${METHOD}.pickle
            export BATCH_SIZE=${BATCH_SIZES[${DATASET%_expenditure}]}
            export DATASET_CONFIG=$SCRIPT_DIR/configs/dataset_with_embeddings.yaml

            if [[ ($DATASET = "retail"* ) || ($DATASET = "alpha"*) ]]
            then
                export MODEL_TRAIN_CONFIG=$SCRIPT_DIR/configs/model_train_large.yaml
            else
                export MODEL_TRAIN_CONFIG=$SCRIPT_DIR/configs/model_train.yaml
            fi

            if [[ $DATASET = *"expenditure" ]]
            then
                OPTIMIZATION_DIRECTION="minimize"
                export METRIC="msle"
            else
                OPTIMIZATION_DIRECTION="maximize"
                export METRIC=${METRICS[$DATASET]}
            fi

            if [[ $EMBEDDING_MODE = "avg" ]]
            then
                export TRAIN_TABLE="NPPR_PAPER.${DATASET}_train_avg_embeddings_${METHOD}"
                export TEST_TABLE="NPPR_PAPER.${DATASET}_test_avg_embeddings_${METHOD}"
                export TARGETS_TABLE="NPPR_PAPER.${DATASET}_targets"
            else
                export TRAIN_TABLE="NPPR_PAPER.${DATASET}_train_embeddings_${METHOD}"
                export TEST_TABLE="NPPR_PAPER.${DATASET}_test_embeddings_${METHOD}"
                export TARGETS_TABLE="NPPR_PAPER.${DATASET}_targets"
            fi

            if [[ $DATASET = "alpha" ]]
            then
                export CLASS_WEIGHTS="{0: 1.0, 1: 35.71}"
                export EXTRA_ARGUMENTS='train.class_weights=${oc.decode:${oc.env:CLASS_WEIGHTS}}'
            fi

            optuna create-study \
                --storage $OPTUNA_STORAGE \
                --study-name $OPTUNA_STUDY_NAME \
                --direction $OPTIMIZATION_DIRECTION

            LOG_DIR=$SCRIPT_DIR/logs/optuna_${OPTUNA_STUDY_NAME} && mkdir -p $LOG_DIR
            sbatch \
                --array=1-${NUM_RUNS}%${MAX_PARALLEL_JOBS} \
                --gres=gpu:1 \
                --job-name=opt:${DATASET}:${METHOD} \
                -o $LOG_DIR/log%a.log \
                -e $LOG_DIR/log%a.log \
                $SCRIPT_DIR/srun_optuna_train_eval.sh
        done
    done
done
