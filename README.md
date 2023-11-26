# Towards a Foundation Purchasing Model: Pretrained Generative Autoregression on Transaction Sequences

This is the official code implementation of the following manuscript:

[Skalski P., Sutton D., Burrell S., Perez I., Wong J. _"Towards a Foundation Purchasing Model: Pretrained Generative Autoregression on Transaction Sequences"_](https://dl.acm.org/doi/10.1145/3604237.3626850).

It contains code to reproduce evaluations on public datasets and is distributed under a Creative Commons Attribution-NonCommercial 4.0 International license.

## Running the code

### Installation

To run the code in this repository, install the _benchmarker_ library inside a new virtual environment by running
```bash
$ pip install benchmark_public_datasets/benchmarker
```
You will also need to install and lunch Slurm for job scheduling and a ClickHouse server that will be used for storing datasets.

### Data preparation

Before running the code, prepare the datasets by following instructions in `public_datasets/README.md`

### Running evaluations

To benchmark hand-engineered features and embeddings extracted using different algorithms (Table 2 in the paper) run `benchmark_public_datasets/1_benchmark_algorithms.sh`.

To perform ablation study comparing performance of NPPR method with _next event prediction_ and _past reconstruction_ tasks used in isolation (Table 3 in the paper) run `benchmark_public_datasets/2_ablate_tasks_in_np_ne_method.sh`.

To compare performance of _"most recent"_ vs _"average"_ embedding modes (Table 4 in the paper) run `benchmark_public_datasets/3_avg_vs_most_recent_embeddings.sh`
