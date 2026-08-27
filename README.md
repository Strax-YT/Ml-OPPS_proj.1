# SMS Spam Classifier — MLOps Pipeline

A binary spam/ham text classifier for SMS messages, built as a reproducible [DVC](https://dvc.org/) pipeline with experiment tracking via [DVCLive](https://dvc.org/doc/dvclive).

## Pipeline

```
data_ingestion -> data_preprocessing -> feature_engineering -> model_building -> model_evaluation
```

| Stage | Script | What it does |
|---|---|---|
| `data_ingestion` | [src/data_ingestion.py](src/data_ingestion.py) | Downloads the raw dataset, splits into train/test (`data_ingestion.test_size`), writes to `data/raw/` |
| `data_preprocessing` | [src/data_preprocessing.py](src/data_preprocessing.py) | Lowercases, tokenizes, strips stopwords/punctuation, stems (NLTK), encodes labels, drops duplicates; writes to `data/interim/` |
| `feature_engineering` | [src/feature_engineering.py](src/feature_engineering.py) | TF-IDF vectorization (`feature_engineering.max_features`); writes to `data/processed/` |
| `model_building` | [src/model_building.py](src/model_building.py) | Trains a `RandomForestClassifier` (`model_building.n_estimators`, `model_building.random_state`); writes `models/model.pkl` |
| `model_evaluation` | [src/model_evaluation.py](src/model_evaluation.py) | Scores the test set (accuracy, precision, recall, AUC), logs the run with DVCLive, writes `reports/metrics.json` |

All stage parameters live in [params.yaml](params.yaml). Every stage logs to `logs/<stage>.log`.

## Setup

```bash
git clone https://github.com/Strax-YT/Ml-OPPS_proj.1.git
cd Ml-OPPS_proj.1
pip install -r requirements.txt
```

## Running the pipeline

```bash
dvc repro
```

This runs only the stages whose code, data, or params have changed since the last run. Check the pipeline graph and latest metrics with:

```bash
dvc dag
dvc metrics show
```

## Running experiments

Adjust a value in `params.yaml` (e.g. `feature_engineering.max_features`), then:

```bash
dvc exp run
dvc exp show
```

Compare, apply, or discard experiments with `dvc exp diff`, `dvc exp apply <exp-name>`, and `dvc exp remove <exp-name>`.

## Project structure

```
src/
  data_ingestion.py
  data_preprocessing.py
  feature_engineering.py
  model_building.py
  model_evaluation.py
data/            # DVC-managed pipeline data (git-ignored)
models/          # trained model artifacts (git-ignored)
reports/         # metrics.json (git-tracked, for dvc metrics diff across commits)
dvclive/         # per-run experiment tracking output (git-ignored)
logs/            # per-stage run logs
experiments/     # exploratory notebook and scratch work
params.yaml      # pipeline parameters
dvc.yaml         # pipeline stage definitions
```
