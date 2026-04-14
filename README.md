# Predict Customer Churn — Kaggle Playground Series S6E3

A machine learning pipeline for predicting customer churn, built for the [Kaggle Playground Series Season 6, Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3) competition.

## Overview

The project trains and stacks multiple classifiers to predict the probability that a customer will churn. Experiments are tracked with **MLflow** and the final submission is produced from either individual models or a stacked ensemble.

## Models

| Model | File |
|---|---|
| Random Forest (tuned) | `experiments/exp_rand_forest.py` |
| Histogram Gradient Boosting | `experiments/exp_hist_gradient_boost.py` |
| Deep MLP (Neural Network) | `experiments/exp_deep_mlp.py` |
| AdaBoost | `experiments/exp_ada_boost.py` |
| Logistic Regression | `experiments/exp_logistic_reg.py` |
| **Stacked Ensemble** (LightGBM meta-learner) | `experiments/exp_stack.py` |

## Project Structure

```
├── main_pipeline.py          # Entry point: trains the stack and generates submissions
├── experiments/              # One file per model experiment
├── functions/
│   ├── data_prep_f/          # Feature engineering & data helpers
│   ├── modeling_f/           # Evaluation, tuning, DALEX explainability
│   └── data_viz_f/           # Visualisation helpers
└── sample_data/
    ├── train.csv
    └── test.csv
```

## Feature Engineering

Features are built through configurable sklearn-compatible pipelines:

- **Tenure features** — customer lifecycle buckets, new-customer flag
- **Value & service features** — value gap (pricing signal), service count, premium-user flag
- **Payment features** — automatic payment flag
- **Preprocessing** — Yeo-Johnson power transform, near-zero-variance removal, standard scaling, one-hot encoding

## Running

1. Place `train.csv` and `test.csv` in `sample_data/`.
2. Train individual experiments first (they register models in the MLflow registry).
3. Run the main pipeline:

```bash
python main_pipeline.py
```

This produces five submission CSVs: `stack_submission.csv`, `rf_submission.csv`, `hist_submission.csv`, `nn_submission.csv`, and `ada_submission.csv`.

## Experiment Tracking

MLflow is used to log metrics, parameters, confusion matrices, and DALEX explainability plots.

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

