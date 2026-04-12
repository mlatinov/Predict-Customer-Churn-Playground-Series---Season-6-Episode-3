
import tempfile
import os
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV ,HalvingRandomSearchCV
import optuna
import optuna.visualization as viz
from functions.modeling_f.nn_helper import build_deep_mlp
from functions.data_prep_f.features_helpers import build_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from scikeras.wrappers import KerasClassifier
import tensorflow as tf 
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def tunner_random(pipeline, param_grid, X_transformed, y_train_encoded, iters = 50) :
    
    # Random Search
    search = RandomizedSearchCV(
        estimator           =  pipeline,
        param_distributions =  param_grid,
        n_iter              =  iters,
        scoring             = "roc_auc",
        n_jobs              = -1,
        random_state        = 42,
        cv                  = 5
    ) 
    # Fit the Random Search on the Transformed Data 
    search.fit(X_transformed, y_train_encoded)
    return search

def tunner_successive_halving(pipeline, param_grid, X_transformed, y_train_encoded,candidates = 100) :

    #  Successive halving
    search = HalvingRandomSearchCV(
        estimator           = pipeline,
        param_distributions = param_grid,
        scoring             = "roc_auc",
        min_resources       = "smallest", 
        factor              = 3,
        n_candidates        = candidates, 
        n_jobs              = 2, 
        random_state        = 42,
        verbose             = 1,
        cv                  = 5 
    )
    # Fit the Successive halving on the Transformed Data 
    search.fit(X_transformed, y_train_encoded)
    return search

def opt_tunner(model_name, x_train, y_train, n_trials) :

    if model_name == "HistGradientBoostingClassifier" :

        def objective(trial, x_train = x_train, y_train = y_train) :
            # Declare parameters 
            params = {
                "max_iter"       : trial.suggest_int("max_iter", 300, 700),
                "learning_rate"  : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_leaf_nodes" : trial.suggest_int("max_leaf_nodes", 20, 100),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20)
            }
            # Build the model 
            model = HistGradientBoostingClassifier(**params, random_state=42, warm_start = True)

            # Evaluate with Cross Validation 
            scores = cross_val_score(
                estimator = model,
                X = x_train,
                y = y_train,
                cv      = 5,
                scoring = "roc_auc",
                n_jobs  = 1
            )
            return scores.mean() 
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials = n_trials, show_progress_bar = True)

    return study

# ========================== DEEP MLP Optimization ========================= 
def deep_mlp_objective(trial, X_train, y_train):

    # Guarantee numpy int — scikeras task detection depends on this
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train, dtype=int)
    else:
        y_train = y_train.astype(int)

    # Number of Layers 
    n_layers     = trial.suggest_int("n_layers", 3, 6)
            
    # Number of units per Layer
    hidden_units = [
        trial.suggest_int(f"units_layer_{i}", 128, 256, step = 32)
        for i in range(n_layers)
        ]
    # Other Hyperparams 
    dropout_rate  = trial.suggest_float("dropout_rate",  0.1,  0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    activation    = trial.suggest_categorical("activation", ["relu", "elu"])

    # Build wrapped model with suggested params
    nn = KerasClassifier(
        model            = build_deep_mlp,
        hidden_units     = hidden_units,
        dropout_rate     = dropout_rate,
        activation       = activation,
        learning_rate    = learning_rate,
        loss             = "binary_crossentropy",
        epochs           = 50,
        validation_split = 0.2,
        verbose          = 0,
        callbacks        = [tf.keras.callbacks.EarlyStopping(
            monitor             = "val_loss",
            patience            = 5,
            restore_best_weights = True
        )]
    )

    # Rebuild pipeline with suggested NN
    trial_pipeline = build_pipeline(
        model               = nn,
        add_tenure          = False,
        add_value_x_service = False,
        add_full_feature    = True,
        add_power_transform = True,
        add_remove_nzv      = True
    ) 
        # ── Manual CV — bypasses sklearn scorer machinery entirely ─────────────
    skf    = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):

        # Handle both DataFrame and numpy inputs
        if hasattr(X_train, "iloc"):
            X_tr  = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
        else:
            X_tr  = X_train[train_idx]
            X_val = X_train[val_idx]

        y_tr  = y_train[train_idx]
        y_val = y_train[val_idx]

        trial_pipeline.fit(X_tr, y_tr)

        # Call predict_proba directly — no sklearn scorer involved
        y_prob = trial_pipeline.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, y_prob))

    return np.mean(scores)

             
def deep_mlp_tune_optuna(X_train, y_train, n_trials = 30) :
    # Create a Optuna Study 
    study = optuna.create_study(
        direction  = "maximize",
        study_name = "Deep_MLP",
        sampler    = optuna.samplers.TPESampler(seed=42)
    )
    # Run the optimization 
    study.optimize(
        lambda trial: deep_mlp_objective(trial, X_train, y_train),
        n_trials  = n_trials,
        show_progress_bar = True
    )
    # Interactive Print Result 
    print(f"Best AUC:    {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
                
    return study



# ================== Tuning Viz Function and Plot Loging =================================
def opt_tune_viz(opt_study, hyper_params) :

    # Parameter Importance 
    imp_params = viz.plot_param_importances(opt_study)

    # Parallel Coordinate Plot
    parallel_c = viz.plot_parallel_coordinate(opt_study)

    # Contour Plot
    contour_p = viz.plot_contour(opt_study, params= hyper_params)

    # Combine into dict 
    results = {
        "parameter_importance" : imp_params,
        "cordinate_plot"       : parallel_c,
        "countour_plot"        : contour_p
    }
    return results

def mlflow_log_optuna_plot(optuna_result, filename, artifact_path="optuna"):
    """
    Save a Optuna plot as an interactive HTML artifact and log it to MLflow.
    """
    fig = optuna_result.plot(show=False)   

    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, filename)
        fig.write_html(filepath)
        mlflow.log_artifact(filepath, artifact_path=artifact_path)
