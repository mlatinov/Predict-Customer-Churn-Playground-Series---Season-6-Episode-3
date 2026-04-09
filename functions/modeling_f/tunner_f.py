import tempfile
import os
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV ,HalvingRandomSearchCV
import optuna
import optuna.visualization as viz
from sklearn.ensemble import HistGradientBoostingClassifier

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
