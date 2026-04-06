
import mlflow 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform
from functions.modeling_f.modeling_helpers_f import evaluate_model, confusion_matrix, model_predict
from functions.modeling_f.tunner_f import tunner_successive_halving
from functions.data_prep_f.features_helpers import build_pipeline
from functions.data_prep_f.data_helpers import data_split
from functions.modeling_f.dalex_f import (
    dx_create_explainer,
    dx_global_importance,
    dx_local_explanations,
    mlflow_log_dalex_plot,
    dx_transform
)

def exp_ada_boost_tune(
    RUN_ada_boost_tune = True,
    RUN_DALEX_GLOBAL_EXPLANATIONS = True,
    RUN_DALEX_LOCAL_EXPLANATIONS = True,
    LOG_MODEL = False
    ):

    with mlflow.start_run(run_name = "Ada Boost Tune") :

        # Split the data 
        model_data = data_split(
            train_url  = "sample_data/train.csv",
            train_size = 0.7,
            test_size  = 0.3 
        )
        # Build Pipeline 
        pipeline = build_pipeline(
            model = AdaBoostClassifier(
                estimator= DecisionTreeClassifier(max_depth=1)),
            add_tenure          = False,
            add_value_x_service = False,
            add_full_feature    = True,
            add_power_transform = True,
            add_remove_nzv      = True 
        )
        # Fit the pipeline 
        pipeline.fit(
            X = model_data["x_train"],
            y = model_data["y_train"]
        )
        # Predict on the test data 
        model_predictions = model_predict(model = pipeline, model_data = model_data)

        # Evaluate the model and create a Confusion Matrix  
        eval = evaluate_model(
            model = pipeline,
            predictions = model_predictions,
            model_data = model_data
        )
        conf_matrix = confusion_matrix(predictions = model_predictions, model_data = model_data)
        mlflow.log_metrics(eval)
        mlflow.log_figure(conf_matrix, "plots/confusion_matrix.png")

    # =========== Tune the Ada Boost Model =========================================
    if RUN_ada_boost_tune : 
        # Create a parameter grid 
        param_grid = {
            'model__n_estimators'        : randint(50, 300),
            'model__learning_rate'       : uniform(0.01, 0.2),
            'model__estimator__max_depth': [1, 2, 3], 
        }
        # Prepare the data for the Tuner 
        tunner_df = dx_transform(
            pipeline = pipeline,
            x_train  = model_data["x_train"],
            y_train  = model_data["y_train"]  
        )
        # Tune Search with Successive Halving
        search = tunner_successive_halving(
            pipeline        = pipeline,
            param_grid      = param_grid,
            X_transformed   = model_data["x_train"],
            y_train_encoded = tunner_df["y_encoded"]
        )
        search.best_score_
        search.best_params_



