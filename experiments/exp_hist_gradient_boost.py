import mlflow 
from sklearn.ensemble import HistGradientBoostingClassifier
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

def exp_hist_gradient_tune(
    RUN_hist_gradient_boost = False,
    RUN_DALEX_GLOBAL_EXPLANATIONS = True,
    RUN_DALEX_LOCAL_EXPLANATIONS = True,
    LOG_MODEL = False
    ) : 
    with mlflow.start_run(run_name = "Tuned Hist Gradient Boosting") : 
        # Split the data 
        model_data = data_split(
            train_url = "sample_data/train.csv",
            train_size = 0.7,
            test_size = 0.3 
        )
        # Create the model pipeline 
        pipeline = build_pipeline(
            model = HistGradientBoostingClassifier(),
            add_tenure           = False,
            add_value_x_service  = False,
            add_full_feature     = True,
            add_power_transform  = True,
            add_remove_nzv       = True
        )
        # Fit the pipeline 
        pipeline.fit(
            X = model_data["x_train"],
            y = model_data["y_train"]
        )
        # Predict on the test data 
        predictions  = model_predict(model = pipeline, model_data = model_data) 

        # Evaluate the model 
        eval = evaluate_model(
            model = pipeline,
            predictions = predictions,
            model_data = model_data
        ) 
        conf_matrix = confusion_matrix(predictions = predictions, model_data = model_data) 
        mlflow.log_metrics(eval)
        mlflow.log_figure(conf_matrix, "plots/confusion_matrix.png")

        # ========== Run Successive halving Tuning ================
        if RUN_hist_gradient_boost : 
            # Create paramter grid 
            hgb_imbalanced = {
                "model__max_iter"          : randint(100, 500),
                "model__learning_rate"     : uniform(0.01, 0.1),
                "model__max_leaf_nodes"    : randint(31, 100),
                "model__min_samples_leaf"  : randint(5, 20),
                "model__class_weight"      : ['balanced']
            }
            # Prepare the data 
            tunner_df = dx_transform(
                pipeline = pipeline,
                x_train  = model_data["x_train"],
                y_train  = model_data["y_train"]
            )
            # Run Successive halving 
            search = tunner_successive_halving(
                pipeline        = pipeline,
                param_grid      = hgb_imbalanced,
                X_transformed   = model_data["x_train"],
                y_train_encoded = tunner_df["y_encoded"],
                candidates      = 200
            )
