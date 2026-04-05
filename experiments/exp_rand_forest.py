
import mlflow 
from functions.modeling_f.modeling_helpers_f import evaluate_model, confusion_matrix, model_predict
from scipy.stats import randint, uniform
from functions.modeling_f.tunner_f import tunner_random
from functions.data_prep_f.features_helpers import build_pipeline
from functions.data_prep_f.data_helpers import data_split
from sklearn.ensemble import RandomForestClassifier 
from functions.modeling_f.dalex_f import (
    dx_create_explainer,
    dx_global_importance,
    dx_local_explanations,
    mlflow_log_dalex_plot,
    dx_transform
)
import pandas as pd 
def exp_random_forest_tune(RUN_tune_random_forest = True) :

    with mlflow.start_run(run_name="Random Forest Test") : 
        
        # Split the data 
        model_data = data_split(
            train_url  ="sample_data/train.csv",
            train_size = 0.7,
            test_size  = 0.3
        )

         # Build Model Pipeline 
        pipeline = build_pipeline(
            model               = RandomForestClassifier(),
            add_tenure          =False,
            add_value_x_service =False,
            add_full_feature    =True,
            add_power_transform =True,
            add_remove_nzv      =True
        )
        
        # Fit the pipeline 
        pipeline.fit(
            X = model_data["x_train"],
            y = model_data["y_train"]
        )
        # ======== Random grid Search ==============
        if RUN_tune_random_forest :

            # Declare a parameter Grid 
            param_grid = {
                "model__n_estimators"      : randint(500, 1000),      
                "model__max_depth"         : randint(5, 50),         
                "model__min_samples_split" : randint(2, 20),         
                "model__min_samples_leaf"  : randint(1, 10),          
                "model__max_features"      : uniform(0.1, 0.9),       
                "model__min_impurity_decrease" : uniform(0.0, 0.05), 
            }
            # Prepare the data for the Tuner 
            tunner_df = dx_transform(
                pipeline = pipeline,
                x_train  = model_data["x_train"],
                y_train  = model_data["y_train"]  
            )
            # Random Search 
            search = tunner_random(
                pipeline        = pipeline,
                param_grid      = param_grid,
                X_transformed   = model_data["x_train"],
                y_train_encoded = tunner_df["y_encoded"],
                iters = 50
            )
            best_params = search.best_params_