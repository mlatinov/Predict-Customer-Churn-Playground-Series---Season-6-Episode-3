
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
    dx_transform,
    dx_residual_analysis
)

def exp_random_forest_tune(
    RUN_tune_random_forest = False,
    RUN_DALEX_GLOBAL_EXPLANATIONS = True,
    RUN_DALEX_LOCAL_EXPLANATIONS = True,
    LOG_MODEL = False
    ) :

    with mlflow.start_run(run_name="Tuned Random Forest ") : 
        
        # Split the data 
        model_data = data_split(
            train_url  ="sample_data/train.csv",
            train_size = 0.7,
            test_size  = 0.3
        )

        # Build Model Pipeline with 
        # Already Tuned parameters Have to Run Tune again if data changes 
        pipeline = build_pipeline(
            model = RandomForestClassifier(
                max_depth             = 29,
                max_features          = 0.6972,
                n_estimators          = 809,
                min_impurity_decrease = 0.0003,
                min_samples_leaf      = 2,
                min_samples_split     = 3 
            ),
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
        # Run predictions with the model 
        model_predictions = model_predict(model = pipeline, model_data = model_data)

        # Evaluate the model 
        eval = evaluate_model(
            model       = pipeline,
            predictions = model_predictions,
            model_data  = model_data
        )
        # Create a Confusion Matirx 
        conf_matrix = confusion_matrix(predictions= model_predictions, model_data = model_data)
        mlflow.log_metrics(eval)
        mlflow.log_figure(conf_matrix, "plots/confusion_matrix.png")

        # ======== Random grid Search ==============
        # Run Only if the data Changes The best parameters are already in the model 
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
            # Refit the pipeline with the best parameters 
            pipeline = search.best_estimator_

        if RUN_DALEX_GLOBAL_EXPLANATIONS or RUN_DALEX_LOCAL_EXPLANATIONS : 
            # Create a Dalex Explainer
            dx_explainer = dx_create_explainer(
                pipeline = pipeline,
                x_train  = model_data["x_train"],
                y_train  = model_data["y_train"],
                label    = "Tuned Random Forest Model "
            )
            features_names  = ["tenure","MonthlyCharges","TotalCharges","value_gap"]
            
            # ============= Global DALEX Explanations ===================
        if RUN_DALEX_GLOBAL_EXPLANATIONS : 
            gfi = dx_global_importance(
                dalex_explainer = dx_explainer,
                features_names  = features_names
            )
            # Log Global Explanations Plots
            mlflow_log_dalex_plot(gfi["Loss_Shuffle_Plot"],"variable_importance.html",  "dalex/global")
            mlflow_log_dalex_plot(gfi["PDP_Plot"]         ,"partial_dependence.html",   "dalex/global")
            mlflow_log_dalex_plot(gfi["RCDR_Plot"]        ,"residuals_rcdf.html",       "dalex/global")
            
        # =============== Local DALEX Explanations =================
        if RUN_DALEX_LOCAL_EXPLANATIONS : 
            lme = dx_local_explanations(
                dalex_explainer = dx_explainer,
                pipeline        = pipeline,
                x_train = model_data["x_train"],
                y_train = model_data["y_train"],
                features_names = features_names
            )
            # Log Local Explanations Plots 
            mlflow_log_dalex_plot(lme["bd_plot_1"]  ,"breakdown_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["bd_plot_2"]  ,"breakdown_no_churn.html",   "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_1"]  ,"ceretis_paribus_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_2"]  ,"ceretis_paribus_no_churn.html",   "dalex/local")
            
        # ============ Log The model ==================
        if LOG_MODEL : 
            mlflow.sklearn.log_model(
                sk_model              = pipeline,
                name                  = "tuned_random_forest",   
                input_example         = model_data["x_train"].head(5),  
                registered_model_name = "tuned_random_forest"
            )
            # ============ Experimental Settings and Loging  ===============
        exp_params = {
            "model"           : "Random Forest",
            "model_type"      : "Ensemble Tree",
            "model_params"    : "Tuned",
            "train_data_size" :  0.7,
            "test_data_size"  :  0.3,
            "preprocessing"   : "Yeo + NZV + Scale + One_Hot",
            "feature_eng"     : "Full"
        }
        mlflow.log_params(exp_params)
        
        model_params = {
            "max_depth"             : 29,
            "max_features"          : 0.6972,
            "n_estimators"          : 809,
            "min_impurity_decrease" : 0.0003,
            "min_samples_leaf"      : 2,
            "min_samples_split"     : 3 
        }
        mlflow.log_params(model_params)
