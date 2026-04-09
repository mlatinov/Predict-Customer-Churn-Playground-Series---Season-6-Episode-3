
import mlflow 
from sklearn.ensemble import HistGradientBoostingClassifier
from functions.modeling_f.modeling_helpers_f import evaluate_model, confusion_matrix, model_predict
from functions.modeling_f.tunner_f import opt_tunner , opt_tune_viz , mlflow_log_optuna_plot
from functions.data_prep_f.features_helpers import build_pipeline
from functions.data_prep_f.data_helpers import data_split
from functions.modeling_f.dalex_f import (
    dx_create_explainer,
    dx_global_importance,
    dx_local_explanations,
    mlflow_log_dalex_plot,
    dx_transform,
    dx_residual_analysis
)

def exp_hist_gradient_tune(
    RUN_hist_gradient_boost = False,
    RUN_DALEX_GLOBAL_EXPLANATIONS = True,
    RUN_DALEX_LOCAL_EXPLANATIONS = True,
    RUN_LOG_MODEL = False
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
            model = HistGradientBoostingClassifier(
                learning_rate    = 0.0267,
                max_iter         = 442,
                max_leaf_nodes   = 65,
                min_samples_leaf = 18,
                class_weight = None
            ),
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

            # Prepare the data 
            tunner_df = dx_transform(
                pipeline = pipeline,
                x_train  = model_data["x_train"],
                y_train  = model_data["y_train"]
            )
            # Run Optuna 
            search = opt_tunner(
                model_name = "HistGradientBoostingClassifier",
                x_train  = tunner_df["dalex_df"],
                y_train  = tunner_df["y_encoded"],
                n_trials = 50
            )
            best_tune_value = search.best_value
            best_parameters = search.best_params

            # Viz the Tuning process
            opt_viz = opt_tune_viz(search, hyper_params=["learning_rate","max_iter"])

            # Log The Tuning process to mlflow 
            mlflow.log_model_params(best_parameters)
            mlflow.log_metric(best_tune_value)
            mlflow_log_optuna_plot(opt_viz["cordinate_plot"]      , "parallel_cordinate_plot.html","optuna")
            mlflow_log_optuna_plot(opt_viz["countour_plot"]       , "countour_plot.html"          ,"optuna")
            mlflow_log_optuna_plot(opt_viz["parameter_importance"], "hyper_params_importance.html","optuna")

        # If Dalex create a explainer 
        if RUN_DALEX_GLOBAL_EXPLANATIONS or RUN_DALEX_LOCAL_EXPLANATIONS :
            dx_explainer = dx_create_explainer(
                pipeline = pipeline,
                x_train  = model_data["x_train"],
                y_train  = model_data["y_train"],
                label    = "Tuned Hist Boost"
            )
            feature_names = ["tenure","MonthlyCharges","TotalCharges","value_gap"]

            # Residual Analysis 
            residual_analysis = dx_residual_analysis(dx_explainer = dx_explainer)
            mlflow.log_figure(residual_analysis["fig1"], "residual_distribution.png")

        # =========== Run Dalex Global Explanations =====================
        if RUN_DALEX_GLOBAL_EXPLANATIONS :
            gfi = dx_global_importance(
                dalex_explainer = dx_explainer,
                features_names = feature_names
            )
             # Log Global Explanations Plots
            mlflow_log_dalex_plot(gfi["Loss_Shuffle_Plot"],"variable_importance.html",  "dalex/global")
            mlflow_log_dalex_plot(gfi["PDP_Plot"]         ,"partial_dependence.html",   "dalex/global")
            mlflow_log_dalex_plot(gfi["RCDR_Plot"]        ,"residuals_rcdf.html",       "dalex/global")

        # ========== Run Dalex Local Explanations 
        if RUN_DALEX_LOCAL_EXPLANATIONS : 
            lme = dx_local_explanations(
                dalex_explainer = dx_explainer,
                pipeline        = pipeline,
                feature_names   = feature_names,
                x_train         = model_data["x_train"],
                y_train         = model_data["y_train"]
            )
             # Log Local Explanations Plots 
            mlflow_log_dalex_plot(lme["bd_plot_1"]  ,"breakdown_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["bd_plot_2"]  ,"breakdown_no_churn.html",   "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_1"]  ,"ceretis_paribus_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_2"]  ,"ceretis_paribus_no_churn.html",   "dalex/local")
        
        # ========= Log the model ========================
        if RUN_LOG_MODEL :
            mlflow.sklearn.log_model(
                sk_model = pipeline,
                artifact_path = "model",
                input_sample = model_data["x_train"].head(5),
                registered_model_name = "Tuned Ada Boost Model"
        )

        # ========= Experimental Settings ================
        exp_params = {
            "model"           : "Hist Boost",
            "model_type"      : "Boosted Trees",
            "model_params"    : "Optuna Tuned",
            "train_data_size" :  0.7,
            "test_data_size"  :  0.3,
            "preprocessing"   : "Yeo + NZV + Scale + One_Hot",
            "feature_eng"     : "Full"
        }
        mlflow.log_params(exp_params)
        model_params = {
            "learning_rate"    : 0.0267,
            "max_iter"         : 442,
            "max_leaf_nodes"   : 65,
            "min_samples_leaf" : 18,
            "class_weight"     : "None"
        }
        mlflow.log_model_params(model_params)

