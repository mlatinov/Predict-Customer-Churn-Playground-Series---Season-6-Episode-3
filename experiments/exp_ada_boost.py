
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
    dx_transform,
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
                estimator= DecisionTreeClassifier(max_depth=1),
                learning_rate = 0.2,
                n_estimators  = 150,
                random_state  = 42
            ),
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
            # Refit the pipeline with the best parameters 
            pipeline = search.best_estimator_
        # If DALEX create a dalex explainer 
        if RUN_DALEX_GLOBAL_EXPLANATIONS or RUN_DALEX_LOCAL_EXPLANATIONS :
            # Create Dalex Explaier 
            dx_explainer = dx_create_explainer(
                pipeline = pipeline,
                x_train  = model_data["x_train"],
                y_train  = model_data["y_train"],
                label    = "Tuned Ada Boosts with DT weak Estenator"
            )
            feature_names = ["tenure","MonthlyCharges","TotalCharges","value_gap"]

        # =========== DALEX Global Explanations ========================== 
        if RUN_DALEX_GLOBAL_EXPLANATIONS : 
            gfi = dx_global_importance(
                dalex_explainer = dx_explainer,
                features_names = feature_names
            )
            # Log Global Explanations Plots
            mlflow_log_dalex_plot(gfi["Loss_Shuffle_Plot"],"variable_importance.html",  "dalex/global")
            mlflow_log_dalex_plot(gfi["PDP_Plot"]         ,"partial_dependence.html",   "dalex/global")
            mlflow_log_dalex_plot(gfi["RCDR_Plot"]        ,"residuals_rcdf.html",       "dalex/global")
        
        # ============ DALEX Local Explanations ========================
        if RUN_DALEX_LOCAL_EXPLANATIONS : 
            lme = dx_local_explanations(
                dalex_explainer = dx_explainer,
                pipeline        = pipeline,
                x_train         = model_data["x_train"],
                y_train         = model_data["y_train"],
                features_names  = feature_names
            )
            # Log Local Explanations Plots 
            mlflow_log_dalex_plot(lme["bd_plot_1"]  ,"breakdown_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["bd_plot_2"]  ,"breakdown_no_churn.html",   "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_1"]  ,"ceretis_paribus_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_2"]  ,"ceretis_paribus_no_churn.html",   "dalex/local")
        
        # Log the model 
        if LOG_MODEL : 
            mlflow.sklearn.log_model(
                sk_model = pipeline,
                name                  = "Ada Boost",   
                input_example         = model_data["x_train"].head(5),  
                registered_model_name = "Ada Boost"
            )

        # ============ Experimental Settings and Loging  ===============
        exp_params = {
            "model"           : "Ada Boost",
            "model_type"      : "Boosted Trees",
            "model_params"    : "Tuned",
            "train_data_size" :  0.7,
            "test_data_size"  :  0.3,
            "preprocessing"   : "Yeo + NZV + Scale + One_Hot",
            "feature_eng"     : "Full"
        }
        mlflow.log_params(exp_params)
        model_params = {
            "max_depth"             : 1,
            "learning_rate"         : 0.2,
            "n_estimators"          : 150 
        }
        mlflow.log_params(model_params)
