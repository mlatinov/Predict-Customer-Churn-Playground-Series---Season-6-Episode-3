
import mlflow 
from functions.modeling_f.modeling_helpers_f import evaluate_model, confusion_matrix, model_predict
from functions.data_prep_f.features_helpers import build_pipeline
from functions.data_prep_f.data_helpers import data_split
from sklearn.linear_model import LogisticRegression
from functions.modeling_f.dalex_f import (
    dx_create_explainer,
    dx_global_importance,
    dx_local_explanations,
    mlflow_log_dalex_plot,
    dx_residual_analysis
)

def exp_logistic_reg_features(
    RUN_Logistic_regression_baseline = False,
    RUN_Logistic_regression_tenure = False,
    RUN_Logistic_regression_value_x_service = False,
    RUN_Logistic_regression_full_feature = False,
    RUN_DALEX_GLOBAL_EXPLANATIONS = True,
    RUN_DALEX_LOCAL_EXPLANATIONS = True,
    LOG_MODEL = False
    ) :
    """
    Run the logistic regression baseline experiment and log results to MLflow.

    Orchestrates the full experiment lifecycle: data loading, pipeline construction,
    model training, evaluation, DALEX explainability, and MLflow logging. Each
    enabled stage is recorded as a separate MLflow run under the 'Logistic_regression
    baseline' run name.
    """
    match True:
        case _ if RUN_Logistic_regression_baseline:
            model_to_run = "Logistic Regressioon Baseline"
            feature_eng = "None"
            features_names = ["tenure","MonthlyCharges","TotalCharges"]
        case _ if RUN_Logistic_regression_tenure:
            model_to_run = "Logistic Regression Tenure"
            feature_eng = "Tenure"
            features_names = ["tenure","MonthlyCharges","TotalCharges"]
        case _ if RUN_Logistic_regression_value_x_service:
            model_to_run = "Logistic Regreesion Value & Service"
            feature_eng = "Value & Service"
            features_names = ["tenure","MonthlyCharges","TotalCharges","value_gap"]
        case _ if RUN_Logistic_regression_full_feature:
            model_to_run = "Logistic Regression Full Eng"
            feature_eng = "Full"
            features_names = ["tenure","MonthlyCharges","TotalCharges","value_gap"]
        case _:
            model_to_run = None
            
    # =============== Logictic Regression Features Experiment =====================================
    with mlflow.start_run(run_name = model_to_run) :

        # Experimental Settings 
        params = {
            "model"           : "Logistic Regreesion",
            "model_type"      : "Linear Models",
            "model_params"    : "Defaults",
            "train_data_size" :  0.7,
            "test_data_size"  :  0.3,
            "preprocessing"   : "Scale + One_Hot",
            "feature_eng"     : feature_eng  
        }
        mlflow.log_params(params)

        # Split the data 
        model_data = data_split(
                train_url= "sample_data/train.csv",
                train_size= 0.7,
                test_size = 0.3
        )

        # Build a model pipeline only with the original features 
        pipeline = build_pipeline(
            model= LogisticRegression(class_weight="balanced"),
            add_tenure= RUN_Logistic_regression_tenure,
            add_value_x_service= RUN_Logistic_regression_value_x_service,
            add_full_feature= RUN_Logistic_regression_full_feature
        ) 
        # Fit the pipeline on the training data 
        pipeline.fit(
            X = model_data["x_train"],
            y = model_data["y_train"]
        )

        # Predict with the model 
        model_predictions = model_predict(pipeline, model_data)

        # Evaluate the model and plot Confusion Matrix 
        eval = evaluate_model(
            model= pipeline,
            predictions=model_predictions,
            model_data= model_data
        )
        conf_matrix = confusion_matrix(model_predictions, model_data)

        # Log Model Evaluations 
        mlflow.log_figure(conf_matrix, "plots/confusion_matrix.png")
        mlflow.log_metrics(eval)

        # Create a DALEX Explainer if Global or Local Explanations = True
        if RUN_DALEX_GLOBAL_EXPLANATIONS or RUN_DALEX_LOCAL_EXPLANATIONS :
            dalex_exp = dx_create_explainer(
                pipeline= pipeline,
                x_train= model_data["x_train"],
                y_train= model_data["y_train"],
                label= model_to_run
            )
            # Residual Analysis 
            residual_analysis = dx_residual_analysis(dx_explainer = dalex_exp)
            mlflow.log_figure(residual_analysis["fig1"], "residual_distribution.png")

        # ======== Global Explations =============
        if RUN_DALEX_GLOBAL_EXPLANATIONS :
            # Run Dalex Global Explanations 
            gfi = dx_global_importance(
                dalex_explainer = dalex_exp,
                features_names = features_names
            )
            # Log Global Explanations Plots
            mlflow_log_dalex_plot(gfi["Loss_Shuffle_Plot"],"variable_importance.html",  "dalex/global")
            mlflow_log_dalex_plot(gfi["PDP_Plot"]         ,"partial_dependence.html",   "dalex/global")
            mlflow_log_dalex_plot(gfi["RCDR_Plot"]        ,"residuals_rcdf.html",       "dalex/global")

        # ======== Local Explations =============
        if RUN_DALEX_LOCAL_EXPLANATIONS :
            # Run Dalex Local Explanations  
            lme = dx_local_explanations(
                pipeline = pipeline,
                dalex_explainer = dalex_exp,
                x_train = model_data["x_train"],
                y_train = model_data["y_train"],
                features_names = features_names
            )
            # Log Local Explanations Plots 
            mlflow_log_dalex_plot(lme["bd_plot_1"]  ,"breakdown_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["bd_plot_2"]  ,"breakdown_no_churn.html",   "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_1"]  ,"ceretis_paribus_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_2"]  ,"ceretis_paribus_no_churn.html",   "dalex/local")

        # ============= Logging the model ====================
        if LOG_MODEL :
            mlflow.sklearn.log_model(
                sk_model = pipeline,
                artifact_path = "model",
                input_sample = model_data["x_train"].head(5),
                registered_model_name = model_to_run
            )

def exp_logistic_reg_preproc(
    RUN_Logistic_regression_full_feature = False,
    RUN_Logistic_regression_baseline = False,
    RUN_DALEX_GLOBAL_EXPLANATIONS = True,
    RUN_DALEX_LOCAL_EXPLANATIONS = True,
    LOG_MODEL = False
    ) : 

    match True :
        case _ if RUN_Logistic_regression_full_feature :
            model_to_run = "Logistic Regression Full Proc"
            feature_eng = "Full"
            preproc = "NZV + Yeo + Scale + One-Hot"
            features_names = ["tenure","MonthlyCharges","TotalCharges","value_gap"]
        case _ if RUN_Logistic_regression_baseline : 
            model_to_run = "Logistic Regression Baseline Proc"
            feature_eng = "None"
            preproc = "NZV + Yeo + Scale + One-Hot"
            features_names = ["tenure","MonthlyCharges","TotalCharges"]
        case _ :
            model_to_run = None
    
    with mlflow.start_run(run_name= model_to_run) : 

        # Experimental Settings 
        params = {
            "model"           : "Logistic Regreesion",
            "model_type"      : "Linear Models",
            "model_params"    : "Defaults",
            "train_data_size" :  0.7,
            "test_data_size"  :  0.3,
            "preprocessing"   : preproc,
            "feature_eng"     : feature_eng  
        }
        mlflow.log_params(params)

        # Split the data 
        model_data = data_split(
            train_url= "sample_data/train.csv",
            train_size= 0.7,
            test_size= 0.3
        )
        # Build a model Pipeline 
        pipeline = build_pipeline(
            model = LogisticRegression(),
            add_remove_nzv= True,
            add_power_transform = True,
            add_tenure= False,
            add_value_x_service=False,
            add_full_feature= RUN_Logistic_regression_full_feature,
        )
        # Fit the pipeline on the training data 
        pipeline.fit(
            X = model_data["x_train"],
            y = model_data["y_train"]
        )
        # Predict with the model 
        model_predictions = model_predict(pipeline, model_data)

        # Evaluate the model and plot Confusion Matrix 
        eval = evaluate_model(
            model= pipeline,
            predictions=model_predictions,
            model_data= model_data
        )
        conf_matrix = confusion_matrix(model_predictions, model_data)

        # Log Model Evaluations 
        mlflow.log_figure(conf_matrix, "plots/confusion_matrix.png")
        mlflow.log_metrics(eval)

        # Create a DALEX Explainer if Global or Local Explanations = True
        if RUN_DALEX_GLOBAL_EXPLANATIONS or RUN_DALEX_LOCAL_EXPLANATIONS :
            dalex_exp = dx_create_explainer(
                pipeline= pipeline,
                x_train= model_data["x_train"],
                y_train= model_data["y_train"],
                label= model_to_run
            )

        # ======== Global Explations =============
        if RUN_DALEX_GLOBAL_EXPLANATIONS :
            # Run Dalex Global Explanations 
            gfi = dx_global_importance(
                dalex_explainer = dalex_exp,
                features_names = features_names
            )
            # Log Global Explanations Plots
            mlflow_log_dalex_plot(gfi["Loss_Shuffle_Plot"],"variable_importance.html",  "dalex/global")
            mlflow_log_dalex_plot(gfi["PDP_Plot"]         ,"partial_dependence.html",   "dalex/global")
            mlflow_log_dalex_plot(gfi["RCDR_Plot"]        ,"residuals_rcdf.html",       "dalex/global")

        # ======== Local Explations =============
        if RUN_DALEX_LOCAL_EXPLANATIONS :
            # Run Dalex Local Explanations  
            lme = dx_local_explanations(
                pipeline = pipeline,
                dalex_explainer = dalex_exp,
                x_train = model_data["x_train"],
                y_train = model_data["y_train"],
                features_names = features_names
            )
            # Log Local Explanations Plots 
            mlflow_log_dalex_plot(lme["bd_plot_1"]  ,"breakdown_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["bd_plot_2"]  ,"breakdown_no_churn.html",   "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_1"]  ,"ceretis_paribus_churn.html",      "dalex/local")
            mlflow_log_dalex_plot(lme["cp_plot_2"]  ,"ceretis_paribus_no_churn.html",   "dalex/local")

        # ============= Logging the model ====================
        if LOG_MODEL :
            mlflow.sklearn.log_model(
                sk_model = pipeline,
                name                  = "Logistic Regression",   
                input_example         = model_data["x_train"].head(5),  
                registered_model_name = "Logistic Regression"
            )
