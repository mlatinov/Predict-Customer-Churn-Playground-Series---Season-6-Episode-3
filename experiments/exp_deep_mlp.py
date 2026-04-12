
import mlflow 
from functions.modeling_f.nn_helper import build_deep_mlp
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from functions.modeling_f.modeling_helpers_f import evaluate_model, confusion_matrix, model_predict
from functions.modeling_f.tunner_f import deep_mlp_tune_optuna , opt_tune_viz , mlflow_log_optuna_plot
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

def exp_deep_mlp(
    RUN_tune = False,
    RUN_DALEX_GLOBAL_EXPLANATIONS = True,
    RUN_DALEX_LOCAL_EXPLANATIONS = True,
    RUN_LOG_MODEL = False
    ) :

    with mlflow.start_run(run_name= "Deep MLP") :

        # Split the data 
        model_data = data_split(
            train_url  = "sample_data/train.csv",
            train_size = 0.7,
            test_size  = 0.3
        )
        # Wrap the Deep MLP model inside Keras Classifier  
        deep_mlp_model = KerasClassifier(
            model = build_deep_mlp,
            hidden_units = [192, 128, 192, 224],
            dropout_rate = 0.15,
            activation = "elu",
            learning_rate = 0.0001,
            epochs = 100,
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor  = "val_loss",
                patience = 5,
                restore_best_weights = True,
            )],
            validation_split = 0.2
        )
        # Create the model pipeline 
        pipeline = build_pipeline(
            model               = deep_mlp_model,
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

        # Predict and Evalute the model 
        model_predictions = model_predict(model = pipeline, model_data = model_data)
        eval = evaluate_model(
            model = pipeline,
            predictions = model_predictions,
            model_data = model_data
        )
        # Log the Evaluation 
        conf_matrix = confusion_matrix(predictions = model_predictions, model_data = model_data) 
        mlflow.log_metrics(eval)
        mlflow.log_figure(conf_matrix, "plots/confusion_matrix.png")

        # ============== Tune Deep MLP model with Optuna ==================
        if RUN_tune :
             # Prepare the data 
            tunner_df = dx_transform(
                pipeline = pipeline,
                x_train  = model_data["x_train"],
                y_train  = model_data["y_train"]
            )
            # Force numpy int array — scikeras reliably detects classification from this
            y_encoded = tunner_df["y_encoded"].to_numpy().astype(int)

            # Run the optimization 
            study = deep_mlp_tune_optuna(
                X_train  = model_data["x_train"],
                y_train  = y_encoded,
                n_trials = 50
            )
            # Log the best parameters
            best_params = study.best_params
            best_tune_values = study.best_value 
    
            # Viz the Tuning process
            opt_viz = opt_tune_viz(study, hyper_params=["learning_rate","n_layers"])

            # Log the Results
            mlflow.log_params(best_params)
            mlflow.log_metric(best_tune_values)
            mlflow_log_optuna_plot(opt_viz["cordinate_plot"]      , "parallel_cordinate_plot.html","optuna")
            mlflow_log_optuna_plot(opt_viz["countour_plot"]       , "countour_plot.html"          ,"optuna")
            mlflow_log_optuna_plot(opt_viz["parameter_importance"], "hyper_params_importance.html","optuna")

        # If Dalex create a explainer 
        if RUN_DALEX_GLOBAL_EXPLANATIONS or RUN_DALEX_LOCAL_EXPLANATIONS :
            dx_explainer = dx_create_explainer(
                pipeline = pipeline,
                x_train  = model_data["x_train"],
                y_train  = model_data["y_train"],
                label    = "Tuned Deep MLP"
            )
            feature_names = ["tenure","MonthlyCharges","TotalCharges","value_gap"]

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
                features_names  = feature_names,
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
                name                  = "Deep MLP",   
                input_example         = model_data["x_train"].head(5),  
                registered_model_name = "Deep MLP"
        )

        # ========= Experimental Settings ================
        exp_params = {
            "model"           : "Tuned Deep MLP",
            "model_type"      : "Neural Network",
            "model_params"    : "Optuna Tuned",
            "train_data_size" :  0.7,
            "test_data_size"  :  0.3,
            "preprocessing"   : "Yeo + NZV + Scale + One_Hot",
            "feature_eng"     : "Full"
        }
        mlflow.log_params(exp_params)
        model_params = {
            "learning_rate"  : 0.0001,
            "n_layers"       : 4,
            "units_layer_0"  : 192,
            "units_layer_1"  : 128,
            "units_layer_2"  : 192,
            "units_layer_3"  : 224,
            "dropout_rate"   : 0.15,
            "activation"     : "elu"
        }
        mlflow.log_params(model_params)

