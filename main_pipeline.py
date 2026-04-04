
#### Main Pipeline to run all experiments ######
import mlflow
from experiments.exp_logistic_reg import exp_logistic_reg_features

if __name__ == "__main__":
    # Mlflow Init 
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("Kaggle : Churn Prediction")

    # Run Logistic Regression Experiment with Diffrent Feature Engineerings 
    exp_logistic_reg_features(
        RUN_Logistic_regression_baseline = True,
        RUN_DALEX_GLOBAL_EXPLANATIONS = True,
        RUN_DALEX_LOCAL_EXPLANATIONS = True,
        LOG_MODEL = False
    )


