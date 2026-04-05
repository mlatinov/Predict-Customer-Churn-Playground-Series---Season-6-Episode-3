
#### Main Pipeline to run all experiments ######
import mlflow
from experiments.exp_logistic_reg import exp_logistic_reg_features
from  experiments.exp_logistic_reg import exp_logistic_reg_preproc

if __name__ == "__main__":
    # Mlflow Init 
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("Kaggle : Churn Prediction")

    # Run Logistic Regression Experiment with Diffrent Feature Engineerings 
    # exp_logistic_reg_features()

    # Run Logistic Regression Experiment with Diffrent Preprocessings 
    exp_logistic_reg_preproc(
        RUN_Logistic_regression_full_feature= True,
        RUN_Logistic_regression_baseline= False,
        RUN_DALEX_LOCAL_EXPLANATIONS=True,
        RUN_DALEX_GLOBAL_EXPLANATIONS=True
    )



