
#### Main Pipeline to run all experiments ######
import mlflow
import pandas as pd 
from experiments.exp_stack import exp_stack

if __name__ == "__main__":
    # Mlflow Init 
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("Kaggle : Churn Prediction")

    # Run the Final Stack Model 
    exp_stack()

    # Get the Stack model 
    stack_model = mlflow.sklearn.load_model("models:/Stacked Model/1")
    rf_pipeline   = mlflow.sklearn.load_model("models:/tuned_random_forest/1")
    hist_pipeline = mlflow.sklearn.load_model("models:/Hist Boost/2")
    nn_pipeline   = mlflow.sklearn.load_model("models:/Deep MLP/1")
    ada_pipeline  = mlflow.sklearn.load_model("models:/Ada Boost/2")
    
    # Predict on the test data 
    test_data = pd.read_csv("sample_data/test.csv")
    model_predictions_stack = stack_model.predict_proba(test_data)[:, 1]
    model_predictions_rf     = rf_pipeline.predict_proba(test_data)[:, 1]
    model_predictions_hist   = hist_pipeline.predict_proba(test_data)[:, 1]
    model_predictions_nn     = nn_pipeline.predict_proba(test_data)[:, 1]
    model_predictions_ada    = ada_pipeline.predict_proba(test_data)[:, 1]

    # Convert to submission
    submission_stack = pd.DataFrame({
        "id": test_data["id"],
        "Churn": model_predictions_stack
    })

    submission_rf = pd.DataFrame({
        "id": test_data["id"],
        "Churn": model_predictions_rf
    })

    submission_hist = pd.DataFrame({
        "id": test_data["id"],
        "Churn": model_predictions_hist
    })
    
    submission_nn = pd.DataFrame({
        "id": test_data["id"],
        "Churn": model_predictions_nn
    })

    submission_ada = pd.DataFrame({
        "id": test_data["id"],
        "Churn": model_predictions_ada
    })

    # Save it 
    submission_stack.to_csv("stack_submission.csv", index=False)
    submission_rf.to_csv("rf_submission.csv", index = False) 
    submission_hist.to_csv("hist_submission.csv", index = False) 
    submission_nn.to_csv("nn_submission.csv", index = False) 
    submission_ada.to_csv("ada_submission.csv", index = False) 