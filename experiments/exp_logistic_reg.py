
import mlflow 
from functions.modeling_f.modeling_helpers_f import evaluate_model
from functions.data_prep_f.features_helpers import build_pipeline
from functions.data_prep_f.data_helpers import data_split
from sklearn.linear_model import LogisticRegression

def exp_logistic_reg_features() :

    #=============== Logictic Regression =====================================
    with mlflow.start_run(run_name = "Logistic_regression baseline") :
        # Split the data 
        model_data = data_split(
            train_url= "sample_data/train.csv",
            train_size= 0.7,
            test_size = 0.3
        )
        # Build a pipeline only with the original features 
        model = build_pipeline(
            model= LogisticRegression(),
            add_tenure= False,
            add_value_x_service=False,
            add_full_feature=False 
        ) 
        # Fit the model on the training set 
        model.fit(
            X = model_data["x_train"],
            y = model_data["y_train"]
        )
        # Logging Parameters, Metrics, and Tags
        params = {
            "model"           : "Logistic Regreesion",
            "model_type"      : "Linear Models",
            "train_data_size" :  0.7,
            "test_data_size"  :  0.3,
            "preprocessing"   : "Scale + One_Hot",
            "feature_eng"     : "None"  
        }
        mlflow.log_params(params)

        # Evaluate the model 
        eval = evaluate_model(model= model, model_data= model_data)
        mlflow.log_metrics(eval)

        ## Log Artifacts ##
        
        # Confusion Matrix 
        
        # DALEX Reasoning  

        # Log Model 




