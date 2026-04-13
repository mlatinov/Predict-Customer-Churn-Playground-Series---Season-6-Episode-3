
import mlflow
from functions.data_prep_f.data_helpers import data_split
from functions.modeling_f.modeling_helpers_f import evaluate_model, confusion_matrix, model_predict
from sklearn.ensemble import StackingClassifier
import lightgbm as lgb

def exp_stack():

    with mlflow.start_run(run_name = "Stack") :
        
        # Load the data 
        model_data = data_split(
            train_url  = "sample_data/train.csv",
            train_size = 0.7,
            test_size  = 0.3
        )
        # Load all models for the stack 
        mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
        rf_pipeline   = mlflow.sklearn.load_model("models:/tuned_random_forest/1")
        hist_pipeline = mlflow.sklearn.load_model("models:/Hist Boost/2")
        nn_pipeline   = mlflow.sklearn.load_model("models:/Deep MLP/1")
        ada_pipeline  = mlflow.sklearn.load_model("models:/Ada Boost/2")

        # Create the stack with LightGBM As Meta learner 
        stack_model = StackingClassifier(
            estimators = [
                ("Random Forest" , rf_pipeline  ),
                ("Hist Boost"    , hist_pipeline),
                ("Deep MLP"      , nn_pipeline  ),
                ("Ada Boosts"    , ada_pipeline )
            ],
            final_estimator = lgb.LGBMClassifier(
                n_estimators      = 100,
                learning_rate     = 0.05,
                num_leaves        = 8,       
                min_child_samples = 20,      
                random_state      = 42,
                verbose           = -1
            ),
            cv           = 5,
            stack_method = "predict_proba",
            passthrough  = False,
            n_jobs       = 1
        )
        # Train the Stack Model 
        stack_model.fit(X = model_data["x_train"], y = model_data["y_train"])

        # Predict with the stack model 
        model_predictions = model_predict(model = stack_model, model_data = model_data)

        # Evaluate the model 
        eval = evaluate_model(
            model       = stack_model,
            predictions = model_predictions,
            model_data  = model_data
        )
        conf_matrix = confusion_matrix(predictions = model_predictions, model_data = model_data)
        mlflow.log_metrics(eval)
        mlflow.log_figure(conf_matrix, "plots/confusion_matrix.png")
        
        # Log the final Model 
        mlflow.sklearn.log_model(
            sk_model = stack_model,
            name                  = "Stacked Model",   
            input_example         = model_data["x_train"].head(5),  
            registered_model_name = "Stacked Model"
            )

        # ============ Experimental Settings and Loging  ===============
        exp_params = {
            "model"           : "Stacked Model",
            "model_type"      : "Stacked",
            "model_params"    : "Not Tuned",
            "train_data_size" :  0.7,
            "test_data_size"  :  0.3,
            "preprocessing"   : "Yeo + NZV + Scale + One_Hot",
            "feature_eng"     : "Full"
        }
        mlflow.log_params(exp_params)
        model_params = {
            "Estimator 1"      : "Random Forest",
            "Estimator 2"      : "Hist Boost",
            "Estimator 3"      : "Deep MLP",
            'Estimator 4'      : "Ada Boost",
            "Meta Estimator"   : "LightGBM"
        }
        mlflow.log_params(model_params)
