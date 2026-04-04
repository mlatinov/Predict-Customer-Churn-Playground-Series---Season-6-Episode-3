
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def model_predict(model, model_data):
    # Train predictions
    y_train_pred = model.predict(model_data["x_train"])
    y_train_pred_prob = model.predict_proba(model_data["x_train"])[:, 1]

    # Test predictions
    y_test_pred = model.predict(model_data["x_test"])
    y_test_pred_prob = model.predict_proba(model_data["x_test"])[:, 1]
    
    predictions  = {
        "y_train_pred"      : y_train_pred,
        "y_train_pred_prob" : y_train_pred_prob,
        "y_test_pred"       : y_test_pred,
        "y_test_pred_prob"  : y_test_pred_prob
    }
    return predictions

def evaluate_model(model, predictions, model_data) :

    # Evaluate the model 
    train_auc = roc_auc_score(model_data["y_train"],predictions["y_train_pred_prob"])
    test_auc  = roc_auc_score(model_data["y_test"], predictions["y_test_pred_prob"])
    train_acc  =  accuracy_score(model_data["y_train"],predictions["y_train_pred"])
    test_acc   =  accuracy_score(model_data["y_test"],predictions["y_test_pred"])

    metrics_dict = {
        "Train AUC" : round(train_auc, 3),
        "Test  AUC" : round(test_auc, 3),
        "AUC Bias"  : round(train_auc - test_auc, 3),

        "Train Accuracy" : round(train_acc,3),
        "Test  Accuracy" : round(test_acc, 3),
        "Accuracy Bias"  : round(train_acc - test_acc, 3)
    }
    return metrics_dict

def confusion_matrix(predictions, model_data) : 
    """
    Function to plot and save a Confusion Matrix Used for Model Logs 
    """
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        model_data["y_test"],
        predictions["y_test_pred"],
        display_labels=["No", "Yes"],
        normalize="true",
        cmap= "Blues",
        ax=ax
    )
    plt.title("Confusion Matrix")
    return fig

