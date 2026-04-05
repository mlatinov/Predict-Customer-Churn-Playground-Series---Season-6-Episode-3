
from sklearn.model_selection import RandomizedSearchCV

def tunner_random(pipeline, param_grid, X_transformed, y_train_encoded, iters = 50) :
    
    # Random Search
    search = RandomizedSearchCV(
        estimator           =  pipeline,
        param_distributions =  param_grid,
        n_iter              =  iters,
        scoring             = "roc_auc",
        n_jobs              = -1,
        random_state        = 42,
        cv                  = 5
    ) 
    # Fit the Random Search on the Transformed Data 
    search.fit(X_transformed, y_train_encoded)
    return search

