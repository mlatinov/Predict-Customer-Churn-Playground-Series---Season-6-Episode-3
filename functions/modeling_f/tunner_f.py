
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV ,HalvingRandomSearchCV

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

def tunner_successive_halving(pipeline, param_grid, X_transformed, y_train_encoded,) :

    #  Successive halving
    search = HalvingRandomSearchCV(
        estimator           = pipeline,
        param_distributions = param_grid,
        scoring             = "roc_auc",
        min_resources       = "smallest", 
        factor              = 3,
        n_candidates        = 100, 
        n_jobs              = 2, 
        random_state        = 42,
        verbose             = 1,
        cv                  = 5 
    )
    # Fit the Successive halving on the Transformed Data 
    search.fit(X_transformed, y_train_encoded)
    return search
