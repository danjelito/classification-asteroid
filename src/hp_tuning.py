import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import gp_minimize
from skopt import space
from functools import partial

import module
import config
import model_dispatcher


def run_cv(
    param_spaces,
    param_names,  
    model, 
    X, 
    y
):

    params= dict(zip(param_names, param_spaces))

    fold= StratifiedKFold(
        n_splits= 10, 
        shuffle= True, 
        random_state= config.RANDOM_STATE
    )

    # create predition pipeline
    prediction= Pipeline([
        ('model', model.set_params(**params))
    ])

    # combine preprocessing, f_selection, compression and prediction pipeline
    pipeline= Pipeline([
        ('preprocessing', module.preprocessing),
        ('f_seelction', module.f_selection),
        ('compression', module.compression),
        ('prediction', prediction)
    ])

    scores= cross_val_score(
        estimator = pipeline, 
        cv= fold, 
        scoring= 'f1',
        X= X, 
        y= y, 
        n_jobs= -1, 
        verbose= 0
    )

    return np.mean(scores) * -1

def optimize(
    param_names, 
    param_spaces,
    model, 
    X, 
    y,
    n_calls
):
    partial_cv= partial(
        run_cv, 
        param_names= param_names, 
        model= model, 
        X= X, 
        y= y, 
    )

    result= gp_minimize(
        partial_cv, 
        dimensions= param_spaces, 
        n_calls= n_calls,
        verbose= 1
    )

    # create best params dict and print it
    best_params = dict(zip(
        param_names,
        result.x
    ))
    print(best_params)

if __name__ == "__main__":

    df_train= pd.read_csv(config.SMOTE_TRAIN_SET)
    X= df_train.drop(columns= 'hazardous')
    y= df_train.loc[:, 'hazardous']

    param_spaces = {
        'rf' : [
            space.Integer(3, 20, name="max_depth"),
            space.Integer(100, 1500, name="n_estimators"),
            space.Categorical(["gini", "entropy"], name="criterion"),
            space.Real(0.01, 1, prior="uniform", name="max_features")
        ], 
        'knn' : [
            space.Integer(3, 20, name="n_neighbors"),
            space.Categorical(["uniform", "distance"], name="weights"),
            space.Categorical(["auto", "ball_tree", 'kd_tree', 'brute'], name="algorithm"),
            space.Integer(10, 100, name="leaf_size"),
        ], 
    }
    param_names = {
        'rf' : [
            "max_depth",
            "n_estimators",
            "criterion",
            "max_features"
        ],
        'knn' : [
            "n_neighbors",
            "weights",
            "algorithm",
            "leaf_size"
        ]
    }

    model= 'knn'

    optimize(
        param_names= param_names[model], 
        param_spaces= param_spaces[model],
        model= model_dispatcher.models[model], 
        X= X, 
        y= y,
        n_calls= 10
    )