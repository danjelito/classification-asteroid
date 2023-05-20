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
    param_spaces: list,
    param_names: list,  
    X: pd.DataFrame, 
    y: pd.DataFrame,
    model
):
    '''Run cross valdiation to be passed to optimization function.

    Args:
        param_spaces (list): parameter spaces as described by skopt.space
        param_names (list): names of parameters inside param_spaces
        X (pd.DataFrame): X
        y (pd.DataFrame): y
        model (_type_): sklearn classifier object

    Returns:
        float: negative mean f1 scores across all folds
    '''

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
    param_spaces: list,
    param_names: list,
    model, 
    X: pd.DataFrame, 
    y: pd.DataFrame,
    n_calls: int
):
    '''Run gp_minimize to search for opimum HP.

    Args:
        param_spaces (list): parameter spaces as specified by skopt.space
        param_names (list): names of parameters inside param_spaces
        model (_type_): sklearn classifier object
        X (pd.DataFrame): X
        y (pd.DataFrame): y
        n_calls (int): number of iterations
    '''

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

if __name__ == '__main__':

    # load train set
    # spcify X and y
    df_train= pd.read_csv(config.SMOTE_TRAIN_SET)
    X= df_train.drop(columns= 'hazardous')
    y= df_train.loc[:, 'hazardous']

    # specify param_spaces and names for all models that will be tested
    param_spaces = {
        'lgb' : [
            space.Integer(3, 150, name='num_leaves'),
            space.Integer(3, 300, name='max_depth'),
            space.Integer(100, 400, name='max_bin'),
            space.Real(0.00001, 1, prior='log-uniform', name= 'learning_rate'),
        ], 
        'knn' : [
            space.Integer(3, 20, name='n_neighbors'),
            space.Categorical(['uniform', 'distance'], name='weights'),
            space.Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
            space.Integer(10, 100, name='leaf_size'),
        ], 
        'dt': [
            space.Categorical(['gini', 'entropy', 'log_loss'], name= 'criterion'), 
            space.Categorical(['best', 'random'], name= 'splitter'), 
            space.Integer(1, 300, name='max_depth'),
            space.Integer(1, 100, name='min_samples_leaf'),
            space.Categorical(['sqrt', 'log2'], name= 'max_features'), 
        ],
        'svc': [
            space.Real(0.00001, 100, prior='log-uniform', name= 'C'),
            space.Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name= 'kernel'), 
            space.Categorical(['scale', 'auto'], name= 'gamma'), 
        ],
    }
    param_names = {
        'lgb' : [
            'num_leaves',
            'max_depth',
            'max_bin',
            'learning_rate',
        ],
        'knn' : [
            'n_neighbors',
            'weights',
            'algorithm',
            'leaf_size'
        ],
        'dt' : [
            'criterion',
            'splitter',
            'max_depth',
            'min_samples_leaf',
            'max_features',
        ],
        'svc' : [
            'C',
            'kernel',
            'gamma',
        ],
    }

    # run optimize
    model= 'svc'
    optimize(
        param_names= param_names[model], 
        param_spaces= param_spaces[model],
        model= model_dispatcher.models[model], 
        X= X, 
        y= y,
        n_calls= 100
    )