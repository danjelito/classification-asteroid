import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline

import module
import config
import model_dispatcher


if __name__ == "__main__":

    df_train= pd.read_csv(config.TRAIN_SET)
    X= df_train.drop(columns= 'hazardous')
    y= df_train.loc[:, 'hazardous']
    
    all_model_scores= []

    for model in model_dispatcher.models.keys():

        fold= StratifiedKFold(
            n_splits= 10, 
            shuffle= True, 
            random_state= config.RANDOM_STATE
        )

        # create predition pipeline
        prediction= Pipeline([
            ('model', model_dispatcher.models[model])
        ])

        # combine preprocessing, f_selection, compression and prediction pipeline
        pipeline= Pipeline([
            ('preprocessing', module.preprocessing),
            ('f_seelction', module.f_selection),
            ('compression', module.compression),
            ('prediction', prediction)
        ])

        model_scores= cross_validate(
            estimator = pipeline, 
            cv= fold, 
            scoring= 'f1',
            X= X, 
            y= y, 
            n_jobs= -1, 
            return_train_score= True, 
            verbose= 0
        )

        # create a df with scores
        # add model name and fold as new columns
        model_scores= pd.DataFrame(model_scores).assign(
            model= model, 
            fold= list(range(10)),
        )

        all_model_scores.append(model_scores)

    # concat all dfs in results
    # group by model and get the average
    all_model_scores = pd.concat(all_model_scores)
    all_model_scores= (all_model_scores
        .drop(columns= 'fold')
        .groupby('model')
        .mean()
        .sort_values('test_score', ascending= False)
    )
    print(all_model_scores)