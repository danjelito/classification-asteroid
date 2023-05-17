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
    all_cv_scores= []

    for model in model_dispatcher.models.keys():

        cv= StratifiedKFold(n_splits= 10, shuffle= True, random_state= config.RANDOM_STATE)
        # model= model_dispatcher.models[model]

        # create predition pipeline
        prediction= Pipeline([
            ('model', model_dispatcher.models[model])
        ])

        # combine preprocessing and prediction pipeline
        pipeline= Pipeline([
            ('preprocessing', module.preprocessing),
            ('prediction', prediction)
        ])

        cv_scores= cross_validate(
            estimator = pipeline, 
            cv= cv, 
            scoring= 'f1',
            X= X, 
            y= y, 
            n_jobs= -1, 
            return_train_score= True, 
            verbose= 0
        )

        # create a df with scores
        # add model name and fold as new columns
        cv_scores_df= pd.DataFrame(cv_scores).assign(
            model= model, 
            fold= list(range(10)),
        )

        all_cv_scores.append(cv_scores_df)

    # concat all dfs in results
    all_cv_scores = pd.concat(all_cv_scores)
    print(all_cv_scores)