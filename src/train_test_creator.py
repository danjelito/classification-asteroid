import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import module
import config

def create_train_test(
    oversampling= False,
    train_path= config.TRAIN_SET, 
):


    df= pd.read_csv(config.OG_SET, usecols= module.USED_COLS)

    # map label column to 0 and 1
    # clean column name
    df_clean = (df
        .assign(Hazardous = df['Hazardous'].map({True : 1, False : 0}))
        .rename(columns= lambda c: module.clean_col(c))            
    )
    
    X= df_clean.drop(columns = 'hazardous')
    y= df_clean.loc[:, 'hazardous']

    # train test split
    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size= 0.2, stratify= y, shuffle= True, random_state= config.RANDOM_STATE
    )

    if oversampling:
        train_path= config.SMOTE_TRAIN_SET
        sm = SMOTE(random_state= config.RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # concat X and y again
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    if not oversampling:
        assert (train_set.shape[0] + test_set.shape[0]) == df_clean.shape[0]
    assert train_set.shape[1] == df_clean.shape[1]
    assert test_set.shape[1] == df_clean.shape[1]

    # save df
    train_set.to_csv(train_path, index= False)
    test_set.to_csv(config.TEST_SET, index= False)

    print('Finish.')

if __name__ == '__main__':
    create_train_test(
        oversampling= True
    )