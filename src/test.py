import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

import module
import config
import model_dispatcher


if __name__ == "__main__":

    df_train= pd.read_csv(config.SMOTE_TRAIN_SET)
    X_train= df_train.drop(columns= 'hazardous')
    y_train= df_train.loc[:, 'hazardous'].values.ravel()

    df_test= pd.read_csv(config.TEST_SET)
    X_test= df_test.drop(columns= 'hazardous')
    y_test= df_test.loc[:, 'hazardous'].values.ravel()


    models = [
        'knn_tuned', 
        'lgb_tuned', 
        'dt_tuned',
        'svc_tuned',
    ]

    for model in models:

        # create predition pipeline
        prediction= Pipeline([
            ('model', model_dispatcher.models[model])
        ])

        # combine preprocessing, f_selection, compression and prediction pipeline
        pipeline= Pipeline([
            ('preprocessing', module.preprocessing),
            ('f_selection', module.f_selection),
            ('compression', module.compression),
            ('prediction', prediction)
        ])

        pipeline.fit(X_train, y_train)
        y_pred= pipeline.predict(X_test)
        y_pred_proba= pipeline.predict_proba(X_test)

        f1= f1_score(y_test, y_pred)
        accuracy= accuracy_score(y_test, y_pred)
        roc_auc= roc_auc_score(y_test, y_pred_proba[:, 1])

        print(f'''
            Model       : {model}
            Accuracy    : {accuracy}
            F1          : {f1}
            ROC AUC     : {roc_auc}
            \n
        ''')

        model_path= config.MODEL / f'{model}.pkl'
        joblib.dump(pipeline, model_path)