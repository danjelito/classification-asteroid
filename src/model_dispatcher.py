from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


lgb_tuned= LGBMClassifier(**{
    'num_leaves': 52,
    'max_depth': 90,
    'max_bin': 331,
    'learning_rate': 0.3040315881101998
})
knn_tuned= KNeighborsClassifier(**{
    'n_neighbors': 16,
    'weights': 'distance',
    'algorithm': 'auto',
    'leaf_size': 11
})
dt_tuned= DecisionTreeClassifier(**{
    'criterion': 'entropy',
    'splitter': 'best',
    'max_depth': 31,
    'min_samples_leaf': 1,
    'max_features': 'log2'
})

models = {
    # base models
    'logres': LogisticRegression(max_iter= 10_000),
    'sgd': SGDClassifier(),
    'svc': SVC(kernel= 'sigmoid'),
    'knn': KNeighborsClassifier(),
    'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(),
    'ada': AdaBoostClassifier(),
    'gb': GradientBoostingClassifier(),
    'xgb': XGBClassifier(),
    'lgb': LGBMClassifier(),
    
    # tuned models
    'lgb_tuned': lgb_tuned,
    'knn_tuned': knn_tuned,
    'dt_tuned': dt_tuned,

    # ensemble
    'hard_voting': VotingClassifier(
        estimators= [lgb_tuned, knn_tuned, dt_tuned], 
        voting= 'hard',
        n_jobs= -1, 
        verbose= 0
    ),
    'soft_voting': VotingClassifier(
        estimators= [lgb_tuned, knn_tuned, dt_tuned], 
        voting= 'soft',
        n_jobs= -1, 
        verbose= 0
    ),
}