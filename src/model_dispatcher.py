from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np

from skopt import space

models = {
    # "logres": LogisticRegression(max_iter= 10_000),
    # "sgd": SGDClassifier(),
    # "svc": SVC(kernel= 'sigmoid'),
    # "knn": KNeighborsClassifier(),
    # "dt": DecisionTreeClassifier(),
    # "rf": RandomForestClassifier(),
    # "ada": AdaBoostClassifier(),
    # "gb": GradientBoostingClassifier(),
    "xgb": XGBClassifier(),
    "lgb": LGBMClassifier(),
}