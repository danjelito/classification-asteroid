from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif

import config


def clean_col(col):
    return (col
        .lower()
        .replace("(", " ")
        .strip()
        .replace(")", " ")
        .strip()
        .replace(" ", "_")
    )


# columns that are used
USED_COLS = [
    "Absolute Magnitude",
    "Est Dia in M(max)",
    "Epoch Date Close Approach",
    "Relative Velocity km per sec",
    "Miss Dist.(Astronomical)",
    "Orbit ID",
    "Orbit Uncertainity",
    "Minimum Orbit Intersection",
    "Jupiter Tisserand Invariant",
    "Epoch Osculation",
    "Eccentricity",
    "Inclination",
    "Asc Node Longitude",
    "Orbital Period",
    "Perihelion Distance",
    "Perihelion Arg",
    "Mean Anomaly",
    "Hazardous",
]

# label
LABEL = ["Hazardous"]

# all numerical columns
NUM_COLS = [col for col in USED_COLS if col not in LABEL]

# numerical columns that are not skewed
NUM_NOT_SKEWED_COLS = [
    "Absolute Magnitude",
    "Epoch Date Close Approach",
    "Miss Dist.(Astronomical)",
    "Orbit Uncertainity",
    "Jupiter Tisserand Invariant",
    "Eccentricity",
    "Asc Node Longitude",
    "Perihelion Distance",
    "Perihelion Arg",
    "Mean Anomaly",
]

# numerical columns that are skewed
NUM_SKEWED_COLS = [
    "Est Dia in M(max)",
    "Epoch Osculation",
    "Orbital Period",
    "Minimum Orbit Intersection",
    "Inclination",
    "Relative Velocity km per sec",
    "Orbit ID",
]

# check column list with assertion
assert sorted(NUM_COLS + LABEL) == sorted(USED_COLS)
assert sorted(NUM_SKEWED_COLS + NUM_NOT_SKEWED_COLS) == sorted(NUM_COLS)

# numerical pipeline
num_pipe = Pipeline([
    ("impute", KNNImputer(n_neighbors=5)), ("scale", StandardScaler())
])

# numerical skewed pipeline
num_skewed_pipe = Pipeline([
    ("impute", KNNImputer(n_neighbors=5)), ("scale", RobustScaler())
])

# full preprocessing pipeline
preprocessing = ColumnTransformer([
    ("numerical", num_pipe, [clean_col(col) for col in NUM_NOT_SKEWED_COLS]),
    ("numerical_skewed", num_skewed_pipe, [clean_col(col) for col in NUM_SKEWED_COLS]),
])

# feature compression pipeline
compression = Pipeline([
    ("pca", PCA(n_components= 2, random_state=config.RANDOM_STATE))
])

# feature selecction pipleine
f_selection = Pipeline([
    ("select_k_best", SelectKBest(k= 3, score_func= mutual_info_classif))
])
