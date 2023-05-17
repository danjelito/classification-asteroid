from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer


def clean_col(col):
    return col.lower().replace('(', ' ').strip().replace(')', ' ').strip().replace(' ', '_')

# columns that are used 
USED_COLS = [
    'Absolute Magnitude', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Epoch Date Close Approach', 'Relative Velocity km per sec',
    'Miss Dist.(Astronomical)', 'Orbit ID', 'Orbit Uncertainity', 'Minimum Orbit Intersection',
    'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
    'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist',
    'Perihelion Time', 'Mean Anomaly', 'Mean Motion', 'Hazardous'
]

# all numerical columns
NUM_COLS = ['Absolute Magnitude', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Epoch Date Close Approach',
    'Relative Velocity km per sec', 'Miss Dist.(Astronomical)', 'Orbit Uncertainity', 'Minimum Orbit Intersection',
    'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
    'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist',
    'Perihelion Time', 'Mean Anomaly', 'Mean Motion', 'Orbit ID'
]

# label
LABEL = ['Hazardous']

# numerical columns that are not skewed
NUM_NOT_SKEWED_COLS = [
    'Absolute Magnitude', 'Epoch Date Close Approach', 'Miss Dist.(Astronomical)', 'Orbit Uncertainity', 
    'Jupiter Tisserand Invariant', 'Eccentricity', 'Asc Node Longitude', 'Perihelion Distance', 'Perihelion Arg',
    'Mean Anomaly','Mean Motion'
]

# numerical columns that are skewed
NUM_SKEWED_COLS = [
    'Est Dia in M(max)', 'Est Dia in M(min)', 'Epoch Osculation', 'Perihelion Time',
    'Orbital Period', 'Minimum Orbit Intersection', 'Inclination', 'Aphelion Dist', 'Semi Major Axis',
    'Relative Velocity km per sec', 'Orbit ID'
]

# check column list with assertion
assert sorted(NUM_COLS + LABEL) == sorted(USED_COLS)
assert sorted(NUM_SKEWED_COLS + NUM_NOT_SKEWED_COLS) == sorted(NUM_COLS)

# numerical pipeline
num_pipe = Pipeline([
    ('impute', KNNImputer(n_neighbors = 5)),
    ('scale', StandardScaler())
])

# numerical skewed pipeline
num_skewed_pipe = Pipeline([
    ('impute', KNNImputer(n_neighbors = 5)),
    ('scale', RobustScaler())
])

# full preprocessing pipeline
preprocessing= ColumnTransformer([
    ('numerical', num_pipe, [clean_col(col) for col in NUM_NOT_SKEWED_COLS]),
    ('numerical_skewed', num_skewed_pipe, [clean_col(col) for col in NUM_SKEWED_COLS]),
])