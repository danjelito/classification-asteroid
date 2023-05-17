from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer

# columns that are used 
used_cols = [
    'Absolute Magnitude', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Epoch Date Close Approach', 'Relative Velocity km per sec',
    'Miss Dist.(Astronomical)', 'Orbit ID', 'Orbit Uncertainity', 'Minimum Orbit Intersection',
    'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
    'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist',
    'Perihelion Time', 'Mean Anomaly', 'Mean Motion', 'Hazardous'
]

# all numerical columns
num_cols = ['Absolute Magnitude', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Epoch Date Close Approach',
    'Relative Velocity km per sec', 'Miss Dist.(Astronomical)', 'Orbit Uncertainity', 'Minimum Orbit Intersection',
    'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
    'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist',
    'Perihelion Time', 'Mean Anomaly', 'Mean Motion'
]

# categorical columns
cat_cols = ['Orbit ID']

# label
label = ['Hazardous']

# numerical columns that are not skewed
num_not_skewed_cols = [
    'Absolute Magnitude', 'Epoch Date Close Approach', 'Miss Dist.(Astronomical)', 'Orbit Uncertainity', 
    'Jupiter Tisserand Invariant', 'Eccentricity', 'Asc Node Longitude', 'Perihelion Distance', 'Perihelion Arg',
    'Mean Anomaly','Mean Motion'
]

# numerical columns that are skewed
num_skewed_cols = [
    'Est Dia in M(max)', 'Est Dia in M(min)', 'Epoch Osculation', 'Perihelion Time',
    'Orbital Period', 'Minimum Orbit Intersection', 'Inclination', 'Aphelion Dist', 'Semi Major Axis',
    'Relative Velocity km per sec'
]

# check column list with assertion
assert sorted(num_cols + cat_cols + label) == sorted(used_cols)
assert sorted(num_skewed_cols + num_not_skewed_cols) == sorted(num_cols)

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

# categorical pipeline
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy= 'constant', fill_value= 'NONE')),
    ('encode', OneHotEncoder(drop= 'first', sparse= False)),
    ('scale', StandardScaler())
])

# full preprocessing pipeline
preprocessing= ColumnTransformer([
    ('numerical', num_pipe, num_not_skewed_cols),
    ('numerical_skewed', num_skewed_cols, num_skewed_cols),
    ('cat', cat_pipe, cat_cols)
])