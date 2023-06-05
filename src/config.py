from pathlib import Path

RANDOM_STATE = 8

OG_SET = Path.cwd() / 'input/nasa.csv'
TRAIN_SET = Path.cwd() / 'input/train.csv'
TEST_SET = Path.cwd() / 'input/test.csv'
SMOTE_TRAIN_SET = Path.cwd() / 'input/smote_train.csv'

MODEL = Path.cwd() / 'model'