import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

problem_title = 'Suicide Rate Challenge'
_target_column_name = 'suicide_total_deaths'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.RelativeRMSE(name='relative_rmse')
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
    return cv.split(X, y)


def _read_data(path, f_name, geo_name):
    data = pd.read_csv(os.path.join(path, 'data/processed', f_name))
    geo_data = pd.read_csv(os.path.join(path, 'data/processed', geo_name))
    
    y_array = data[_target_column_name].values / data['population_total'].values
    X_df = data.drop([_target_column_name], axis=1)
    X_df = pd.merge(
        X_df, geo_data, left_on='country',
        right_on='country', how='left') 
    
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'challenge_suicide_total_deaths_dataset.csv'
    geo_name = 'geo_data_per_country.csv'
    return _read_data(path, f_name, geo_name)


def get_test_data(path='.'):
    f_name = 'test_2016_dataset.csv'
    geo_name = 'geo_data_per_country.csv'
    return _read_data(path, f_name, geo_name)
