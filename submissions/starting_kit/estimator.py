import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, \
    OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

N_JOBS = 1

def get_estimator():
    cat_col = ['country']
    
    categorical_transformer = Pipeline(steps=[
        ('encode', OrdinalEncoder())
    ])
    

    preprocessor = ColumnTransformer(
        transformers=[
            #('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_col),
        ], remainder='passthrough')
    

    regressor = RandomForestRegressor(
            n_estimators=5, max_depth=50, max_features="auto", n_jobs=N_JOBS)
    

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('regressor', regressor)
    ])

    return pipeline
    