import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from oagre import OAGRE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

test_percent = 0.1

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

datasets = ["data/abalone/data.csv", "data/OnlineNews/OnlineNewsPopularity.csv", "data/WineQuality/data.csv"]
targets = ['Rings', ' shares', 'quality']
drop_columns = [[], ['url'], []]

results = pd.DataFrame(columns=["records", "variables", "target_mean", "target_std", "GBM", "LGBM", "XTR", "OAGRE"])


for i, data in enumerate(datasets):
    targ = targets[i]
    drop_cols = drop_columns[i] 
    print(data, targ)
    rez = {}
    df = pd.read_csv(data)
    rez["records"] = len(df) 
    rez["variables"] = len(df.columns) - 1
    rez["target_mean"] = df[targ].mean() 
    y = df[targ]
    drop_cols = drop_cols + [targ]
    X = df.drop(drop_cols, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=0)

    print("Training data:", len(X_train))
    print("Testing data:", len(X_test))

    numeric_cols = list( X_train.select_dtypes(include="number").columns)
    categorical_cols = list( X_train.select_dtypes(exclude="number").columns)

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OrdinalEncoder()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    gbm = GradientBoostingRegressor()
    lgbm = LGBMRegressor()
    xtr = ExtraTreesRegressor(random_state=0)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('gbm', gbm )
    ])
    model.fit(X_train, y_train)
    temp = model.predict(X_test)
    gbm_rmse = rmse(temp, y_test)
    rez["GBM"] = gbm_rmse
        
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('lgbm', lgbm )
    ])
    model.fit(X_train, y_train)
    temp = model.predict(X_test)
    lgbm_rmse = rmse(temp, y_test)
    rez["LGBM"] = lgbm_rmse

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xtr', xtr )
    ])  
    model.fit(X_train, y_train)
    temp = model.predict(X_test)
    xtr_rmse = rmse(temp, y_test)
    rez["XTR"] = xtr_rmse

    oagre = OAGRE(classifier=ExtraTreesClassifier(n_estimators=10, max_depth=5, random_state=0), regressor=ExtraTreesRegressor(n_estimators=10, max_depth=5, random_state=0))
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('oagre', oagre)
    ])  
    model.fit(X_train, y_train)
    temp = model.predict(X_test)
    oagre_rmse = rmse(temp, y_test)
    rez["OAGRE"] = oagre_rmse

    results = results.append(rez, ignore_index = True)
    
print(results)





