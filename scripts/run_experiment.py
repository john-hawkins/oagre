import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from oagre import OaGRe
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

samples = 100
test_percent = 0.1
dataset_sizes = [2000, 5000, 10000]
param_sizes = [15,20,25,30] 
outlier_props = [0.4,0.5,0.6,0.7]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

dist_set = ['normal','chisqr','exp','gamma','f','gumbel','laplace','lognormal','poisson','weibull','wald']

def get_random_data(rows):
    dist_index = np.random.randint(len(dist_set))
    distribution = dist_set[dist_index]
    param1 = np.random.randint(2,10)
    param2 = np.random.randint(2,10)
    if distribution == "normal":
        return np.random.normal(param1,param2,rows)
    if distribution == "chisqr":
        return np.random.chisquare(param1,rows)
    if distribution == "exp":
        return np.random.exponential(param1,rows)
    if distribution == "gamma":
        return np.random.gamma(param1,param2,rows)
    if distribution == "f":
        return np.random.f(param1,param2,rows)
    if distribution == "gumbel":
        return np.random.gumbel(param1,param2,rows)
    if distribution == "laplace":
        return np.random.laplace(param1,param2,rows)
    if distribution == "lognormal":
        return np.random.lognormal(param1,param2,rows)
    if distribution == "poisson":
        return np.random.poisson(param1,rows)
    if distribution == "weibull":
        return np.random.weibull(param1,rows)
    if distribution == "wald":
        return np.random.wald(param1,param2,rows)
    return np.random.normal(param1,param2,rows)


def eval_expression(expr, row):
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    inputs = {}
    inputs['sigmoid'] = sigmoid
    columns = list(row.index)
    for c in columns:
       inputs[c] = row[c]
    return eval(expr, inputs)

def add_noise(base_u, base_s, outlier_u, outlier_s, row):
    noise = 0.0
    if row['is_outlier'] == 1:
        noise = np.random.normal(outlier_u, outlier_s)
    else:
        noise = np.random.normal(base_u, base_s)
    return row['base'] + noise

symbols = ["+", "-", "*", "/", "**", "%"]
def get_random_expression(vars):
    expr = ""
    nonlinear_count = 0
    linear_count = 0
    for v in vars:
        coef = round(np.random.random(),3)
        elem = str(coef) + "*" + str(v)
        if np.random.randint(10)>0: # Leave Variables out 10% of the time
            if expr=="":
                expr = expr + elem
            else:
                operator = symbols[np.random.randint(len(symbols))]
                expr = expr + " " + operator +  " " + elem
                if operator in ["+", "-"]:
                    linear_count += 1
                else:
                    nonlinear_count += 1
            if np.random.randint(3)==0: # Add constants for 30% of variables
                operator = symbols[np.random.randint(len(symbols))]
                constant = str( np.random.randint(1,20) )
                expr = expr +  " " + operator +  " " + constant
                if operator in ["+", "-"]:
                    linear_count += 1
                else:
                    nonlinear_count += 1
    return expr, nonlinear_count/(nonlinear_count+linear_count)  

results = pd.DataFrame(columns=["dataset_size", "param_size", "expression", "expression_nonline_prop", "outlier_expression", "outlier_expression_nonline_prop", "mean_base", "outlier_prop", "noise_std", "outlier_std", "base_err", "outlier_mean_err", "GBM", "LGBM", "XTR", "DTR", "OAGRE-XT", "OAGRE-DT"])

for s in range(samples):
    for d in dataset_sizes:
        rez = {}
        rez["dataset_size"] = d
        for p in param_sizes:
            rez["param_size"] = p
            data = pd.DataFrame()
            for i in range(p):
                colname = "C"+str(i)
                data[colname] = get_random_data(d)
            notdone = True
            while notdone:          # Ensure that the expression generates no NULLS
                expression, prop = get_random_expression(list(data.columns))
                try:
                   data['base'] = data.apply(lambda x: eval_expression(expression, x), axis=1)
                   if data['base'].isnull().max():
                       notdone=True
                   else:
                       notdone=False
                except:
                   notdone=True
            rez["expression"] = expression
            rez["expression_nonline_prop"] = prop
            notdone = True
            while notdone:          # Enures that the outlier expression generate no NULLS and sufficient separation
                expr2, prop = get_random_expression(list(data.columns))
                expr2 = "sigmoid(" + expr2 + ")"
                try:
                   data['outlier'] = data.apply(lambda x: eval_expression(expr2, x), axis=1)
                   mini = data['outlier'].min() 
                   maxi = data['outlier'].max()
                   diff = maxi-mini
                   if diff>0.5:
                       notdone=False
                   if data['outlier'].isnull().max():
                       notdone=True
                except:
                   notdone=True
            rez["outlier_expression"] = expr2
            rez["outlier_expression_nonline_prop"] = prop
            #########################################################################################
            # We now have the base target value and the output of a logistic function that determines
            # if a value will be an outlier. We iterate over the possible outlier proportions and add
            # noise determined by the outlier
            for o in outlier_props:
                rez["outlier_prop"] = o
                f = np.random.randint(3,9)
                rez["outlier_std_factor"] = f
                outliers = int(o*d)
                sorted = data.sort_values('outlier', ascending=False).reset_index()
                sorted['is_outlier'] = 0
                for r in range(outliers):
                    sorted.at[r, 'is_outlier'] = 1
                # Add noise to the record
                mean_base = sorted['base'].mean()
                rez["mean_base"] = mean_base
                range_base = sorted['base'].max()-sorted['base'].min()
                base_noise_mean = 0
                std_limit = min(int(range_base/50), 100)
                out_limit = min(int(range_base/50), 1000)
                base_noise_std = np.random.normal( std_limit/2, std_limit/10 )
                outlier_mean = 0 
                outlier_std = f * base_noise_std
                sorted['target'] = sorted.apply(lambda x: add_noise(base_noise_mean, base_noise_std, outlier_mean, outlier_std, x), axis=1)
                # WE NOW HAVE THE DATA -- SPLIT INTO TRAIN AND TEST
                y = sorted['target']
                outies = sorted['is_outlier']
                drop_cols = ['target', 'base', 'outlier', 'is_outlier']
                X = sorted.drop(drop_cols, axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, stratify=outies, random_state=0)
                gbm = GradientBoostingRegressor()
                lgbm = LGBMRegressor()
                dtr = DecisionTreeRegressor(max_depth=5, random_state=0)
                xtr = ExtraTreesRegressor(n_estimators=10, max_depth=5, random_state=0)
                ####
                try:
                    gbm.fit(X_train, y_train)
                    temp = gbm.predict(X_test)
                    gbm_rmse = rmse(temp, y_test)
                except:
                    gbm_rmse = np.nan
                rez["GBM"] = gbm_rmse
                ####
                try:
                    lgbm.fit(X_train, y_train)
                    temp = lgbm.predict(X_test)
                    lgbm_rmse = rmse(temp, y_test)
                except:
                    lgbm_rmse = np.nan
                rez["LGBM"] = lgbm_rmse
                ####
                try:
                    xtr.fit(X_train, y_train)
                    temp = xtr.predict(X_test)
                    xtr_rmse = rmse(temp, y_test)
                except:
                    xtr_rmse = np.nan
                rez["XTR"] = xtr_rmse
                ####
                try:
                    dtr.fit(X_train, y_train)
                    temp = dtr.predict(X_test)
                    dtr_rmse = rmse(temp, y_test)
                except:
                    dtr_rmse = np.nan
                rez["DTR"] = dtr_rmse
                #############################################################
                try:
                    oagre = OaGRe(
                        classifier=ExtraTreesClassifier(n_estimators=10, max_depth=5, random_state=0), 
                        regressor=ExtraTreesRegressor(n_estimators=10, max_depth=5, random_state=0)
                    )
                    oagre.fit(X_train, y_train)
                    temp = oagre.predict(X_test)
                    oagre_rmse = rmse(temp, y_test)
                except:
                    oagre_rmse = np.nan
                rez["OAGRE-XT"] = oagre_rmse
                #############################################################
                try:
                    oagre = OaGRe(
                        classifier=DecisionTreeClassifier(max_depth=5, random_state=0),
                        regressor=DecisionTreeRegressor(max_depth=5, random_state=0)
                    )
                    oagre.fit(X_train, y_train)
                    temp = oagre.predict(X_test)
                    oagre_rmse = rmse(temp, y_test)
                except:
                    oagre_rmse = np.nan
                rez["OAGRE-DT"] = oagre_rmse
                ######
                sorted['err'] = sorted['target'] - sorted['base']
                sorted['abserr'] = sorted['err'].apply(abs) 
                rez["outlier_prop"] = o
                rez["noise_std"] = base_noise_std
                rez["outlier_std"] = outlier_std
                rez["base_err"] = sorted[sorted['is_outlier']==0]['abserr'].mean()
                rez["outlier_mean_err"] = sorted[sorted['is_outlier']==1]['abserr'].mean()
                results = results.append(rez, ignore_index = True)
                results.to_csv("results_summary_file_full_new.csv", header=True)




