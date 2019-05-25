# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_prepare
from evaluation_metrics import *
from sklearn.model_selection import KFold
import statistics as stat
import random as random
from numpy.random import uniform

#%% Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %% Import file with pandas
train_data = pd.read_csv('artificial_train.data', delim_whitespace=True, header=None)
train_labels = pd.read_csv('artificial_train.labels', delim_whitespace=True, header=None)
valid_data = pd.read_csv('artificial_valid.data', delim_whitespace=True, header=None)
df = pd.concat([train_data, train_labels[0]], axis=1)

#%% Rename dataset
X = train_data
y = train_labels

# %% Preprocess data
data_transformer = data_prepare.DataTransformer()

#%% XGBoost, but run only once, cross validated
import xgboost as xgb

single_xg_switch = True
if single_xg_switch:
    max_depth = 7
    bootstrap = False
    min_samples_leaf = 1
    min_samples_split = 5
    n_estimators = 103
    eta = 0.0222
    gamma = 0.094
    subsample = 0.8567
    colsample_bytree = 0.5106

    results = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        X_train = X.iloc[train]
        X_test = X.iloc[test, :]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        # X_train = data_transformer.fit_transform(X_train, y_train)
        # X_test = data_transformer.transform(X_test)

        data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
        xg_clas = xgb.XGBClassifier(max_depth=max_depth,
                                    bootstrap=bootstrap,
                                    min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    n_estimators=n_estimators,
                                    learning_rate=eta,
                                    gamma=gamma,
                                    subsample=subsample,
                                    colsample_bylevel=colsample_bytree,
                                    n_jobs=8)
        xg_clas.fit(X_train, y_train)

        # print(xg_clas.predict_proba(X_test))
        # print_classifier_scores(xg_clas, X_train, y_train, X_test, y_test)
        results.append(balanced_scorer(xg_clas, X_test, y_test))

    curr_score = stat.mean(results)
    dev = stat.stdev(results)
    print(f"Score = {curr_score}, deviation = {dev}, params = {max_depth}, "
          f"{bootstrap}, {min_samples_leaf}, {min_samples_split},"
          f"{n_estimators}, {eta}, {gamma}, {subsample}, {colsample_bytree}")

#%% XGBoost, but random grid search and cross validated
import xgboost as xgb

cv_xg_switch = False
results = []
kf = KFold(n_splits=10)

best_score = 0.

if cv_xg_switch:
    for i in range(40):
        max_depth = random.choice(np.arange(1, 15, 2))
        bootstrap = random.choice([True, False])
        min_samples_leaf = random.choice([1, 2, 4])
        min_samples_split = random.choice([2, 5, 10])
        n_estimators = random.choice(np.arange(3, 200, 10))
        eta = round(np.random.uniform(0.01, 0.03), 4)
        gamma = round(np.random.uniform(0.0, 0.2), 4)
        subsample = round(np.random.uniform(0.6, 0.9), 4)
        colsample_bytree = round(np.random.uniform(0.5, 0.8), 4)

        for train, test in kf.split(X):
            X_train = X.iloc[train]
            X_test = X.iloc[test, :]
            y_train = y.iloc[train]
            y_test = y.iloc[test]
            # X_train = data_transformer.fit_transform(X_train, y_train)
            # X_test = data_transformer.transform(X_test)

            data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
            xg_clas = xgb.XGBClassifier(max_depth=max_depth,
                                        bootstrap=bootstrap,
                                        min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split,
                                        n_estimators=n_estimators,
                                        learning_rate=eta,
                                        gamma=gamma,
                                        subsample=subsample,
                                        colsample_bylevel=colsample_bytree,
                                        n_jobs=8)
            xg_clas.fit(X_train, y_train)

            # print(xg_clas.predict_proba(X_test))
            #print_classifier_scores(xg_clas, X_train, y_train, X_test, y_test)
            results.append(top10_scorer(xg_clas, X_test, y_test))
        curr_score = stat.mean(results)
        dev = stat.stdev(results)
        print(f"Score = {curr_score}, deviation = {dev}, params = {max_depth}, "
              f"{bootstrap}, {min_samples_leaf}, {min_samples_split},"
              f"{n_estimators}, {eta}, {gamma}, {subsample}, {colsample_bytree}")

        if curr_score > best_score:
            best_score = curr_score
            best_max_depth = max_depth
            best_bootstrap = bootstrap
            best_min_samples_leaf = min_samples_leaf
            best_min_samples_split = min_samples_split
            best_n_estimators = n_estimators
            best_eta = eta
            best_gamma = gamma
            best_subsample = subsample
            best_colsample_bytree = colsample_bytree

    print(f"Best score = {best_score}, params = {best_max_depth}, "
      f"{best_bootstrap}, {best_min_samples_leaf}, {best_min_samples_split},"
      f"{best_n_estimators}, {best_eta}, {best_gamma}, {best_subsample},"
      f"{best_colsample_bytree}")

# %% XGBoost, final classifier
    import xgboost as xgb

    final_xg_switch = True
    if final_xg_switch:
        max_depth = 7
        bootstrap = False
        min_samples_leaf = 1
        min_samples_split = 5
        n_estimators = 103
        eta = 0.0222
        gamma = 0.094
        subsample = 0.8567
        colsample_bytree = 0.5106

        X_train = X
        y_train = y
        X_train = data_transformer.fit_transform(X_train, y_train)
        X_test = data_transformer.transform(X_test)

        data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
        xg_clas = xgb.XGBClassifier(max_depth=max_depth,
                                    bootstrap=bootstrap,
                                    min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    n_estimators=n_estimators,
                                    learning_rate=eta,
                                    gamma=gamma,
                                    subsample=subsample,
                                    colsample_bylevel=colsample_bytree,
                                    n_jobs=8)
        xg_clas.fit(X_train, y_train)

        probas = xg_clas.predict_proba(X_test)[:, 1]
        np.savetxt("MICTUR.txt", probas, fmt='%1.8f')



