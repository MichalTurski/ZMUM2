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
import copy

#%% Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %% Import file with pandas
train_data = pd.read_csv('artificial_train.data', delim_whitespace=True, header=None)
train_labels = pd.read_csv('artificial_train.labels', header=None)
valid_data = pd.read_csv("artificial_valid.data", delim_whitespace=True, header=None)

#%% Rename dataset
X = train_data
y = train_labels
# y = train_labels.reshape(-1)
# y.columns = ["class"]

#%% XGBoost, but run only once, cross validated
import xgboost as xgb

single_xg_switch = False
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
    corr_th: float = 0.90

    results = []
    selected = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        X_train = X.iloc[train, :]
        X_test = X.iloc[test, :]
        y_train = y.iloc[train, :]
        y_test = y.iloc[test, :]

        data_transformer = data_prepare.DataTransformerBoruta(corr_th)
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

        # print(xg_clas.predict_proba(X_test))
        # print_classifier_scores(xg_clas, X_train, y_train, X_test, y_test)
        results.append(balanced_scorer(xg_clas, X_test, y_test))
        selected.append(data_transformer.get_selected_num())

    curr_score = stat.mean(results)
    dev = stat.stdev(results)
    print(f"Score = {curr_score}, deviation = {dev}")

    mean_selected = stat.mean(selected)
    dev_selected = stat.stdev(selected)
    print(f"Selected mean = {mean_selected}, deviation = {dev_selected}")

#%% XGBoost, but run only once, cross validated, Recursive feature selection
import xgboost as xgb

single_xg_switch_rfe = False
if single_xg_switch_rfe:
    max_depth = 7
    bootstrap = False
    min_samples_leaf = 1
    min_samples_split = 5
    n_estimators = 103
    eta = 0.0222
    gamma = 0.094
    subsample = 0.8567
    colsample_bytree = 0.5106
    corr_th = 0.90
    features_num = 20

    results = []
    selected = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        X_train = X.iloc[train, :]
        X_test = X.iloc[test, :]
        y_train = y.iloc[train, :]
        y_test = y.iloc[test, :]

        data_transformer = data_prepare.DataTransformerRFE(corr_th, features_num)
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

        # print(xg_clas.predict_proba(X_test))
        # print_classifier_scores(xg_clas, X_train, y_train, X_test, y_test)
        results.append(balanced_scorer(xg_clas, X_test, y_test))
        selected.append(data_transformer.get_selected_num())

    curr_score = stat.mean(results)
    dev = stat.stdev(results)
    print(f"Score = {curr_score}, deviation = {dev}")

    mean_selected = stat.mean(selected)
    dev_selected = stat.stdev(selected)
    print(f"Selected mean = {mean_selected}, deviation = {dev_selected}")


#%% XGBoost, but random grid search and cross validated
import xgboost as xgb

cv_xg_switch = False
kf = KFold(n_splits=10)

best_score = 0.

if cv_xg_switch:
    for i in range(100):
        max_depth = random.choice(np.arange(1, 15, 2))
        bootstrap = random.choice([True, False])
        min_samples_leaf = random.choice([1, 2, 4])
        min_samples_split = random.choice([2, 5, 10])
        n_estimators = random.choice(np.arange(3, 200, 10))
        eta = round(np.random.uniform(0.01, 0.03), 4)
        gamma = round(np.random.uniform(0.0, 0.2), 4)
        subsample = round(np.random.uniform(0.6, 0.9), 4)
        colsample_bytree = round(np.random.uniform(0.5, 0.8), 4)
        corr_th = round(np.random.uniform(0.8, 0.96), 4)

        results = []
        selected = []

        for train, test in kf.split(X):
            X_train = X.iloc[train, :]
            X_test = X.iloc[test, :]
            y_train = y.iloc[train]
            y_test = y.iloc[test]

            data_transformer = data_prepare.DataTransformerBoruta(corr_th)
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

            # print(xg_clas.predict_proba(X_test))
            #print_classifier_scores(xg_clas, X_train, y_train, X_test, y_test)
            results.append(balanced_scorer(xg_clas, X_test, y_test))
            selected.append(data_transformer.get_selected_num())
        curr_score = stat.mean(results)
        dev = stat.stdev(results)
        print(f"Score = {curr_score}, deviation = {dev}, params = {max_depth}, "
              f"{bootstrap}, {min_samples_leaf}, {min_samples_split},"
              f"{n_estimators}, {eta}, {gamma}, {subsample}, {colsample_bytree}, {corr_th}")

        mean_selected = stat.mean(selected)
        dev_selected = stat.stdev(selected)
        print(f"Selected mean = {mean_selected}, deviation = {dev_selected}")

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
      f"{best_colsample_bytree}, {corr_th}")

# %% lightgbm, but run only once, cross validated
import lightgbm as lgb

single_light_switch = False
if single_light_switch:
    max_depth = 7
    bootstrap = False
    min_samples_leaf = 1
    min_samples_split = 5
    n_estimators = 103
    eta = 0.0222
    gamma = 0.094
    subsample = 0.8567
    colsample_bytree = 0.5106
    corr_th = 0.90

    results = []
    selected = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        X_train = X.iloc[train, :]
        X_test = X.iloc[test, :]
        y_train = y.iloc[train, :]
        y_test = y.iloc[test, :]

        data_transformer = data_prepare.DataTransformerBoruta(corr_th)
        X_train = data_transformer.fit_transform(X_train, y_train)
        X_test = data_transformer.transform(X_test)

        # t_train = lgb.Dataset(data=X_train, label=y_train)
        lg_clas = lgb.LGBMClassifier(n_jobs=8)
        lg_clas.fit(X_train, y_train)

        # print(xg_clas.predict_proba(X_test))
        # print_classifier_scores(xg_clas, X_train, y_train, X_test, y_test)
        results.append(balanced_scorer(lg_clas, X_test, y_test))
        selected.append(data_transformer.get_selected_num())

    curr_score = stat.mean(results)
    dev = stat.stdev(results)
    print(f"Score = {curr_score}, deviation = {dev}")

    mean_selected = stat.mean(selected)
    dev_selected = stat.stdev(selected)
    print(f"Selected mean = {mean_selected}, deviation = {dev_selected}")

# %% lightgbm, but random grid search and cross validated
import lightgbm as lgb

cv_lgbm_switch = False
kf = KFold(n_splits=10)

best_score = 0.

if cv_lgbm_switch:
    for i in range(100):
        corr_th: float = 0.90
        # boosting_type = random.choice(['gbdt', 'dart', 'goss', 'rf'])
        num_leaves = random.choice(np.arange(3, 46, 6))
        max_depth = random.choice(np.arange(1, 15, 2))
        learning_rate = round(np.random.uniform(0.03, 0.3), 3)
        n_estimators = random.choice(np.arange(5, 200, 10))
        min_child_samples = random.choice(np.arange(5, 50, 10))
        reg_alpha = round(np.random.uniform(0, 0.05), 3)
        reg_lambda = round(np.random.uniform(0, 0.05), 3)

        results = []
        selected = []

        for train, test in kf.split(X):
            X_train = X.iloc[train, :]
            X_test = X.iloc[test, :]
            y_train = y.iloc[train]
            y_test = y.iloc[test]

            data_transformer = data_prepare.DataTransformerBoruta(corr_th)
            X_train = data_transformer.fit_transform(X_train, y_train)
            X_test = data_transformer.transform(X_test)

            lg_clas = lgb.LGBMClassifier(#boosting_type=boosting_type,
                                         num_leaves=num_leaves,
                                         max_depth=max_depth,
                                         learning_rate=learning_rate,
                                         n_estimators=n_estimators,
                                         min_child_samples=min_child_samples,
                                         reg_alpha=reg_alpha,
                                         reg_lambda=reg_lambda,
                                         n_jobs=8)
            lg_clas.fit(X_train, y_train)

            # print(xg_clas.predict_proba(X_test))
            # print_classifier_scores(xg_clas, X_train, y_train, X_test, y_test)
            results.append(balanced_scorer(lg_clas, X_test, y_test))
            selected.append(data_transformer.get_selected_num())
        curr_score = stat.mean(results)
        dev = stat.stdev(results)
        print(f"Score = {curr_score}, deviation = {dev}, params = {num_leaves}, "
              f"{max_depth}, {learning_rate}, {n_estimators},"
              f"{min_child_samples}, {reg_alpha}, {reg_lambda}")

        mean_selected = stat.mean(selected)
        dev_selected = stat.stdev(selected)
        print(f"Selected mean = {mean_selected}, deviation = {dev_selected}")

        if curr_score > best_score:
            best_score = curr_score
            best_num_leaves = num_leaves
            best_max_depth = max_depth
            best_learning_rate = learning_rate
            best_n_estimators = n_estimators
            best_min_child_samples = min_child_samples
            best_reg_alpha = reg_alpha
            best_reg_lambda = reg_lambda

    print(f"Best score = {best_score}, params = {best_num_leaves}, "
          f"{best_max_depth}, {best_learning_rate}, {best_n_estimators},"
          f"{best_min_child_samples}, {best_reg_alpha}, {best_reg_lambda}")

#%% LightGBM, but run only once, cross validated, Recursive feature selection

single_lgbm_switch_rfe = False
if single_lgbm_switch_rfe:
    num_leaves = 21
    max_depth = 9
    learning_rate = 0.113
    n_estimators = 195
    min_child_samples = 15
    reg_alpha = 0.021
    reg_lambda = 0.042
    features_num = 20
    corr_th = 0.90

    results = []
    selected = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        X_train = X.iloc[train, :]
        X_test = X.iloc[test, :]
        y_train = y.iloc[train, :]
        y_test = y.iloc[test, :]

        data_transformer = data_prepare.DataTransformerRFE(corr_th, features_num)
        X_train = data_transformer.fit_transform(X_train, y_train)
        X_test = data_transformer.transform(X_test)

        lg_clas = lgb.LGBMClassifier(  # boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=8)
        lg_clas.fit(X_train, y_train)

        # print(xg_clas.predict_proba(X_test))
        # print_classifier_scores(xg_clas, X_train, y_train, X_test, y_test)
        results.append(balanced_scorer(lg_clas, X_test, y_test))
        selected.append(data_transformer.get_selected_num())

    curr_score = stat.mean(results)
    dev = stat.stdev(results)
    print(f"Score = {curr_score}, deviation = {dev}")

    mean_selected = stat.mean(selected)
    dev_selected = stat.stdev(selected)
    print(f"Selected mean = {mean_selected}, deviation = {dev_selected}")

# %% XGBoost, final classifier
import xgboost as xgb

final_xg_switch = True
if final_xg_switch:
    max_depth = 9
    bootstrap = False
    min_samples_leaf = 2
    min_samples_split = 10
    n_estimators = 173
    eta = 0.0255
    gamma = 0.0464
    subsample = 0.7711
    colsample_bytree = 0.7802
    corr_th = 0.90

    data_transformer = data_prepare.DataTransformerBoruta(corr_th)

    X_train = X
    y_train = y
    X_test = valid_data
    X_copy = copy.copy(X_test)
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

    print(f"result on train set: {balanced_scorer(xg_clas, X_train, y_train)}")

    probas = xg_clas.predict_proba(X_test)[:, 1]
    selected_variables = data_transformer.get_selected_vec(X_copy)
    np.savetxt("MICTUR_artificial_prediction.txt", probas, fmt='%1.8f')
    np.savetxt("MICTUR_artificial_features.txt", selected_variables, fmt='%d')



