import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.feature_selection import RFE


class CorrelationRemover:
    def __init__(self, corr_th):
        self.pairs=[]
        self.max_corr =  corr_th

    def fit_transform(self, X, ):
        self.pairs = []
        corr = X.corr()
        for i in corr.columns: # -1 is for self-correlation
            for j in corr.columns:
                if corr.loc[i, j] > self.max_corr:
                    self.pairs.append((i, j))

        self.to_remove = []
        to_keep = []
        for idx1, idx2 in self.pairs:
            if idx1 > idx2:  # Remove duplicates
                if idx1 not in to_keep:
                    if idx1 not in self.to_remove:
                        self.to_remove.append(idx1)
                    to_keep.append(idx2)
                else:
                    self.to_remove.append(idx2)

        return self.transform(X)

    def transform(self, X):
        X = X.drop(self.to_remove, axis=1)
        return X

    def get_removed_num(self):
        return len(self.to_remove)


class DataTransformerBoruta:
    def __init__(self, corr_th, n_est=500, seed=123):
        self.boruta = True
        rfc = RandomForestClassifier(n_estimators=n_est, class_weight="balanced", n_jobs=6)
        self.feature_selector = BorutaPy(rfc, n_estimators="auto", verbose=0, random_state=seed, max_iter=100)
        self.corr_rem = CorrelationRemover(corr_th)

    def fit_transform(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y).reshape(-1)
        self.feature_selector.fit(X_arr, y_arr)
        X_columns = X.columns
        selected_columns = X_columns[self.feature_selector.support_]
        X = X[selected_columns]
        X = self.corr_rem.fit_transform(X)
        return X


    def transform(self, X):
        X_columns = X.columns
        selected_columns = X_columns[self.feature_selector.support_]
        X = X[selected_columns]
        X = self.corr_rem.transform(X)
        return X

    def get_selected_num(self):
        return self.feature_selector.n_features_ - self.corr_rem.get_removed_num()


class DataTransformerRFE:
    def __init__(self, corr_th, features_num):
        self.corr_rem = CorrelationRemover(corr_th)
        model = linear_model.LogisticRegression()
        self.feature_selector = RFE(model, features_num)

    def fit_transform(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y).reshape(-1)
        self.feature_selector.fit(X_arr, y_arr)
        X_columns = X.columns
        selected_columns = X_columns[self.feature_selector.support_]
        X = X[selected_columns]
        X = self.corr_rem.fit_transform(X)
        return X

    def transform(self, X):
        X_columns = X.columns
        selected_columns = X_columns[self.feature_selector.support_]
        X = X[selected_columns]
        X = self.corr_rem.transform(X)
        return X

    def get_selected_num(self):
        return self.feature_selector.n_features_ - self.corr_rem.get_removed_num()




