import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


class DataTransformer:
    def __init__(self):
        self.drop_th = 0.005
        self.drop_categories = False
        self.m = 300
        self.encoding_map = {}
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

    def __drop_empty_col_fit(self, data):
        # drops nearly empty columns
        entry_count = data.shape[0]
        not_enough_flt = (data.count() < self.drop_th * entry_count)
        self.to_drop_empty = data.columns[not_enough_flt]
        return self.__drop_empty_col(data)

    def __drop_empty_col(self, data):
        return data.drop(list(self.to_drop_empty), axis=1, inplace=True)

    def __drop_empty_row(self, data):
        # Drop empty rows:
        values_in_row = (~(data.isna())).sum(axis=1)
        data = data[values_in_row > 1]
        return data

    def __drop_categories_fit(self, data):
        # Maybe drop column with too many categories?
        self.to_drop_categories = []
        if self.drop_categories:
            columns = data.select_dtypes(exclude=["number"]).columns
            to_drop = []
            for col in columns:
                if data[col].nunique() > 100:
                    to_drop.append(col)

            self.to_drop_categories = data[to_drop]
            self.__drop_categories(data)

    def __drop_categories(self, data):
        data = data.drop(self.to_drop_categories, axis=1, inplace=True)

    def __unify_categories_fit(self, data):
        # If there are too many categories, unify them
        to_unify = []
        for col in data.columns:
            if np.issubdtype(data[col].dtype, np.number):
                to_unify.append(False)
            else:
                if data[col].nunique() > 64:
                    to_unify.append(True)
                else:
                    to_unify.append(False)

        self.to_unify_categories = data.columns[to_unify]
        self.__unify_categories(data)


    def __unify_categories(self, data):
        for col in self.to_unify_categories:
            data[col] = data[col].replace(to_replace=r'.*', value="a", regex=True)

    def __mean_encoding_fit(self, data):
        # mean encoding
        def smooth_mean(df, col, m):
            col_mean = df["class"].mean()
            cat_stats = df.fillna(-1).groupby(col)["class"].agg(["count", "mean"])
            cat_count = cat_stats["count"]
            cat_mean = cat_stats["mean"]
            means_smoothed = (cat_count * cat_mean + m * col_mean) / (cat_count + m)
            return means_smoothed

        columns = data.select_dtypes(exclude=["number"]).columns
        for col in columns:
            self.encoding_map[col] = smooth_mean(data, col, self.m)
        self.__mean_encoding(data)

    def __mean_encoding(self, data):
        columns = data.select_dtypes(exclude=["number"]).columns
        for col in columns:
            data[col] = data[col].fillna(-1).map(self.encoding_map[col])

    def fit_transform(self, X, y):
        data = pd.merge(X, y, left_index=True, right_index=True)
        self.__drop_empty_col_fit(data)
        self.__drop_empty_row(data)
        self.__drop_categories_fit(data)
        self.__unify_categories_fit(data)
        self.__mean_encoding_fit(data)
        X = data.drop(data.columns[len(data.columns) - 1], axis=1)
        self.imputer.fit(X, y)
        X_imputed = pd.DataFrame(self.imputer.transform(X))
        X_imputed.columns = X.columns
        X_imputed.index = X.index

        return X_imputed


    def transform(self, data):
        self.__drop_empty_col(data)
        self.__drop_categories(data)
        self.__unify_categories(data)
        self.__mean_encoding(data)
        data_imputed = pd.DataFrame(self.imputer.transform(data))
        data_imputed.columns = data.columns
        data_imputed.index = data.index

        return data_imputed



