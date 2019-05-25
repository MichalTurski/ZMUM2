#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.impute import *
from sklearn.decomposition import *
from sklearn.feature_selection import *

import category_encoders as ce
from collections import defaultdict


# In[14]:


import os
print(os.getcwd())


# ## 1. Wczytanie danych

# In[17]:


print()
train_df = pd.read_csv("ZMUM_project1/train.txt", sep = " ").sort_index()
test_df = pd.read_csv("ZMUM_project1/testx.txt", sep = " ").sort_index()


# ## 2. Usunięcie pustych kolumn

# In[24]:


def drop_empty_columns(df):
    columns_to_drop =[]
    for column in df.columns:
        if df[column].isna().all():
            columns_to_drop.append(column)
    if columns_to_drop:
        ret_df = df.drop(columns=columns_to_drop)
    return ret_df


# In[25]:


train_df = drop_empty_columns(train_df)
print(train_df)


# ## 3. Zakodowanie kolumn typu str, obj do category

# In[5]:


# def object_as_category(df):
#     new_df = df.copy()
#     categorical_columns = new_df.select_dtypes(exclude=["number"]).columns
#     try:
#         new_df[categorical_columns] = new_df[categorical_columns].fillna("NAN").astype("category")
#         return new_df
#     except ValueError:
#         print("No categorical columns")


# In[6]:


# train_df = object_as_category(train_df)


# ## 4. Usunięcie kolumn kategorycznych powtarzających się

# In[7]:


def count_categories_in_columns(df):
    cat_df = df.select_dtypes(exclude=["number"])
    num_unique_values_map = defaultdict(list)
    for i in range(0, cat_df.shape[1]):
        column = cat_df.iloc[:, i]
        num_unique = column.nunique(dropna=False)
        num_unique_values_map[num_unique].append(column.name)
    return num_unique_values_map


# In[8]:


num_unique_values_map = count_categories_in_columns(train_df)
num_unique_values_map


# In[9]:


def drop_repeated_cat_columns(df, num_unique_values_map):
    columns_to_drop = set()
    for key, value in num_unique_values_map.items():
        len_col_list = len(value)
        if len_col_list > 1:
            for i in range(len_col_list):
                col_name_i = value[i]
                for j in range(i+1, len_col_list):
                    col_name_j = value[j]
                    lab_encoder = ce.OrdinalEncoder()
                    transformed = lab_encoder.fit_transform(df[[col_name_i, col_name_j]])
                    if accuracy_score(transformed.iloc[:, 0], transformed.iloc[:, 1]) > 0.99:
                        columns_to_drop.add(col_name_i)
                        #print("Break" ,key,  i, j, accuracy_score(transformed.iloc[:, 0], transformed.iloc[:, 1]))
                        break
    return df.drop(columns=columns_to_drop)


# In[10]:


train_df = drop_repeated_cat_columns(train_df, num_unique_values_map)


# ## 5. Problem niewystępujących kategorii w zbiorze treningowym

# In[11]:


mod_train = train_df.iloc[:,:-1]
joined_df = pd.concat([mod_train, test_df], sort=False).sort_index()
count_categories_in_columns(joined_df)


# ##  6. Funkcja licząca liczbę kategorii o konkretnej liczbie wystąpień

# In[12]:


def count_categories_by_size(df):
    columns_dict = {}
    cat_columns = df.select_dtypes(exclude=["number"]).columns
    for column in cat_columns:
        columns_dict[column] = train_df[column].value_counts(dropna=False).to_frame().groupby(column).size()
    return columns_dict


# In[13]:


column_dict_categories_by_size = count_categories_by_size(train_df)


# ## 7. Funkcja określająca frakcję jedynek i liczność w danej kategorii

# In[14]:


def count_ones_fraction_and_size_in_category(df):
    columns_dict = {}
    cat_columns = df.select_dtypes(exclude=["number"]).columns
    for column in cat_columns:
        ones_fraction = df[[column, "class"]].fillna(-1).groupby(column)["class"].mean()
        cat_size = df[[column]].fillna(-1).groupby(column).size()
        columns_dict[column] = pd.DataFrame(data={"ones_fraction": ones_fraction, "cat_size": cat_size})
    return columns_dict


# In[15]:


column_dict_categories_description = count_ones_fraction_and_size_in_category(train_df)


# In[16]:


column_dict_categories_description["Var197"].head()


# ## 8. Zdropowanie kolumn z kategoriami o liczności więcej niż 

# In[17]:


def drop_columns_above_n_categories(df, n):
    columns_to_drop = set()
    cat_columns = df.select_dtypes(exclude=["number"]).columns
    for column in cat_columns:
        if train_df[column].nunique(dropna=False) > n:
            columns_to_drop.add(column)
    return df.drop(columns=columns_to_drop)


# In[18]:


train_df = drop_columns_above_n_categories(train_df, 100)


# ## 9. Wypełnienie pustych zmiennych numerycznych i dodanie kolumn informujących o NaN

# In[19]:


def fill_na_in_numerical_add_column(df):
    new_df = df.copy()
    numerical_columns = new_df.select_dtypes(include=["number"]).columns
    for column in numerical_columns:
        if new_df[column].isna().any():
            new_df[column+"_isfilled"] = new_df[column].isna().map({True:1, False:0})
        new_df[column] = new_df[column].fillna(new_df[column].median())
    return new_df


# In[20]:


train_df = fill_na_in_numerical_add_column(train_df)


# In[21]:


train_df.shape


# ## 10. Target encoding

# In[22]:


import xgboost
import lightgbm
import catboost
from sklearn.ensemble import RandomForestClassifier


# In[23]:


def calculate_scores(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return acc, recall, precision


# In[24]:


cat_columns = list(train_df.select_dtypes(exclude=["number"]).columns)
for column in cat_columns:
    train_df[column] = train_df[column].astype("category").cat.codes


# In[25]:


#label_encoder = ce.BinaryEncoder()
#label_encoder = ce.OrdinalEncoder()
X = train_df.loc[:, train_df.columns != "class"]
y = train_df["class"]
#label_encoder.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)


# In[ ]:


X_train = label_encoder.transform(X_train)
X_test = label_encoder.transform(X_test)


# #### XGBModel

# In[ ]:


xgb_model = xgboost.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000)
eval_set = [(X_test, y_test)]
xgb_model.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", eval_set=eval_set, verbose=True)


# In[151]:


y_pred_xgb = xgb_model.predict(X_test)


# In[152]:


calculate_scores(y_test, y_pred_xgb)


# #### LightGBM model

# In[26]:


param_dist = {"max_depth": [-1, 100],
              "learning_rate" : [0.04, 0.05, 0.06],
              "num_leaves": [35, 50],
              "n_estimators": [1000],
              "max_bin": [100, 150],
             }


# In[27]:


lightgbm_model = lightgbm.LGBMClassifier(scale_pos_weight = 2)


# In[45]:


for param in ParameterGrid(param_dist):
    lightgbm_model = lightgbm.LGBMClassifier(scale_pos_weight = 2, **param)
    eval_set = [(X_test, y_test)]
    lightgbm_model.fit(X_train, y_train, early_stopping_rounds=100, eval_set=eval_set, eval_metric="logloss", verbose=False, 
                   categorical_feature=list(cat_columns))
    y_pred_lightgbm = lightgbm_model.predict(X_test)
    print(param)
    print(calculate_scores(y_test, y_pred_lightgbm))


# In[28]:


grid_search = GridSearchCV(lightgbm_model, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=10)
grid_search.fit(X, y, categorical_feature=cat_columns)
grid_search.best_estimator_


# In[39]:


lgb_model = grid_search.best_estimator_
lgb_model.fit(X_train, y_train, categorical_feature=cat_columns)


# In[40]:


y_pred_lightgbm = lgb_model.predict(X_test)


# In[41]:


calculate_scores(y_test, y_pred_lightgbm)


# In[187]:


y_pred_train_lightgbm = lightgbm_model.predict(X_train)


# In[188]:


calculate_scores(y_train, y_pred_train_lightgbm)


# In[ ]:


lightgbm_model.get_params()


# #### CatBoost model

# In[32]:


cat_model = catboost.CatBoostClassifier()


# In[91]:


cat_model.fit(X_train, y_train, verbose=False, cat_features=list(cat_columns))


# In[92]:


y_pred_cat = cat_model.predict(X_test)
calculate_scores(y_test, y_pred_cat)


# In[ ]:




