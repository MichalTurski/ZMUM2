import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score


#%% Evaluation metrics
def print_dataset_scores(y_true, y_predicted, set_name):
    balanced = balanced_accuracy_score(y_true, y_predicted)
    accuracy = accuracy_score(y_true, y_predicted)
    precission = precision_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted)
    print(f"On {set_name} set: balanced score = {balanced}, accuracy = {accuracy}, precission = {precission}, recall = {recall}")


def print_classifier_scores(classifier, X_train, y_train, X_test, y_test):
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    print_dataset_scores(y_train, y_train_pred, "train")
    print_dataset_scores(y_test, y_test_pred, "test")


def balanced_scorer(classifier, X, y):
    y_pred = classifier.predict(X)
    balanced = balanced_accuracy_score(y, y_pred)
    return balanced
