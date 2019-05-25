import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


#%% Evaluation metrics
def assignment_score(y_true, y_predicted):
    pred_pairs = pd.DataFrame(np.transpose([y_true.to_numpy(), y_predicted]), columns=["y_true", "y_pred"])
    pred_pairs = pred_pairs.sort_values(by=["y_pred"], ascending=False)
    valuated_size = int(0.1 * pred_pairs.shape[0])
    valuated = pred_pairs.head(n=valuated_size)
    counts = valuated.groupby("y_true").count()
    return counts.iloc[1].values[0] / valuated_size


def top10_scorer(model, X, y_true):
    y_proba = model.predict_proba(X)
    return assignment_score(y_true, y_proba[:,1])


def print_dataset_scores(y_true, y_predicted, y_pred_proba, set_name):
    best_10p = assignment_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_predicted)
    precission = precision_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted)
    print(f"On {set_name} set: 10p score = {best_10p}, accuracy = {accuracy}, precission = {precission}, recall = {recall}")


def print_classifier_scores(classifier, X_train, y_train, X_test, y_test):
    y_train_pred = classifier.predict(X_train)
    y_train_proba = classifier.predict_proba(X_train)
    y_test_pred = classifier.predict(X_test)
    y_test_proba = classifier.predict_proba(X_test)
    print_dataset_scores(y_train, y_train_pred, y_train_proba[:, 1], "train")
    print_dataset_scores(y_test, y_test_pred, y_test_proba[:, 1], "test")
