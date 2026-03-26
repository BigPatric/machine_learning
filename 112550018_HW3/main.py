import numpy as np
import pandas as pd
from loguru import logger
import random

import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, plot_feature_importance
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    """You can control the seed for reproducibility"""
    random.seed(777)
    torch.manual_seed(777)

    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"].to_numpy()  # (n_samples, )

    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"].to_numpy()

    feature_names = list(train_df.drop(["target"], axis=1).columns)

    """
    TODO: Implement you preprocessing function.
    """

    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])
        X_train_num = X_train.drop(categorical_cols, axis=1).to_numpy()
        X_test_num = X_test.drop(categorical_cols, axis=1).to_numpy()
        X_train = np.hstack([X_train_num, X_train_cat])
        X_test = np.hstack([X_test_num, X_test_cat])
        feature_names = list(
            train_df.drop(["target"], axis=1).drop(categorical_cols, axis=1).columns
        ) + list(encoder.get_feature_names_out(categorical_cols))
    else:
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    """
    TODO: Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    clf_adaboost = AdaBoostClassifier(input_dim=X_train.shape[1], num_learners=10)
    _ = clf_adaboost.fit(X_train, y_train, num_epochs=100, learning_rate=0.01)

    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    final_pred = np.round(np.mean(y_pred_classes, axis=0)).astype(int)
    accuracy_ = np.mean(final_pred == y_test)
    logger.info(f"AdaBoost - Accuracy: {accuracy_:.4f}")
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="./adaboost_roc.png",
    )
    feature_importance = clf_adaboost.compute_feature_importance()
    plot_feature_importance(
        feature_names=feature_names,
        importances=feature_importance,
        fpath="./adaboost_feature_importance.png",
    )

    # Bagging
    clf_bagging = BaggingClassifier(input_dim=X_train.shape[1])
    _ = clf_bagging.fit(X_train, y_train, num_epochs=100, learning_rate=0.01)

    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    final_pred = np.round(np.mean(y_pred_classes, axis=0)).astype(int)
    accuracy_ = np.mean(final_pred == y_test)
    logger.info(f"Bagging - Accuracy: {accuracy_:.4f}")
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="./bagging_roc.png",
    )
    feature_importance = clf_bagging.compute_feature_importance()
    plot_feature_importance(
        feature_names=feature_names,
        importances=feature_importance,
        fpath="./bagging_feature_importance.png",
    )

    # Decision Tree
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f"DecisionTree - Accuracy: {accuracy_:.4f}")
    feature_importance = clf_tree.compute_feature_importance()
    plot_feature_importance(
        feature_names=feature_names,
        importances=feature_importance,
        fpath="./decisiontree_feature_importance.png",
    )


if __name__ == "__main__":
    main()
