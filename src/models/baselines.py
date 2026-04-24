"""Baseline model factories for tabular healthcare experiments."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_baseline_estimators() -> dict:
    """
    Return baseline estimators used for ML comparison.
    """

    return {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced" # thêm bằng tay khi thầy bảo cho implanced dataset
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"  # thêm bằng tay khi thầy bảo cho implanced dataset
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"  # thêm bằng tay khi thầy bảo cho implanced dataset
        ),

        #"extra_trees": ExtraTreesClassifier(
        #    n_estimators=200,
        #    random_state=42
        #),

        #"gradient_boosting": GradientBoostingClassifier(
        #    random_state=42
        #),

        "knn": KNeighborsClassifier(
            n_neighbors=5
        ),

        #"svm": SVC(
        #    #probability=True
        #    probability=False
        #),
    }