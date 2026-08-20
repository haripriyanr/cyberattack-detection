"""Workshop classifiers: Logistic Regression, KNN, Naive Bayes."""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def build_logistic_regression(**kwargs):
    kwargs.setdefault("max_iter", 1000)
    kwargs.setdefault("random_state", 42)
    return LogisticRegression(**kwargs)


def build_knn(n_neighbors: int = 5, **kwargs):
    kwargs.setdefault("n_neighbors", n_neighbors)
    return KNeighborsClassifier(**kwargs)


def build_naive_bayes(**kwargs):
    return GaussianNB(**kwargs)


MODEL_FACTORIES = {
    "Logistic Regression": build_logistic_regression,
    "KNN": build_knn,
    "Naive Bayes": build_naive_bayes,
}