"""Train the workshop classifiers."""

from src.models.classifiers import MODEL_FACTORIES


def train_all(X_train, y_train, n_neighbors: int = 5) -> dict:
    """Train every workshop model; return {name: fitted_estimator}."""
    models = {}
    for name, factory in MODEL_FACTORIES.items():
        kwargs = {}
        if name == "KNN":
            kwargs["n_neighbors"] = n_neighbors
        model = factory(**kwargs)
        model.fit(X_train, y_train)
        models[name] = model
        print(f"[models] Trained {name} ...")
    return models