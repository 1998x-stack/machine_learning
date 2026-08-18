"""Productization smoke tests for the `machine_learning` algorithm collection.

1. Every algorithm module imports cleanly (no import-time side effects, no hangs).
2. KNN runs end-to-end on the iris toy dataset with a high accuracy score.
"""
from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from module_loader import REPO_ROOT, load_module


def test_all_modules_import_cleanly():
    """Every module imports without raising (fixes import-time demo side effects)."""
    imported = []
    for p in sorted(REPO_ROOT.rglob("*.py")):
        if ".git" in p.parts or "figures" in p.parts or "tests" in p.parts or "examples" in p.parts or "scripts" in p.parts:
            continue
        if p.name in {"ml_dataset.py", "conftest.py"}:
            continue
        load_module(p)
        imported.append(p.relative_to(REPO_ROOT).as_posix())
    assert len(imported) == 22, f"expected 22 modules, got {len(imported)}"


def test_knn_classifier_on_iris():
    """KNN/KNNClassifier learns iris with >= 90% accuracy on a held-out split."""
    knn_mod = load_module(REPO_ROOT / "KNN" / "KNN.py")
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = knn_mod.KNNClassifier(k=3)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    assert acc >= 0.90, f"KNN accuracy too low: {acc:.4f}"