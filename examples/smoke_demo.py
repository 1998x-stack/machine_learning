"""Deterministic smoke demo: KNN on the iris toy dataset.

Short smoke run (not a tuned production result) — evidence that the algorithm
executes end-to-end and learns on a toy dataset.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(rel: str):
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(p.stem.replace("-", "_"), p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=2026)

    knn_mod = _load("KNN/KNN.py")
    knn = knn_mod.KNNClassifier(k=3)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    print(f"KNN(k=3) iris acc = {acc:.4f}")
    print("smoke demo OK")


if __name__ == "__main__":
    main()