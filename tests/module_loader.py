"""Import each loose algorithm module by file path (they are not a package)."""
from __future__ import annotations

import importlib.util
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every editable module under the repo, sorted, excluding helpers/scratch.
SKIP = {"ml_dataset.py"}


def iter_py_modules(root: pathlib.Path):
    for p in sorted(root.rglob("*.py")):
        if ".git" in p.parts or "figures" in p.parts or "tests" in p.parts:
            continue
        if p.name in SKIP:
            continue
        yield p


def load_module(file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        file_path.stem.replace("-", "_"), file_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module