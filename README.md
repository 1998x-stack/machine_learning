# machine_learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/1998x-stack/machine_learning/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/machine_learning/actions/workflows/ci.yml)

> From-scratch implementations and tutorials for **15 classic machine-learning
> algorithms**, each with a self-contained script and reusable iris/blobs/digits
> datasets. —— 15 个经典机器学习算法手写实现 + 教程,每个算法独立脚本,配套统一数据接口 `ml_dataset.py`。

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python3 examples/smoke_demo.py        # KNN on iris → ~0.93 accuracy
```

每个算法目录内是独立的可运行脚本(self-contained)。Each algorithm folder is a set of
self-contained, runnable scripts.

## 🧠 Algorithms

| Algorithm | Folder | Notes |
|---|---|---|
| AdaBoost | `Adaboost/Adaboost.py` | 提升方法 boosting |
| Conditional Random Field | `CRF/CRF.py` | 概率图模型 |
| Bi-KMeans / KMeans / KMedoids / KernelKMeans | `Cluster/` | 聚类 |
| Decision Tree | `DecisionTree/DecisionTree.py` | 分类回归树 |
| EM (GMM) | `EM/EM.py` | 期望最大化 |
| Hidden Markov Model | `HMM/HiddenMarkovModel.py` | 隐马尔可夫 |
| KNN | `KNN/KNN.py`, `KNN/NearestNeighbor.py` | 最近邻 |
| LDA / Gibbs Sampling | `LDA/LDA.py`, `LDA/Gibbs.py` | 主题模型 |
| Logistic Regression | `Logistic/LR.py`, `Logistic/LR_NewTon.py` | 逻辑回归 / 牛顿法 |
| PCA | `Matrix/PCA.py` | 主成分分析 |
| Naive Bayes | `Naive-Bayes/naive_bayes.py` | 朴素贝叶斯 |
| Optimizers | `Optimization/optimizers.py` | 梯度/牛顿优化 |
| PageRank | `PageRank/PageRank.py` | 网页排序 |
| SVM | `SVM/SVM.py` | 支持向量机 |
| UMAP | `UMAP.py` | 流形降维 |

公共工具: `ml_dataset.py` 统一加载 `iris` / `digits` / `blobs`; `figures/` 存放生成的图。

## 🎓 Tutorial Style

每个算法模块配 `.md` 说明: 概念 → 公式 → 可运行代码 → 预期输出 → 与其他算法的关联。
Each algorithm module carries a concept → formula → runnable-code → expected-output → related-algorithms
tutorial note.

## ✅ Quality Bar

- 所有模块**导入无副作用**: 演示代码已用 `if __name__ == "__main__":` 隔离(修复了 EM / LDA /
  NearestNeighbor 的 import-time 副作用与缺失依赖), 见 `tests/test_smoke.py`。
- 冒烟测试覆盖全部 22 个模块导入 + KNN 端到端学习: `pytest` 全部通过。
- CI(lint + test)见 `.github/workflows/ci.yml`。
- 仅 `numpy / scipy / matplotlib / scikit-learn / seaborn / tqdm` 依赖。

## 🔬 Verified Demo Evidence

> 短时冒烟运行 — 证明代码可端到端执行并能在玩具数据上学习 (not a tuned production result).
> `examples/smoke_demo.py` 输出:

```text
$ python examples/smoke_demo.py
KNN(k=3) iris acc = 0.9333
smoke demo OK

$ python -m pytest tests/ -q -p no:warnings
2 passed in 4.58s
```

## 📄 License

MIT — see `LICENSE`.