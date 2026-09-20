"""
Evaluation and domain separability for Task 2.

Domain separability: logistic regression (C=1) on frozen backbone features.
70/30 split, seed 6304. 50% = chance.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


def domain_separability(source_feats, target_feats, seed=SEED):
    """Binary logistic regression to distinguish source vs target."""
    n = min(len(source_feats), len(target_feats))
    rng = np.random.RandomState(seed)
    si = rng.choice(len(source_feats), n, replace=False)
    ti = rng.choice(len(target_feats), n, replace=False)
    X = np.concatenate([source_feats[si], target_feats[ti]])
    y = np.array([0]*n + [1]*n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.score(Xte, yte)
