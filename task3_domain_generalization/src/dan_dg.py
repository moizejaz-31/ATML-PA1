"""
DAN-DG: pairwise source-domain alignment (NO target access).

    L = L_ERM + (lambda_DG / 3) * sum_{e<e'} MMD^2(F(X_e), F(X_e'))

Aligns Photo-Art, Photo-Cartoon, Art-Cartoon. Uses exactly the Task 2 MMD
(`shared.src.losses.compute_mmd`): three RBF kernels with bandwidths 0.5/1/2 x the
median pairwise squared distance, recomputed for each domain pair's combined batch.
"""

from itertools import combinations

from shared.src.losses import compute_mmd


def pairwise_source_mmd(feats_by_domain):
    """Mean MMD^2 over all unordered pairs of source domains.

    feats_by_domain: dict domain -> (B_e, 512) feature tensor.
    Returns (mean_mmd, {"photo|cartoon": value, ...}).
    """
    pair_vals = {}
    for a, b in combinations(list(feats_by_domain), 2):
        pair_vals[f"{a}|{b}"] = compute_mmd(feats_by_domain[a], feats_by_domain[b])
    mean = sum(pair_vals.values()) / len(pair_vals)
    return mean, pair_vals
