"""
DAN-DG: Pairwise Source-Domain Alignment (NO target access).

L = L_ERM + (lambda_DG/3) * sum_{e<e'} MMD^2(F(X_e), F(X_e'))

Aligns Photo-ArtPainting, Photo-Cartoon, ArtPainting-Cartoon.
Same MMD implementation as Task 2.
"""

# TODO: Implement DAN-DG training
# Uses shared/src/losses.py compute_mmd
