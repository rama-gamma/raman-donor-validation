"""PCA used throughout the analysis.
With svd_solver='auto' (the setting used to produce the reported results), scikit-learn selects a randomized
SVD without a fixed random state for matrices of this size, so repeated runs differ slightly (in a test,
>99% identical predictions; accuracy changes well below one seed-SD). Set RAMAN_PCA_SOLVER=full for exact,
deterministic runs."""
import os
from sklearn.decomposition import PCA as _PCA
SOLVER = os.environ.get('RAMAN_PCA_SOLVER', 'auto')
def PCA(n_components=None, **kw):
    kw.setdefault('svd_solver', SOLVER)
    return _PCA(n_components, **kw)
