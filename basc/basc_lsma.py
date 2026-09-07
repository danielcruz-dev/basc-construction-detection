"""
Fully constrained Linear Spectral Mixture Analysis on the Sentinel-2 stacks.

Wu & Murray (2003) publish their endmember spectra only as a plot (Fig. 5) --
there is no numeric table in the paper -- and those spectra were image-derived
from feature-space extremes of one July-1999 Columbus ETM+ scene. So this
replicates their METHOD rather than transplanting their numbers: endmembers are
extracted from the imagery itself as the vertices of the data simplex (N-FINDR
on the leading principal components), then labelled by spectral character.

Endmembers: Vegetation, Soil, High Albedo Surface, Low Albedo Surface.
Unmixing is fully constrained (non-negative, sums to one), as in the paper.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import tifffile
from sklearn.decomposition import PCA

BANDS = ["B02","B03","B04","B08","B11","B12"]
EM_NAMES = ["vegetation","soil","high_albedo","low_albedo"]


CLP_MAX = 0.65 * 255      # s2cloudless probability, as in the paper


def load_stack(path, clp_max=CLP_MAX):
    """-> reflectance (H,W,6) and valid mask (H,W) bool.

    Handles both stack layouts: 7 bands (6 reflectance + SCL-derived valid) and
    8 bands (…+ CLP before valid). Where CLP is present it is ANDed into the
    mask: SCL alone leaves thin cirrus in, and that cirrus unmixes as high
    albedo -- the artefact this fraction is most sensitive to.
    """
    a = tifffile.imread(path)
    if a.ndim == 2:
        raise ValueError(f"{path}: unexpected 2-D raster")
    if a.ndim == 3 and a.shape[0] in (7, 8) and a.shape[0] < a.shape[-1]:
        a = np.transpose(a, (1, 2, 0))    # band-first -> band-last
    nb = a.shape[-1]
    refl = a[..., :6].astype(np.float32)
    if nb >= 8:
        valid = (a[..., 7] > 0.5) & (a[..., 6] <= clp_max)
    else:
        valid = a[..., 6] > 0.5
    return refl, valid


def site_pixels(site_dir, max_per_period=4000, seed=0):
    """Pooled valid reflectance samples across every period of one site."""
    rng = np.random.default_rng(seed)
    out = []
    for fn in sorted(glob.glob(os.path.join(site_dir, "*.tif"))):
        try:
            r, m = load_stack(fn)
        except Exception:
            continue
        v = r[m]
        if len(v) == 0:
            continue
        # drop physically impossible reflectance (cloud edges, sensor artefacts)
        v = v[(v > 0).all(1) & (v < 1.2).all(1)]
        if len(v) == 0:
            continue
        if len(v) > max_per_period:
            v = v[rng.choice(len(v), max_per_period, replace=False)]
        out.append(v)
    return np.vstack(out) if out else np.zeros((0, 6), np.float32)


def nfindr(X3, n=4, iters=60, seed=0):
    """Indices of the n pixels maximising simplex volume in 3-D PC space."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X3), n, replace=False)
    def vol(ix):
        E = np.hstack([np.ones((n, 1)), X3[ix]])   # n x (1+3)
        return abs(np.linalg.det(E))
    best = vol(idx)
    for _ in range(iters):
        improved = False
        for j in range(n):
            cand = rng.choice(len(X3), 512, replace=False)
            for c in cand:
                trial = idx.copy(); trial[j] = c
                v = vol(trial)
                if v > best:
                    best, idx, improved = v, trial, True
        if not improved:
            break
    return idx, best


def extract_endmembers(X, seed=0, trim=0.005, knn=200):
    """X (N,6) -> (4,6) endmember matrix, labelled, plus diagnostics.

    Two robustness steps, both needed. N-FINDR maximises simplex volume, so it
    is drawn to whatever single pixel is most extreme -- a thin cloud the SCL
    mask missed, or a sensor artefact. Left raw it produced a "soil" endmember
    peaking at B12=0.85 (real soil peaks at B11) and a "high albedo" rising
    into the NIR, which is an ice signature, not a roof.

      1. per-band winsorising, not just brightness: the bad soil vertex sat at
         moderate mean brightness, so a brightness-only trim never saw it.
      2. each vertex is replaced by the MEDIAN of its `knn` nearest neighbours
         in PC space -- keeps the vertex location, drops the single-pixel noise.
    """
    n0 = len(X)
    if trim > 0 and n0 > 1000:
        lo = np.quantile(X, trim, axis=0)
        hi = np.quantile(X, 1 - trim, axis=0)
        X = X[((X >= lo) & (X <= hi)).all(1)]
    pca = PCA(n_components=3, random_state=seed).fit(X)
    X3 = pca.transform(X)
    idx, v = nfindr(X3, 4, seed=seed)

    E = np.empty((4, X.shape[1]))
    for j, i in enumerate(idx):
        d = ((X3 - X3[i]) ** 2).sum(1)
        near = np.argpartition(d, min(knn, len(d) - 1))[:knn]
        E[j] = np.median(X[near], axis=0)

    b = E.mean(1)
    red, nir = E[:, 2], E[:, 3]
    ndvi = (nir - red) / (nir + red + 1e-6)

    lab, taken = {}, set()
    def claim(name, order):
        for i in order:
            if i not in taken:
                lab[name] = int(i); taken.add(int(i)); return
    claim("vegetation",  np.argsort(-ndvi))
    claim("high_albedo", np.argsort(-b))
    claim("low_albedo",  np.argsort(b))
    claim("soil",        np.argsort(-(E[:, 4] - E[:, 3])))
    M = np.vstack([E[lab[n]] for n in EM_NAMES])
    return M, {"n_pixels": int(n0), "n_after_trim": int(len(X)),
               "simplex_volume": float(v), "knn": knn,
               "pca_explained": pca.explained_variance_ratio_.tolist(),
               "ndvi": ndvi[[lab[n] for n in EM_NAMES]].tolist(),
               "brightness": b[[lab[n] for n in EM_NAMES]].tolist()}


def _subset_solvers(M):
    """Precompute equality-constrained LS solvers for every active set.

    For a subset S of endmembers, min ||M_S f - r||^2 s.t. sum(f)=1 has the
    KKT system [[M_S^T M_S, 1],[1^T, 0]] [f; lam] = [M_S^T r; 1]. The matrix
    depends only on M, so it is inverted once per subset and applied to every
    pixel at once -- exact FCLS without a per-pixel solver.
    """
    out = []
    for mask in range(1, 16):
        S = [i for i in range(4) if mask >> i & 1]
        A = M[S].T                                  # 6 x k
        k = len(S)
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = A.T @ A
        K[:k, k] = 1.0
        K[k, :k] = 1.0
        try:
            Ki = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            continue
        out.append((S, A, Ki[:k, :k], Ki[:k, k]))
    return out


def unmix(refl, M, _delta=None):
    """Fully constrained LSMA (non-negative, sums to one), vectorised.

    refl (...,6) -> fractions (...,4), residual RMSE (...).
    """
    shp = refl.shape[:-1]
    R = refl.reshape(-1, 6).astype(np.float64)
    n = len(R)
    best_f = np.zeros((n, 4))
    best_e = np.full(n, np.inf)
    for S, A, Binv, c in _subset_solvers(M):
        f_s = R @ A @ Binv.T + c                    # n x k
        ok = (f_s >= -1e-9).all(1)                  # feasible active sets only
        if not ok.any():
            continue
        pred = f_s @ A.T                            # n x 6
        e = np.sqrt(((pred - R) ** 2).mean(1))
        take = ok & (e < best_e)
        if take.any():
            best_e[take] = e[take]
            F = np.zeros((take.sum(), 4))
            F[:, S] = np.clip(f_s[take], 0, None)
            best_f[take] = F
    s = best_f.sum(1, keepdims=True)
    best_f = np.divide(best_f, s, out=np.zeros_like(best_f), where=s > 0)
    return best_f.reshape(*shp, 4), best_e.reshape(shp)
