"""

Bases d'ondelettes orthonormées (NMF+QR et SVD) matchées à la taille d'agrégat,
projection, sensibilité WPA, et utilitaires pour l'étude erreur-vs-(p, n_target).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD

from .metrics import per_day_metrics, acf_norm


# --------------------------------------------------------------------------- #
def synthetic_aggregates(pool_tensor, agg_size, n_groups, fit_subsample=None, seed=0):
    """
    Empile `n_groups` agrégats (moyennes de `agg_size` foyers tirés dans pool_tensor)
    → matrice X (n_groups, 48), éventuellement sous-échantillonnée à fit_subsample.
    """
    rng = np.random.default_rng(seed)
    n = pool_tensor.shape[0]
    X = np.vstack([pool_tensor[rng.choice(n, size=agg_size, replace=False)].mean(0)
                   for _ in range(n_groups)])
    if fit_subsample and X.shape[0] > fit_subsample:
        X = X[rng.choice(X.shape[0], size=fit_subsample, replace=False)]
    return X


def fit_svd_basis(X, pmax):
    """SVD hiérarchique : un seul fit, renvoie Q_svd_full (48, pmax) orthonormée."""
    pmax = min(int(pmax), X.shape[1] - 1)
    return TruncatedSVD(n_components=pmax, random_state=42).fit(X).components_.T


def fit_nmf_basis(X, p, max_iter=400, tol=1e-4):
    """NMF (non négative) + QR → base orthonormée (48, p). À refit pour chaque p."""
    Xnn = np.maximum(X, 0.0)
    nmf = NMF(n_components=int(p), init="nndsvda", max_iter=max_iter,
              tol=tol, random_state=42).fit(Xnn)
    Q, _ = np.linalg.qr(nmf.components_.T)
    return Q


def build_matched_basis(pool_tensor, agg_size, cfg, p_list=None, methods=("svd", "nmf")):
    """
    Construit les dictionnaires de bases matchées aux agrégats de `agg_size`.
    SVD : un fit, toutes les tranches gratuites. NMF : un fit par p de `p_list`.
    Renvoie {'svd': {p: W}, 'nmf': {p: W}} (selon `methods`).
    """
    p_list = list(range(1, cfg.pmax + 1)) if p_list is None else [int(p) for p in p_list]
    X = synthetic_aggregates(pool_tensor, agg_size, cfg.n_groups, cfg.fit_subsample, seed=cfg.seed)
    out = {}
    if "svd" in methods:
        Q_svd_full = fit_svd_basis(X, max(p_list))
        out["svd"] = {p: Q_svd_full[:, :p] for p in p_list}
    if "nmf" in methods:
        out["nmf"] = {p: fit_nmf_basis(X, p, cfg.nmf_max_iter, cfg.nmf_tol) for p in p_list}
    return out


# --------------------------------------------------------------------------- #
def project_on_basis(df_agg, W):
    """
    Projette les agrégats journaliers sur span(W) (W orthonormée → H = W·Wᵀ).
    Renvoie (df_coeffs (T,p), df_proj (T,48), df_errors (T, [MAPE,NMAE,MAE,RMSE,AEcorr])).
    MAPE/NMAE en %.
    """
    p = W.shape[1]
    L = df_agg.values
    alpha = L @ W
    L_hat = alpha @ W.T

    df_coeffs = pd.DataFrame(alpha, index=df_agg.index, columns=range(1, p + 1))
    df_proj = pd.DataFrame(L_hat, index=df_agg.index, columns=df_agg.columns)

    md = per_day_metrics(L, L_hat)
    aec = np.array([np.abs(acf_norm(L[t]) - acf_norm(L_hat[t])).mean() for t in range(len(L))])
    df_errors = pd.DataFrame(
        {"MAPE": md["MAPE"], "NMAE": md["NMAE"], "MAE": md["MAE"],
         "RMSE": md["RMSE"], "AEcorr": aec},
        index=df_agg.index,
    )
    return df_coeffs, df_proj, df_errors


def reconstruct(alpha, W):
    """L̂ = α · Wᵀ."""
    return np.asarray(alpha) @ W.T


# --------------------------------------------------------------------------- #
def panel_sensitivity(panel_profiles, W, n_target, quantile=0.99):
    """
    Sensibilité WPA par coordonnée (calculée sur le PANEL public).
    Renvoie (Delta, Delta_mean) où Delta = q-quantile(|coef individuel|),
    Delta_mean = Delta / n_target.
    """
    coeffs = np.asarray(panel_profiles) @ W
    Delta = np.quantile(np.abs(coeffs), quantile, axis=0)
    return Delta, Delta / n_target


def panel_sensitivity_l2(panel_profiles, W, n_target, quantile=0.99):
    """Clip L2 (pour le mécanisme gaussien) : C = q-quantile(||coef||₂), Δ₂ = C / n_target."""
    coeffs = np.asarray(panel_profiles) @ W
    C = np.quantile(np.linalg.norm(coeffs, axis=1), quantile)
    return C, C / n_target


# --------------------------------------------------------------------------- #
def build_pools(df_daily, n_basis_max=600, seed=2024):
    """
    Repartition publique disjointe base / cible (pour l'étude erreur-vs-n_target).
    Renvoie (basis_pool, target_pool) : tenseurs (n, T, 48).
    """
    ids = list(df_daily.index.get_level_values("household_id").unique())
    sub = df_daily.loc[ids]
    full = np.stack([sub.loc[h].sort_index().values for h in ids]).astype(np.float32)
    perm = np.random.default_rng(seed).permutation(full.shape[0])
    n_basis = min(n_basis_max, 3 * full.shape[0] // 5)
    return full[perm[:n_basis]], full[perm[n_basis:]]


def make_targets(target_pool, n, n_eval_groups=25, seed=1):
    """Agrégats cibles d'évaluation de taille n : (n_eval_groups, T, 48)."""
    rng = np.random.default_rng(seed)
    m = target_pool.shape[0]
    return np.stack([target_pool[rng.choice(m, size=n, replace=False)].mean(0)
                     for _ in range(n_eval_groups)])


def projection_error_vs_p(basis_pool, target_pool, n_target_grid, p_list, cfg,
                          n_eval_groups=25, method="svd"):
    """
    Erreur de projection (SVD ou NMF) vs p, par taille d'agrégat, avec IQR.
    Renvoie store[n][metric] = {'mean','q25','q75'} (sur p_list) + p_arr.
    NB : SVD recommandée (≈ NMF en projection, et un seul fit par n).
    """
    from .metrics import stats_over_axis
    p_arr = np.asarray(p_list, float)
    store = {}
    for n in n_target_grid:
        X = synthetic_aggregates(basis_pool, n, cfg.n_groups, cfg.fit_subsample, seed=cfg.seed)
        G = make_targets(target_pool, n, n_eval_groups, seed=1)
        if method == "svd":
            Q_full = fit_svd_basis(X, max(p_list))
            Ws = {int(p): Q_full[:, : int(p)] for p in p_list}
        else:
            Ws = {int(p): fit_nmf_basis(X, p, cfg.nmf_max_iter, cfg.nmf_tol) for p in p_list}
        store[n] = {m: {"mean": [], "q25": [], "q75": []} for m in
                    ("MAPE", "NMAE", "MAE", "MSE", "RMSE")}
        for p in p_list:
            W = Ws[int(p)]
            md = per_day_metrics(G, (G @ W) @ W.T)
            for m, v in md.items():
                s = stats_over_axis(v.ravel(), axis=0)
                store[n][m]["mean"].append(s["mean"])
                store[n][m]["q25"].append(s["lo"])
                store[n][m]["q75"].append(s["hi"])
        for m in store[n]:
            for k in store[n][m]:
                store[n][m][k] = np.array(store[n][m][k])
    return store, p_arr