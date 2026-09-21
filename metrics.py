"""
Convention :
    - MAPE et NMAE sont renvoyés en POURCENTAGE (×100).
    - MAE / MSE / RMSE dans l'unité (resp. unité²) de la charge.
    - per_day_metrics agit sur le dernier axe (les 48 créneaux) et préserve
      toutes les dimensions de tête : marche pour (48,), (T,48), (n_eps,T,48).
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-9
METRICS = ["MAPE", "NMAE", "MAE", "MSE", "RMSE"]
METRIC_LABEL = {
    "MAPE": "MAPE (%)", "NMAE": "NMAE (%)", "MAE": "MAE (unité de charge)",
    "MSE": "MSE (unité²)", "RMSE": "RMSE (unité de charge)", "AEcorr": "AEcorr",
}


def per_day_metrics(L_true, L_pred) -> dict:
    """(..., 48) → dict de tableaux (...,). MAPE/NMAE en %, MAE/MSE/RMSE en unité."""
    L_true = np.asarray(L_true, float)
    L_pred = np.asarray(L_pred, float)
    diff = L_pred - L_true
    ad = np.abs(diff)
    mae = ad.mean(-1)
    mse = (diff ** 2).mean(-1)
    rmse = np.sqrt(mse)
    mape = (ad / np.maximum(np.abs(L_true), _EPS)).mean(-1) * 100
    nmae = mae / np.maximum(np.abs(L_true).mean(-1), _EPS) * 100
    return {"MAPE": mape, "NMAE": nmae, "MAE": mae, "MSE": mse, "RMSE": rmse}


#  scalaires de commodité (un seul appel) 
def mape_pct(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 100 * np.mean(np.abs(a - b) / np.maximum(np.abs(a), _EPS))


def nmae_pct(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 100 * np.mean(np.abs(a - b)) / max(np.mean(np.abs(a)), _EPS)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


#  autocorrélation (métrique AEcorr) 
def acf_norm(x, max_lag=12):
    """ACF normalisée (acf[0]=1) d'un signal 1D, jusqu'à max_lag."""
    x = np.asarray(x, float)
    x = x - x.mean()
    r = np.correlate(x, x, mode="full")[len(x) - 1:]
    return r[: max_lag + 1] / r[0] if r[0] > 0 else np.zeros(max_lag + 1)


def aecorr_per_day(L_true, L_pred, max_lag=12):
    """Écart absolu moyen entre ACF réelle et ACF reconstruite, jour par jour. (T,)→(T,)."""
    L_true = np.atleast_2d(L_true); L_pred = np.atleast_2d(L_pred)
    return np.array([
        np.abs(acf_norm(L_true[t], max_lag) - acf_norm(L_pred[t], max_lag)).mean()
        for t in range(L_true.shape[0])
    ])


def stats_over_axis(values, axis=-1, q_lo=0.25, q_hi=0.75):
    """mean + bande de quantiles (q_lo, q_hi) le long d'un axe.
    Défaut (0.25, 0.75) = bande IQR. Clés renvoyées : 'mean', 'lo', 'hi'."""
    return dict(
        mean=values.mean(axis=axis),
        lo=np.quantile(values, q_lo, axis=axis),
        hi=np.quantile(values, q_hi, axis=axis),
    )


#  batch (n_eps, T, 48) pour les figures utility/privacy 
def compute_wpa_metrics(L_noisy_all, L_true, max_lag=12) -> dict:
    """
    Métriques par (eps, day) pour un tenseur (n_eps, T, 48) vs L_true (T, 48).
    Renvoie un dict {metric: (n_eps, T)} incluant AEcorr.
    """
    L_noisy_all = np.asarray(L_noisy_all, float)
    L_true = np.asarray(L_true, float)
    n_eps, T, _ = L_noisy_all.shape
    md = per_day_metrics(L_true[None, :, :], L_noisy_all)      # broadcast (n_eps,T,48)
    acf_true = np.array([acf_norm(L_true[t], max_lag) for t in range(T)])
    aec = np.zeros((n_eps, T))
    for i in range(n_eps):
        for t in range(T):
            aec[i, t] = np.abs(acf_true[t] - acf_norm(L_noisy_all[i, t], max_lag)).mean()
    md["AEcorr"] = aec
    return md