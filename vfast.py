"""
FAST vectoriel (VFAST) : filtre de Kalman vectoriel sur les coefficients
d'ondelette + contrôleur PID d'échantillonnage adaptatif. Deux mécanismes de
publication, sélectionnés par `FastConfig.mechanism` :

  - 'laplace'  : bruit de Laplace par coordonnée, budget ε/p puis réparti sur les
                 M publications (composition séquentielle) ; R = 2·diag(scale²).
                 `Delta_mean` = sensibilité L1 par coordonnée.
  - 'gaussian' : bruit gaussien isotrope N(0, σ²·I_p), σ calibré par comptabilité
                 RDP sur les M publications (gain en √M) ; R = σ²·I_p.
                 Requiert `Delta2` (sensibilité L2 scalaire).

Inclut le tuning du scalaire q (Q = q·I_p) et les balayages de sensibilité.

Réfs : Fan & Xiong (FAST), Kalman (1960), Mironov (2017, RDP), de Vilmarest (2022, ch. 7).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class FastConfig:
    """Hyperparamètres FAST vectoriel."""
    M_ratio: float = 0.15
    initial_interval: int = 1
    Q_scalar: float = 1e-3        # utilisé si Q_matrix=None : Q = Q_scalar · I_p
    R_mode: str = "theoretical"   # 'theoretical' = 2·diag(scale_lap²) (Laplace) ou σ²·I (gauss)
    P0: float = 1e6               # priore diffuse
    Cp: float = 0.9
    Ci: float = 0.1
    Cd: float = 0.0
    Ti: int = 5
    theta: float = 10.0
    xi: float = 0.1
    delta_norm: float = 1e-3
    mechanism: str = "laplace"    # 'laplace' | 'gaussian'
    delta: float = 1e-5           # δ du gaussien (ignoré pour Laplace)
    seed: int = 42


class VectorialKalmanFAST:
    """Kalman vectoriel : x_k = x_{k-1}+ω (Q),  z_k = x_k+ν (R). Forme de Joseph."""

    def __init__(self, Q, R, x0, P0):
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.x_post = np.asarray(x0, float).copy()
        self.P_post = np.asarray(P0, float).copy()
        self.p = self.x_post.size

    def predict(self):
        return self.x_post.copy(), self.P_post + self.Q

    def correct(self, z, x_prior, P_prior):
        S = P_prior + self.R
        K = np.linalg.solve(S.T, P_prior.T).T
        x_post = x_prior + K @ (z - x_prior)
        I = np.eye(self.p)
        IKH = I - K
        P_post = IKH @ P_prior @ IKH.T + K @ self.R @ K.T
        self.x_post, self.P_post = x_post, P_post
        return x_post, P_post, K


def _release_spec(eps_total, p, M, sensitivity, config, Delta2):
    """Construit (fonction de tirage du bruit, matrice R, info diag) selon le mécanisme."""
    mech = getattr(config, "mechanism", "laplace")
    if mech == "laplace":
        eps_j = np.full(p, eps_total / p)
        scale = sensitivity * M / eps_j
        R = (np.diag(2 * scale ** 2) if config.R_mode == "theoretical"
             else np.asarray(config.R_mode, float))
        return (lambda rng: rng.laplace(0.0, scale)), R, dict(scale_lap=scale, eps_j=eps_j)
    elif mech == "gaussian":
        if Delta2 is None:
            raise ValueError("mechanism='gaussian' requiert Delta2 (sensibilité L2).")
        from .mechanisms import sigma_for_total
        sigma = sigma_for_total(eps_total, config.delta, Delta2, M)[0]
        R = np.eye(p) * sigma ** 2
        return (lambda rng: rng.normal(0.0, sigma, p)), R, dict(sigma=float(sigma), Delta2=float(Delta2))
    elif mech == "gaussian_aniso":
        # sensitivity = Delta_mean = vecteur Δ_j (L1 par coordonnée), déjà disponible
        from .mechanisms import sigma_anisotropic
        sigma_vec = sigma_anisotropic(eps_total, config.delta, sensitivity,
                                       mode="sequential", T=M)
        R = np.diag(sigma_vec ** 2)
        sv = sigma_vec.copy()          # capture explicite pour le lambda
        return (lambda rng, sv=sv: rng.standard_normal(p) * sv), R, \
               dict(sigma_vec=sv, var_total=float((sv ** 2).sum()))
    raise ValueError(mech)


def fast_vectorial_pid(alpha_arr, eps_total, sensitivity, config, Q_matrix=None, Delta2=None):
    """Kalman vectoriel + PID d'échantillonnage adaptatif. Renvoie (released (T,p), diag)."""
    rng = np.random.default_rng(config.seed)
    T, p = alpha_arr.shape
    M = max(1, int(np.ceil(config.M_ratio * T)))
    draw, R, noise_info = _release_spec(eps_total, p, M, sensitivity, config, Delta2)
    Q = np.asarray(Q_matrix, float) if Q_matrix is not None else np.eye(p) * config.Q_scalar

    z0 = alpha_arr[0] + draw(rng)
    kf = VectorialKalmanFAST(Q=Q, R=R, x0=z0, P0=config.P0 * np.eye(p))
    xp0, Pp0 = kf.predict()
    kf.correct(z0, xp0, Pp0)

    released = np.zeros_like(alpha_arr)
    sampled = np.zeros(T, dtype=bool)
    P_trace = np.zeros(T)
    fb = []
    released[0], sampled[0], P_trace[0] = kf.x_post, True, np.trace(kf.P_post)
    current_interval = max(1, config.initial_interval)
    next_sample, n_samples = current_interval, 1

    for k in range(1, T):
        x_prior, P_prior = kf.predict()
        if k == next_sample and n_samples < M:
            z = alpha_arr[k] + draw(rng)
            x_post, P_post, _ = kf.correct(z, x_prior, P_prior)
            released[k], sampled[k], P_trace[k] = x_post, True, np.trace(P_post)
            n_samples += 1
            err = np.linalg.norm(x_post - x_prior) / max(np.linalg.norm(x_post), config.delta_norm)
            fb.append(err)
            Dp = config.Cp * err
            Di = (config.Ci / config.Ti) * sum(fb[-min(config.Ti, len(fb)):]) if config.Ci > 0 else 0.0
            Dd = config.Cd * (fb[-1] - fb[-2]) if (config.Cd > 0 and len(fb) >= 2) else 0.0
            exp_arg = (Dp + Di + Dd - config.xi) / config.xi
            new_int = max(1, int(np.round(current_interval + config.theta * (1 - np.exp(min(exp_arg, 50))))))
            current_interval, next_sample = new_int, k + new_int
        else:
            released[k], P_trace[k] = x_prior, np.trace(P_prior)

    return released, dict(sampled_mask=sampled, P_trace=P_trace,
                          feedback_errors=np.array(fb), n_samples=int(n_samples),
                          M=M, mechanism=getattr(config, "mechanism", "laplace"),
                          R=R, Q=Q, **noise_info)


def fast_vectorial(alpha_arr, W, Delta_mean, eps_total, config, Q_matrix=None, Delta2=None):
    """FAST vectoriel complet : renvoie (L̃ = α̃·Wᵀ, α̃ (coefs DP), diag).
    Pour mechanism='gaussian', fournir Delta2 (sensibilité L2)."""
    alpha_tilde, diag = fast_vectorial_pid(alpha_arr, eps_total, Delta_mean, config,
                                           Q_matrix=Q_matrix, Delta2=Delta2)
    return alpha_tilde @ W.T, alpha_tilde, diag


def lpa_coefs_baseline(alpha_arr, W, Delta_mean, eps_total, seed=42):
    """Baseline WPA séquentielle "naïve" sur (jour, coef) : ε réparti sur T·p. Renvoie (L̃, α̃)."""
    rng = np.random.default_rng(seed)
    T, p = alpha_arr.shape
    scales = Delta_mean / (eps_total / (T * p))
    noisy = alpha_arr + rng.laplace(0.0, scales[None, :], size=(T, p))
    return noisy @ W.T, noisy


# ============================ tuning de q ================================== #
def grid_search_q(eval_aggregates, W, Delta_mean, eps, config, q_grid,
                  n_seed=3, seed0=1000, Delta2=None):
    """
    Balaie q (Q = q·I_p) et évalue VFAST sur des agrégats PUBLICS (DP-safe).
    `eval_aggregates` : liste de L_g (T,48) ; on projette G@W en interne.
    Le mécanisme suit `config.mechanism` ('laplace' ou 'gaussian' → fournir Delta2).
    Renvoie store_q[metric] = {'mean','q25','q75'} (sur q_grid) — MAPE/NMAE/MAE/RMSE.

    NB : à tuner au RÉGIME DE DÉPLOIEMENT (même p, même taille d'agrégat, ε réaliste) —
    q* ne se transfère pas d'un (p, n) à un autre.
    """
    from .metrics import per_day_metrics
    METR = ["MAPE", "NMAE", "MAE", "RMSE"]
    eval_a = [Lg @ W for Lg in eval_aggregates]
    store = {m: {"mean": [], "q25": [], "q75": []} for m in METR}
    for q in q_grid:
        Q = q * np.eye(W.shape[1])
        vals = {m: [] for m in METR}
        for Lg, ag in zip(eval_aggregates, eval_a):
            for s in range(n_seed):
                L_vf, _, _ = fast_vectorial(ag, W, Delta_mean, eps, replace(config, seed=seed0 + s),
                                            Q_matrix=Q, Delta2=Delta2)
                md = per_day_metrics(Lg, L_vf)
                for m in METR:
                    vals[m].append(md[m])
        for m in METR:
            v = np.concatenate(vals[m])
            store[m]["mean"].append(v.mean())
            store[m]["q25"].append(np.quantile(v, 0.25))
            store[m]["q75"].append(np.quantile(v, 0.75))
    for m in METR:
        for k in store[m]:
            store[m][k] = np.array(store[m][k])
    q_best = {m: float(np.asarray(q_grid)[int(np.argmin(store[m]["mean"]))]) for m in METR}
    return store, q_best


# ===================== sensibilité aux hyperparamètres ===================== #
def sensitivity_1d(param_name, param_values, alpha_arr, W, Delta_mean, Q_matrix,
                   base_config, L_true, eps=10.0, metric="RMSE", Delta2=None):
    """Fait varier UN hyperparamètre de FastConfig ; renvoie une liste de dicts
    {param, error (metric vs L_true), n_samples, M}."""
    from .metrics import per_day_metrics
    L_true = np.asarray(L_true, float)
    rows = []
    for v in param_values:
        c = replace(base_config, **{param_name: float(v)})
        L_tilde, _, diag = fast_vectorial(alpha_arr, W, Delta_mean, eps, c,
                                          Q_matrix=Q_matrix, Delta2=Delta2)
        rows.append({param_name: float(v),
                     "error": float(per_day_metrics(L_true, L_tilde)[metric].mean()),
                     "n_samples": int(diag["n_samples"]), "M": int(diag["M"])})
    return rows