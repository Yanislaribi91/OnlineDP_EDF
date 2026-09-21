"""
Mécanismes DP "non adaptatifs" :
  - LPA point-par-point sur l'agrégat brut (NB 2),
  - WPA sur les coefficients d'ondelette (Laplace), budget ε/jour ou ε total (NB 4),
  - mécanisme gaussien isotrope sur les coefficients + calibration RDP (Mironov).

Le filtre adaptatif VFAST est dans online_dp.vfast.
"""
from __future__ import annotations
import numpy as np


#  LPA point-par-point (NB 2) 
def lpa_pointwise(L_true, eps, sensitivity, mode="per_day", T_compose=None, seed=42):
    """
    Laplace naïf sur chaque (jour, créneau) de l'agrégat brut L_true (T, 48).
    Budget par créneau = ε / 48 (composition séquentielle sur les 48 créneaux).

    mode='per_day'    : chaque jour reçoit ε        → scale = sensitivity·48/ε
    mode='sequential' : ε réparti sur T jours       → scale = sensitivity·48·T/ε
    Renvoie L_noisy (T, 48).
    """
    L_true = np.asarray(L_true, float)
    T, S = L_true.shape
    rng = np.random.default_rng(seed)
    if mode == "per_day":
        scale = sensitivity * S / eps
    elif mode == "sequential":
        Tc = T if T_compose is None else T_compose
        scale = sensitivity * S * Tc / eps
    else:
        raise ValueError(mode)
    return L_true + rng.laplace(0.0, scale, size=(T, S))


def lpa_pointwise_grid(L_true, eps_grid, sensitivity, mode="per_day", seed=42):
    """Tenseur (n_eps, T, 48) des reconstructions LPA pour une grille d'ε."""
    return np.stack([lpa_pointwise(L_true, float(e), sensitivity, mode, seed=seed)
                     for e in eps_grid])

   
def gaussian_pointwise(L_true, eps, sensitivity, delta, mode="per_day",
                       T_compose=None, seed=42):
    """
    mode='per_day'    : chaque jour reçoit (ε, δ)          → σ = sigma_one_release
    mode='sequential' : (ε, δ) réparti sur T jours (RDP)   → σ = sigma_for_total
    Renvoie L_noisy (T, 48).
    """
    L_true = np.asarray(L_true, float)
    T, S = L_true.shape
    rng = np.random.default_rng(seed)
    Delta2 = sensitivity * np.sqrt(S)
    if mode == "per_day":
        sigma, _ = sigma_one_release(eps, delta, Delta2)
    elif mode == "sequential":
        Tc = T if T_compose is None else T_compose
        sigma, _ = sigma_for_total(eps, delta, Delta2, Tc)
    else:
        raise ValueError(mode)
    return L_true + rng.normal(0.0, sigma, size=(T, S))


def gaussian_pointwise_grid(L_true, eps_grid, sensitivity, delta,
                            mode="per_day", seed=42):
    """Tenseur (n_eps, T, 48) des reconstructions gaussiennes pour une grille d'ε."""
    return np.stack([gaussian_pointwise(L_true, float(e), sensitivity, delta, mode, seed=seed)
                     for e in eps_grid])

#  WPA coefficients — Laplace (NB 4)
def clip_aggregate_coeffs(df_daily_target, W, Delta):
    """
    Coefficients de la cible, clippés par coordonnée à ±Delta puis agrégés par jour.
    Renvoie alpha_clip (DataFrame T×p) — l'objet réellement perturbé par la DP.
    """
    import pandas as pd
    C = np.clip(df_daily_target.values @ W, -Delta[None, :], Delta[None, :])
    out = (pd.DataFrame(C, index=df_daily_target.index)
           .groupby(level="date").mean().sort_index())
    out.index = pd.to_datetime(out.index)
    return out


def wpa_laplace(alpha_clip, W, Delta_mean, eps, mode="per_day", T_compose=None, seed=42):
    """
    WPA : Laplace par coordonnée sur les coefficients agrégés, allocation uniforme ε_j=ε/p.
    mode='per_day'    : scale_j = Delta_mean_j · p / ε        (budget ε CHAQUE jour, total Tε)
    mode='sequential' : scale_j = Delta_mean_j · p · T / ε    (budget total ε réparti sur T)
    Renvoie L_tilde (T, 48).
    """
    alpha = np.asarray(alpha_clip.values if hasattr(alpha_clip, "values") else alpha_clip, float)
    T, p = alpha.shape
    rng = np.random.default_rng(seed)
    if mode == "per_day":
        scales = Delta_mean * p / eps
    elif mode == "sequential":
        Tc = T if T_compose is None else T_compose
        scales = Delta_mean * p * Tc / eps
    else:
        raise ValueError(mode)
    return (alpha + rng.laplace(0.0, scales[None, :], size=(T, p))) @ W.T


def wpa_laplace_grid(alpha_clip, W, Delta_mean, eps_grid, mode="per_day", seed=42):
    return np.stack([wpa_laplace(alpha_clip, W, Delta_mean, float(e), mode, seed=seed)
                     for e in eps_grid])


#  Mécanisme gaussien + RDP (Mironov) 
def sigma_one_release(eps, delta, Delta2):
    """σ du gaussien isotrope pour (ε, δ)-DP d'UNE publication (conversion RDP). Renvoie (σ, z)."""
    L = np.log(1.0 / delta)
    u = -np.sqrt(2 * L) + np.sqrt(2 * L + 2 * eps)        # u = 1/z
    return Delta2 / u, 1.0 / u


def sigma_for_total(eps_total, delta, Delta2, T):
    """
    σ du gaussien isotrope pour un budget TOTAL (ε_total, δ) sur T publications
    (composition séquentielle, comptabilité RDP → gain en √T). Renvoie (σ, z).
    """
    L = np.log(1.0 / delta)
    u = (-np.sqrt(2 * T * L) + np.sqrt(2 * T * (L + eps_total))) / T
    return Delta2 / u, 1.0 / u


def rdp_eps(alpha_order, z):
    """Courbe RDP du gaussien : ε_RDP(α) = α / (2 z²)."""
    return alpha_order / (2.0 * z ** 2)


def wpa_gaussian(alpha_clip, W, Delta2, sigma, seed=42):
    """
    WPA gaussien isotrope : α̃ = α + N(0, σ² I_p), reconstruction L̃ = α̃ Wᵀ.
    Pas de split par coordonnée (la sensibilité L2 absorbe le vecteur entier).
    """
    alpha = np.asarray(alpha_clip.values if hasattr(alpha_clip, "values") else alpha_clip, float)
    T, p = alpha.shape
    rng = np.random.default_rng(seed)
    return (alpha + rng.normal(0.0, sigma, size=(T, p))) @ W.T


def wpa_gaussian_grid(alpha_clip, W, Delta2, eps_grid, delta, mode="per_day",
                      T_compose=None, seed=42):
    """
    Grille (n_eps, T, 48) du WPA gaussien.
    mode='per_day'    : chaque jour calibré (ε, δ) indépendamment   → σ via sigma_one_release.
    mode='sequential' : budget TOTAL (ε, δ) sur T jours (RDP, √T)   → σ via sigma_for_total.
    """
    T = (alpha_clip.values if hasattr(alpha_clip, "values") else alpha_clip).shape[0]
    Tc = T if T_compose is None else T_compose
    out = []
    for e in eps_grid:
        if mode == "per_day":
            sig = sigma_one_release(float(e), delta, Delta2)[0]
        elif mode == "sequential":
            sig = sigma_for_total(float(e), delta, Delta2, Tc)[0]
        else:
            raise ValueError(mode)
        out.append(wpa_gaussian(alpha_clip, W, Delta2, sig, seed=seed))
    return np.stack(out)

# ── Mécanisme gaussien ANISOTROPE (calibration optimale par coordonnée) ─── #

def sigma_anisotropic(eps, delta, Delta_coord, mode="per_day", T=None):
    """
    Calibrage optimal des σ_j pour le gaussien anisotrope à budget (ε,δ)-DP fixé.

    Résout  min Σ_j σ_j²  s.c.  ρ = ½ Σ_j Δ_j²/σ_j²  par Lagrange :
        σ_j² = Δ_j · (Σ_k Δ_k) / (2ρ),   σ_j ∝ √Δ_j.
    Budget zCDP ρ = u²/2, u issu de la calibration RDP (Mironov).

    Gain vs isotrope (Cauchy-Schwarz) :
        Σ σ_j² = (Σ Δ_j)² / (2ρ)  ≤  p · Δ₂² / (2ρ),
    avec égalité ssi tous les Δ_j sont égaux (énergie non concentrée).

    mode='per_day'    : UNE publication, budget ε par jour.
    mode='sequential' : budget TOTAL ε sur T publications (gain √T, RDP).
    Renvoie sigma_vec (p,).
    """
    Delta_coord = np.asarray(Delta_coord, float)
    S = Delta_coord.sum()
    L = np.log(1.0 / delta)
    if mode == "per_day":
        u = -np.sqrt(2 * L) + np.sqrt(2 * L + 2 * eps)
    elif mode == "sequential":
        if T is None:
            raise ValueError("mode='sequential' requiert T (nombre de publications).")
        u = (-np.sqrt(2 * T * L) + np.sqrt(2 * T * (L + eps))) / T
    else:
        raise ValueError(mode)
    # σ_j = √(Δ_j · S) / u  ←  σ_j² = Δ_j · S / u²  = Δ_j · S / (2ρ)
    return np.sqrt(Delta_coord * S) / u


def wpa_gaussian_anisotropic(alpha_clip, W, Delta_coord, eps, delta,
                              mode="per_day", T_compose=None, seed=42):
    """
    WPA gaussien anisotrope : α̃_t = α_t + N(0, diag(σ_1²,…,σ_p²)),
    σ_j calibré par Lagrange sur Δ_j (sensibilité L1 par coordonnée du panel).

    Avantage vs isotrope : variance totale = (Σ Δ_j)²/(2ρ) ≤ p·Δ₂²/(2ρ).
    Gain maximal quand l'énergie SVD est concentrée (Δ_1 ≫ Δ_2,…,Δ_p).

    mode='per_day'    : σ calibré pour UNE publication (budget ε par jour).
    mode='sequential' : budget TOTAL ε réparti sur T jours (gain √T, RDP).
    Renvoie L_tilde (T, 48).
    """
    alpha = np.asarray(alpha_clip.values if hasattr(alpha_clip, "values") else alpha_clip, float)
    T, p = alpha.shape
    Tc = T if T_compose is None else int(T_compose)
    sigma_vec = sigma_anisotropic(eps, delta, Delta_coord,
                                   mode=mode,
                                   T=Tc if mode == "sequential" else None)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=(T, p)) * sigma_vec[np.newaxis, :]
    return (alpha + noise) @ W.T


def wpa_gaussian_anisotropic_grid(alpha_clip, W, Delta_coord, eps_grid, delta,
                                   mode="per_day", T_compose=None, seed=42):
    """
    Grille (n_eps, T, 48) du WPA gaussien anisotrope.
    Symétrique de wpa_gaussian_grid — remplace Delta2 scalaire par Delta_coord (p,).
    """
    alpha = np.asarray(alpha_clip.values if hasattr(alpha_clip, "values") else alpha_clip, float)
    Tc = alpha.shape[0] if T_compose is None else int(T_compose)
    return np.stack([
        wpa_gaussian_anisotropic(alpha_clip, W, Delta_coord, float(e), delta,
                                  mode=mode, T_compose=Tc, seed=seed)
        for e in eps_grid
    ])