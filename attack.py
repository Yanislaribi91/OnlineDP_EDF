"""
online_dp.attack
=============
Attaques adversariales de RECONSTRUCTION D'AGRÉGAT (NB 5).

Cadre (différent des MIA sur modèles ML) : l'adversaire observe l'agrégat d'une
cible et connaît l'ensemble des candidats D = D_in ∪ D_out (les courbes
individuelles, en clair) ainsi que N = |D_in|, mais pas D_in. Il cherche

    P* ∈ argmin_{P ⊆ D, |P| = N}  dist( (1/N) Σ_{i∈P} ℓ_i , cible ),

de cardinalité C(|D|, N) — inexplorable exhaustivement. On résout l'optimum
EXACT par MILP (subset-sum vectoriel / moindres carrés L1 sous contrainte de
cardinalité). L'approche gloutonne par 2-échange a été abandonnée (ne converge
pas à grand |D|, et converge sinon vers de mauvais optima locaux).

Deux régimes :
  - BRUT (R^48)        : courbes sans projection ni bruit  → exact_attack_l1.
  - PROJETÉ (R^p)      : agrégats projetés sur span(W) (W ondelettes orthonormée,
                         publique car calculable sur panel public), avec option
                         d'injecter un bruit DP gaussien sur les coefficients de
                         l'agrégat cible → run_attack(mode='clear'|'noise').
    W orthonormée ⇒ attaquer en coefficients α=L·W ∈ R^p équivaut à attaquer la
    reconstruction L̂=α·Wᵀ ∈ R^48 (isométrie), mais p ≤ 48 → MILP plus léger, et
    c'est l'espace où agit la DP. Les candidats ne sont JAMAIS bruités (l'adversaire
    les possède en clair) ; seul l'agrégat publié de la cible l'est, en mode 'noise'.

Contenu :
    build_attack_instance   construit (D, D_in, L_real) depuis df_daily
    exact_attack_l1         optimum exact L1 (espace brut R^48) via MILP
    milp_recovery_curve     % retrouvés vs ratio card(D_in)/card(D) (brut), avec timing
    load_matched_W          extrait W (48,p) de la base mise en cache (build_matched_basis)
    project_instance        projette candidats + cible sur span(W)
    noisy_target_coeffs     clip L2 + bruit gaussien DP sur les coefs de la cible
    exact_attack_proj_l1    optimum exact L1 dans l'espace des coefficients R^p
    run_attack              attaque projetée clear/noise (point d'entrée unique)
    proj_recovery_curve     % retrouvés vs ratio (projeté), clear/noise, avec timing
    overlap / advantage     métriques d'évaluation (recouvrement avec D_in)
    mutual_coherence        cohérence mutuelle de D (contexte d'identifiabilité)

Toutes les fonctions sont pures (D, L_real, W, σ, etc. passés en argument).
"""
from __future__ import annotations
import time
import numpy as np


# ===================== construction de l'instance ========================== #
def build_attack_instance(df_daily, n_in=50, n_decoy=50, mode="mean", day=None, seed=7):
    """
    Tire n_in contributeurs (D_in) + n_decoy leurres (D_out) parmi les foyers de
    df_daily et construit l'instance d'attaque.

    mode='mean' : chaque candidat = profil journalier MOYEN du foyer (sur tous ses
                  jours) → vecteur R^48 stable et distinctif.
    mode='day'  : chaque candidat = profil du foyer un jour donné (date commune ;
                  `day` = index dans les dates communes, défaut = jour médian).

    Renvoie un dict : D (|D|,48), ids, in_idx (indices de D_in dans D), in_ids,
    out_ids, L_real (48,) = moyenne de D_in, N, mode.
    """
    rng = np.random.default_rng(seed)
    users = np.asarray(df_daily.index.get_level_values("household_id").unique())
    assert len(users) >= n_in + n_decoy, (
        f"{len(users)} foyers dispo, {n_in + n_decoy} requis — augmente cfg.N."
    )
    chosen = rng.choice(users, size=n_in + n_decoy, replace=False)
    in_ids, out_ids = chosen[:n_in], chosen[n_in:]

    if mode == "mean":
        D = np.stack([df_daily.loc[h].mean(axis=0).values for h in chosen]).astype(float)
    elif mode == "day":
        common = sorted(set.intersection(*[set(df_daily.loc[h].index) for h in chosen]))
        assert common, "aucune date commune aux foyers tirés"
        d0 = common[len(common) // 2] if day is None else common[int(day)]
        D = np.stack([df_daily.loc[h].loc[d0].values for h in chosen]).astype(float)
    else:
        raise ValueError(mode)

    in_idx = np.arange(n_in)                     # D_in = les n_in premières lignes
    L_real = D[in_idx].mean(0)
    return dict(D=D, ids=list(chosen), in_idx=in_idx, in_ids=in_ids, out_ids=out_ids,
                L_real=L_real, N=int(n_in), mode=mode)


# ===================== attaque exacte brute (MILP L1, R^48) ================ #
def exact_attack_l1(D, L_real, N):
    """
    Optimum EXACT pour la distance L1 (Σ_t |·|) dans l'espace brut R^48, via MILP
    (scipy.optimize.milp). Renvoie (P_idx, dist_L1). Comme D_in atteint dist 0, le
    solveur s'arrête dès qu'il trouve un sous-ensemble de coût nul (borne inf).
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    D = np.asarray(D, float); L_real = np.asarray(L_real, float)
    M, S = D.shape
    c = np.concatenate([np.zeros(M), np.ones(S)])           # objectif Σ u_t
    A_up = np.hstack([D.T / N, -np.eye(S)])                 # (1/N)Dᵀx − u ≤ L_real
    A_lo = np.hstack([-D.T / N, -np.eye(S)])                # −(1/N)Dᵀx − u ≤ −L_real
    cons = [
        LinearConstraint(A_up, -np.inf, L_real),
        LinearConstraint(A_lo, -np.inf, -L_real),
        LinearConstraint(np.concatenate([np.ones(M), np.zeros(S)])[None, :], N, N),
    ]
    integ = np.concatenate([np.ones(M), np.zeros(S)])
    bounds = Bounds(np.concatenate([np.zeros(M), np.zeros(S)]),
                    np.concatenate([np.ones(M), np.full(S, np.inf)]))
    res = milp(c=c, constraints=cons, integrality=integ, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"MILP non résolu : {res.message}")
    x = res.x[:M]
    return np.where(x > 0.5)[0], float(res.fun)


def milp_recovery_curve(df_daily, Dsize, ratios, n_seeds=3, base_seed=2024,
                        mode="mean", progress=None):
    """
    Attaque EXACTE brute (MILP L1) : pour une taille de pool |D| = Dsize, balaie le
    ratio card(D_in)/card(D) (ratios ∈ ]0,1[) et mesure le % de foyers retrouvés
    |P*∩D_in|/N, moyenné sur n_seeds instances (+ IQR), en chronométrant chaque MILP.

    progress : callable optionnel appelé APRÈS chaque MILP avec un dict
        {Dsize, ratio, seed, n_in, dt, recovery} — pour brancher une barre tqdm.

    Renvoie dict {'ratios','mean','q25','q75','Dsize','t_mean'} (% pour mean/q25/q75,
    secondes pour t_mean = temps moyen de résolution par ratio).
    """
    ratios = np.asarray(ratios, float)
    mean = np.zeros_like(ratios); q25 = np.zeros_like(ratios)
    q75 = np.zeros_like(ratios); tmean = np.zeros_like(ratios)
    for r, ratio in enumerate(ratios):
        n_in = min(max(int(round(ratio * Dsize)), 1), Dsize - 1)
        fr = np.empty(n_seeds); dts = np.empty(n_seeds)
        for s in range(n_seeds):
            inst = build_attack_instance(df_daily, n_in=n_in, n_decoy=Dsize - n_in,
                                         mode=mode, seed=base_seed + 1000 * r + s)
            t0 = time.perf_counter()
            P, _ = exact_attack_l1(inst["D"], inst["L_real"], inst["N"])
            dt = time.perf_counter() - t0
            rec = overlap(P, inst["in_idx"])[1] * 100.0
            fr[s] = rec; dts[s] = dt
            if progress is not None:
                progress(dict(Dsize=int(Dsize), ratio=float(ratio), seed=int(s),
                              n_in=int(n_in), dt=float(dt), recovery=float(rec)))
        mean[r] = fr.mean(); q25[r] = np.quantile(fr, 0.25)
        q75[r] = np.quantile(fr, 0.75); tmean[r] = dts.mean()
    return dict(ratios=ratios, mean=mean, q25=q25, q75=q75, Dsize=int(Dsize), t_mean=tmean)


# ===================== extraction de la base W ============================ #
def load_matched_W(bases, method="svd", p=None):
    """
    Extrait une base orthonormée W (48, p) de l'objet renvoyé par
    basis.build_matched_basis : {'svd': {p: W}, 'nmf': {p: W}}.

    method : 'svd' (recommandé) ou 'nmf'.
    p      : rang voulu ; défaut = rang max disponible pour la méthode.
    Tolère aussi qu'on passe directement un W (ndarray 48×p) ou un {p: W}.
    """
    if isinstance(bases, np.ndarray):
        return bases
    sub = bases[method] if (isinstance(bases, dict) and method in bases) else bases
    if isinstance(sub, np.ndarray):
        return sub
    p = max(sub) if p is None else int(p)
    if p not in sub:
        raise KeyError(f"rang p={p} absent ; rangs dispo : {sorted(sub)}")
    return np.asarray(sub[p], float)


# ===================== projection de l'instance =========================== #
def project_instance(inst, W):
    """
    Projette une instance d'attaque (issue de build_attack_instance) sur span(W).

    Renvoie un dict enrichi : A (|D|, p) = coefficients des candidats (a_i = ℓ_i·W),
    alpha_real (p,) = coefficients de l'agrégat réel (= moyenne des a_i de D_in),
    W, p ; et recopie in_idx, N, ids.
    """
    W = np.asarray(W, float)
    A = np.asarray(inst["D"], float) @ W                 # (|D|, p)
    alpha_real = np.asarray(inst["L_real"], float) @ W   # (p,)
    out = dict(inst)
    out.update(A=A, alpha_real=alpha_real, W=W, p=int(W.shape[1]))
    return out


# ===================== bruit DP sur l'agrégat reél ======================= #
def noisy_target_coeffs(alpha_real, sigma, Delta2=None, clip_C=None, seed=0):
    """
    Agrégat cible publié sous DP gaussienne, DANS l'espace des coefficients.

    Optionnellement clippe la norme L2 du vecteur de coefficients à clip_C (cohérent
    avec la calibration L2 : Δ₂ = clip_C / N), puis ajoute N(0, σ² I_p).
    Renvoie alpha_tilde (p,).
    NB : σ doit être calibré sur la sensibilité de l'AGRÉGAT (Δ₂ = clip_C / N),
    cf. mechanisms.sigma_one_release(eps, delta, Delta2).
    """
    a = np.asarray(alpha_real, float).copy()
    if clip_C is not None:
        nrm = np.linalg.norm(a)
        if nrm > clip_C:
            a *= clip_C / nrm
    rng = np.random.default_rng(seed)
    return a + rng.normal(0.0, sigma, size=a.shape)


# ===================== attaque exacte projetée (MILP L1, CP-SAT) ========== #
def exact_attack_proj_l1(A, target, N, space="coef", W=None,
                         time_limit=30.0, mip_gap=None, scale=10000, workers=8):
    """
    Attaque L1 projetée résolue par OR-Tools CP-SAT (bien plus rapide que scipy.milp
    sur ce problème de cardinalité Σx=N).

    space='coef'   : compare dans l'espace des coefficients R^p (2p contraintes)
        min_{x∈{0,1}^M, Σx=N}  ‖ (1/N) Aᵀ x − target ‖₁ .
    space='signal' : compare dans l'espace signal R^48 (96 contraintes, unités de charge)
        agrège P, reconstruit via H=W·Wᵀ, compare à la cible reconstruite.

    Détails solveur :
      - L'objectif L1 est mis à l'échelle en ENTIERS (×scale, et l'égalité est
        multipliée par N pour éliminer la division) — CP-SAT est un solveur entier.
      - En 'clear', l'optimum vaut 0 (D_in l'atteint). CP-SAT trouve une solution de
        coût nul rapidement ; un callback arrête la recherche dès qu'une solution de
        coût 0 est rencontrée (inutile de certifier l'optimalité).
      - En 'noise', le solveur renvoie la meilleure solution dans `time_limit`.
    time_limit : OBLIGATOIRE (s). mip_gap : tolérance de gap relatif (sinon best-so-far).
    scale : facteur d'entiérisation des coûts (10^4 par défaut). workers : threads.

    Renvoie (P_idx, dist_L1, info) où dist_L1 est en unité de l'espace choisi et
    info = {status, optimal(bool), obj_scaled, bound_scaled}.
    """
    from ortools.sat.python import cp_model
    if time_limit is None:
        raise ValueError("time_limit est obligatoire (ex. 30.0).")
    A = np.asarray(A, float)
    target = np.asarray(target, float)
    M, p = A.shape

    # matrice de reconstruction selon l'espace : on compare ‖ B x / N − rhs ‖₁
    if space == "coef":
        B = A.T                                       # (p, M) : Σ a_i
        rhs = target                                  # (p,)
    elif space == "signal":
        if W is None:
            raise ValueError("space='signal' requiert W (48, p).")
        W = np.asarray(W, float)
        B = W @ A.T                                   # (48, M) : reconstruction Σ ℓ̂_i
        rhs = target if target.shape[0] == W.shape[0] else target @ W.T
    else:
        raise ValueError(space)
    S = B.shape[0]

    # entiérisation : |Σ_i B[k,i] x_i − N·rhs_k| ≤ u_k   (×scale)
    Bi = np.round(B * scale).astype(np.int64)
    bk = np.round(N * rhs * scale).astype(np.int64)
    ub_u = int(np.abs(Bi).sum(axis=1).max() + np.abs(bk).max() + 1)

    mdl = cp_model.CpModel()
    x = [mdl.NewBoolVar(f"x{i}") for i in range(M)]
    u = [mdl.NewIntVar(0, ub_u, f"u{k}") for k in range(S)]
    mdl.Add(sum(x) == N)
    for k in range(S):
        e = sum(int(Bi[k, i]) * x[i] for i in range(M)) - int(bk[k])
        mdl.Add(e <= u[k]); mdl.Add(-e <= u[k])
    mdl.Minimize(sum(u))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    if mip_gap is not None:
        solver.parameters.relative_gap_limit = float(mip_gap)

    # callback : stoppe dès qu'une solution de coût nul est trouvée (cas 'clear')
    class _StopAtZero(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
        def on_solution_callback(self):
            if self.ObjectiveValue() <= 0.5:          # coût entier 0 (½ marge)
                self.StopSearch()

    status = solver.Solve(mdl, _StopAtZero())
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT : aucune solution dans le budget ({time_limit}s) "
                           f"— augmente time_limit. (status={solver.StatusName(status)})")
    P = np.array([i for i in range(M) if solver.Value(x[i]) == 1])
    dist = solver.ObjectiveValue() / scale            # déscalé ; en 'coef'/'signal' = L1·N? non :
    # NB : l'objectif scalé = scale·Σ_k|Σ_i B[k,i]x_i − N·rhs_k| = scale·N·‖B x/N − rhs‖₁
    dist = dist / N                                   # → ‖ B x / N − rhs ‖₁ (unité de l'espace)
    info = dict(status=solver.StatusName(status),
                optimal=(status == cp_model.OPTIMAL or solver.ObjectiveValue() <= 0.5),
                obj_scaled=solver.ObjectiveValue(),
                bound_scaled=solver.BestObjectiveBound())
    return P, float(dist), info


# ===================== identifiabilité en clair (sans solveur) ============ #
def clear_identifiability(A, in_idx, N):
    """
    Régime CLEAR : analyse d'identifiabilité EXACTE et instantanée, SANS optimiseur.

    En clair, l'agrégat publié est la projection exacte de la moyenne de D_in, donc
    D_in atteint une distance NULLE — c'est toujours un minimiseur. Inutile de lancer
    un MILP (le problème de cardinalité Σx=N est intractable à certifier). La seule
    question est l'UNICITÉ : existe-t-il un autre sous-ensemble de taille N de même
    moyenne projetée ? Une collision exige un z = x' − x_in non nul avec Σz = 0 et
    Aᵀz = 0, i.e. z dans le noyau de [Aᵀ ; 1ᵀ]. Si ce noyau est trivial (rang = M),
    aucune collision n'est possible → D_in est l'unique minimiseur (CERTIFIÉ).
    Sinon, on ne peut pas certifier l'unicité par l'algèbre seule ; dim(noyau) borne
    la richesse de l'espace d'ambiguïté (à p ≪ M, ce noyau est grand : la projection
    seule ne protège pas, mais l'unicité n'est ni prouvée ni infirmée par ce test).

    Renvoie dict : dist0 (≈0, vérif), rank, null_dim (dim de l'ambiguïté),
    unique_certified (bool), frac (1.0 : D_in EST un minimiseur, c'est la vérité-terrain).
    """
    A = np.asarray(A, float)
    M = A.shape[0]
    in_idx = np.asarray(in_idx, int)
    xin = np.zeros(M); xin[in_idx] = 1.0
    alpha_real = A[in_idx].mean(0)
    dist0 = float(np.abs(A.T @ xin / N - alpha_real).sum())
    tilde = np.vstack([A.T, np.ones(M)])          # [Aᵀ ; 1ᵀ]  (p+1, M)
    rank = int(np.linalg.matrix_rank(tilde))
    null_dim = int(M - rank)
    return dict(dist0=dist0, rank=rank, null_dim=null_dim,
                unique_certified=(null_dim == 0), frac=1.0)


# ===================== une attaque clear / noise ========================== #
def run_attack(inst, W, mode="clear", sigma=0.0, Delta2=None, clip_C=None, seed=0,
               space="coef", time_limit=30.0, mip_gap=None):
    """
    Lance l'attaque projetée sur une instance.

    mode='clear' : PAS de solveur — analyse d'identifiabilité directe (clear_identifiability).
        D_in atteint dist 0 (vérité-terrain), donc frac=1.0 ; le résultat renseigne
        l'unicité (null_dim, unique_certified). La projection seule ne protège pas.
    mode='noise' : cible = clip(α_real) + N(0, σ²I_p) → plus de zéro exact ; on résout
        l'attaque L1 par CP-SAT (exact_attack_proj_l1), best-so-far dans time_limit.
    space='coef' (R^p) ou 'signal' (R^48, unités de charge). time_limit/mip_gap : noise.

    Renvoie dict : mode, frac, N, et —
      clear : dist0, rank, null_dim, unique_certified ;
      noise : P, dist, inter, adv, sigma, space, optimal.
    """
    pj = project_instance(inst, W)
    if mode == "clear":
        idf = clear_identifiability(pj["A"], pj["in_idx"], pj["N"])
        return dict(mode="clear", frac=idf["frac"], N=pj["N"],
                    dist0=idf["dist0"], rank=idf["rank"], null_dim=idf["null_dim"],
                    unique_certified=idf["unique_certified"])
    elif mode == "noise":
        target = noisy_target_coeffs(pj["alpha_real"], sigma, Delta2, clip_C, seed=seed)
        P, dist, info = exact_attack_proj_l1(pj["A"], target, pj["N"], space=space,
                                             W=pj["W"], time_limit=time_limit, mip_gap=mip_gap)
        inter, frac = overlap(P, pj["in_idx"])
        adv = advantage(frac, pj["N"], pj["A"].shape[0])
        return dict(mode="noise", P=P, dist=dist, inter=inter, frac=frac, adv=adv,
                    sigma=float(sigma), space=space, N=pj["N"], optimal=info["optimal"])
    else:
        raise ValueError(mode)


def proj_recovery_curve(df_daily, W, Dsize, ratios, mode="clear",
                        sigma=0.0, Delta2=None, clip_C=None,
                        n_seeds=3, base_seed=2026, build_mode="mean",
                        space="coef", time_limit=30.0, mip_gap=None, progress=None):
    """
    % de foyers retrouvés vs ratio card(D_in)/card(D), attaque projetée, moyenné sur
    n_seeds instances (+ IQR), en chronométrant chaque MILP. Branche-toi via `progress`.

    mode='clear'/'noise' : voir run_attack. En 'noise', un seed de bruit distinct est
    tiré par (ratio, seed) — l'aléa d'instance et l'aléa DP sont décorrélés.
    space='coef'/'signal', time_limit/mip_gap : voir exact_attack_proj_l1.
    Δ₂ / clip_C : si fournis, le clip L2 est appliqué à la cible avant bruit (cohérent
    avec la calibration σ = sigma_one_release(eps, δ, Δ₂)).

    Renvoie dict {'ratios','mean','q25','q75','Dsize','t_mean','mode','sigma','space'}.
    """
    ratios = np.asarray(ratios, float)
    mean = np.zeros_like(ratios); q25 = np.zeros_like(ratios)
    q75 = np.zeros_like(ratios); tmean = np.zeros_like(ratios)
    for r, ratio in enumerate(ratios):
        n_in = min(max(int(round(ratio * Dsize)), 1), Dsize - 1)
        fr = np.empty(n_seeds); dts = np.empty(n_seeds)
        for s in range(n_seeds):
            inst = build_attack_instance(df_daily, n_in=n_in, n_decoy=Dsize - n_in,
                                         mode=build_mode, seed=base_seed + 1000 * r + s)
            t0 = time.perf_counter()
            res = run_attack(inst, W, mode=mode, sigma=sigma, Delta2=Delta2,
                             clip_C=clip_C, seed=base_seed + 7 * r + s,
                             space=space, time_limit=time_limit, mip_gap=mip_gap)
            dt = time.perf_counter() - t0
            fr[s] = res["frac"] * 100.0; dts[s] = dt
            if progress is not None:
                progress(dict(Dsize=int(Dsize), ratio=float(ratio), seed=int(s),
                              n_in=int(n_in), dt=float(dt), recovery=float(fr[s]),
                              mode=mode, sigma=float(sigma)))
        mean[r] = fr.mean(); q25[r] = np.quantile(fr, 0.25)
        q75[r] = np.quantile(fr, 0.75); tmean[r] = dts.mean()
    return dict(ratios=ratios, mean=mean, q25=q25, q75=q75, Dsize=int(Dsize),
                t_mean=tmean, mode=mode, sigma=float(sigma), space=space)


# ===================== métriques d'évaluation ============================== #
def overlap(P_idx, in_idx):
    """(#|P ∩ D_in|, fraction de D_in retrouvé)."""
    s, ins = set(map(int, P_idx)), set(map(int, in_idx))
    inter = len(s & ins)
    return inter, inter / len(ins)


def advantage(frac, N, Dsize):
    """Avantage sur le hasard : un P aléatoire recouvre en moyenne N/|D| de D_in."""
    chance = N / Dsize
    return (frac - chance) / (1.0 - chance + 1e-12)


def mutual_coherence(D):
    """μ(D) = max_{i≠j} |⟨ℓ_i,ℓ_j⟩| / (‖ℓ_i‖‖ℓ_j‖) — proche de 1 ⇒ identifiabilité difficile."""
    D = np.asarray(D, float)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    G = np.abs(Dn @ Dn.T)
    np.fill_diagonal(G, 0.0)
    return float(G.max())

# ============== score continu par frequence de selection + ROC ============ #
def selection_frequency(P_list, M):
    """
    Score continu par candidat a partir de K realisations de l'attaque dure.

    P_list : liste de K tableaux d'indices (chacun la sortie P_idx d'une attaque
             CP-SAT sur un tirage de bruit DP independant, MEME instance).
    M      : taille du pool |D| (nombre de candidats).

    Renvoie p_hat (M,) : p_hat[i] = fraction des K realisations ou i in P.
    Analogue, pour cette attaque, de la frequence de vote d'un ensemble (bagging /
    stability selection, Meinshausen & Buhlmann 2010) : un score continu substitue
    a une decision dure ponctuelle, sans re-resoudre aucun MILP -- on reutilise les
    memes runs que ceux deja lances pour la quantification d'incertitude.
    """
    counts = np.zeros(M)
    for P in P_list:
        counts[np.asarray(P, int)] += 1.0
    return counts / len(P_list)


def _auc_trapz(x, y):
    """AUC par la regle des trapezes, independante de la version de numpy."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def roc_from_scores(p_hat, in_idx, M):
    """
    Courbe ROC (FPR, TPR) obtenue en seuillant p_hat a tous les tau distincts
    observes (bornes 0 et 1 ajoutees), et son AUC.

    in_idx : indices verite-terrain de D_in (taille N). Le point obtenu en
    seuillant au n-ieme plus grand p_hat (top-N) doit coincider avec le %
    retrouve par l'attaque dure au meme eps -- verification de coherence utile.

    Renvoie dict {tau, tpr, fpr, auc}.
    """
    in_idx = np.asarray(in_idx, int)
    y_true = np.zeros(M, dtype=int)
    y_true[in_idx] = 1
    N = in_idx.size
    thresholds = np.unique(np.concatenate([[0.0, 1.0 + 1e-9], p_hat]))[::-1]
    tpr = np.empty(thresholds.size); fpr = np.empty(thresholds.size)
    for k, tau in enumerate(thresholds):
        pred = (p_hat >= tau).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr[k] = tp / N
        fpr[k] = fp / (M - N)
    return dict(tau=thresholds, tpr=tpr, fpr=fpr, auc=_auc_trapz(fpr, tpr))

def roc_from_scores_grid(p_hat, in_idx, M, n_tau=50):
    """
    Courbe ROC evaluee sur une grille reguliere de n_tau seuils entre 0 et 1,
    identique pour tous les eps -- plutot que les valeurs distinctes de p_hat.
    Utile pour comparer plusieurs courbes point a point sur la meme grille.

    in_idx : indices verite-terrain de D_in (taille N).
    Renvoie dict {tau, tpr, fpr, auc}.
    """
    in_idx = np.asarray(in_idx, int)
    y_true = np.zeros(M, dtype=int)
    y_true[in_idx] = 1
    N = in_idx.size
    thresholds = np.linspace(1.0, 0.0, n_tau)   # decroissant, de 1 vers 0
    tpr = np.empty(n_tau); fpr = np.empty(n_tau)
    for k, tau in enumerate(thresholds):
        pred = (p_hat >= tau).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr[k] = tp / N
        fpr[k] = fp / (M - N)
    return dict(tau=thresholds, tpr=tpr, fpr=fpr, auc=_auc_trapz(fpr, tpr))