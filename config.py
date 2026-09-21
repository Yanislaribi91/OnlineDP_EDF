"""
Configuration centralisée de la pipeline online-DP.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # chemins 
    data_dir: str = "../data/DataDiffusionDeepCourboGen/"
    cache_dir: str = "cache"

    #  échantillonnage / split (déterministes) 
    N: int = 1000                 # nb de foyers tirés parmi les 10 000
    n_panel: int = 500             # foyers publics (base, sensibilité, Q, GAM)
    n_target: int = 50             # foyers de l'agrégat protégé
    seed: int = 0                 # graine du tirage + du split

    # calendrier  
    calendar_start: str = "2022-10-02 20:00:00"
    slots_per_day: int = 48

    #  base d'ondelettes 
    pmax: int = 47                # rang max (SVD / NMF)
    n_groups: int = 100           # nb d'agrégats synthétiques pour fitter la base
    fit_subsample: int = 10000    # plafond d'échantillons pour le fit
    nmf_max_iter: int = 500
    nmf_tol: float = 1e-4

    #  DP 
    clip_quantile: float = 0.95   # quantile pour la borne de clip (sensibilité)
    delta_dp: float = 1e-5        # δ du mécanisme gaussien (RDP)

    def __post_init__(self):
        # garantit que les dossiers existent
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def key(self, name: str, **extra) -> str:
        """
        Construit une clé de cache versionnée par les paramètres pertinents.

            cfg.key("dataset")                      -> dataset__N1000_np50_nt5_s0
            cfg.key("basis", agg=5, p=10)           -> basis__N1000_np50_nt5_s0_agg5_p10
        """
        base = f"{name}__N{self.N}_np{self.n_panel}_nt{self.n_target}_s{self.seed}"
        for k, v in extra.items():
            base += f"_{k}{v}"
        return base

    def as_dict(self) -> dict:
        return asdict(self)
