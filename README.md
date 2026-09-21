# OnlineDP_EDF
Code des expériences de mon stage de M2  réalisé à EDF R&amp;D Lab Paris-Saclay sous la direction de Margaux Brégère portant sur la publication continue de courbes de consommation agrégées sous confidentialité différentielle (DP). Les données utilisées sont disponibles à l'adresse suivante : 

## Arborescence

```
projet/                        # = onlinedp_stage26/
├── README.md
├── pyproject.toml             # `pip install -e .` depuis ce dossier (optionnel)
├── online_dp/                 # le package
│   ├── __init__.py
│   ├── config.py              # Config (chemins, seeds, params) + clés de cache
│   ├── cache.py               # compute_or_load (joblib / npz / parquet)
│   ├── data.py                # NB1 : load, split journalier, panel/cible, agrégat
│   ├── metrics.py             # métriques canoniques UNIQUES (per_day_metrics, acf, stats)
│   ├── basis.py               # NB3 : bases NMF/SVD matchées, projection, sensibilité, pools
│   ├── gam.py                 # NB3 : GAMResidualModel (baseline + résidus)
│   ├── mechanisms.py          # NB2/NB4 : LPA & gaussien ponctuels, WPA Laplace,
│   │                          #           WPA gaussien isotrope (RDP) et anisotrope (zCDP)
│   ├── vfast.py               # NB4 : Kalman vectoriel + PID (laplace / gaussien /
│   │                          #       gaussien anisotrope), tuning q, sensibilité
│   ├── attack.py              # NB5 : attaque de reconstruction d'agrégat (MILP L1,
│   │                          #       brut / projeté / bruité, recovery, ROC/AUC)
│   └── viz.py                 # couleurs, libellés FR, helpers de figures
├── cache/                     # artefacts lourds
└── notebooks/
    ├── 01_data_prep.ipynb
    ├── 02_pointwise_mechanism.ipynb
    ├── 03_projections.ipynb
    ├── 04_wpa_vfast.ipynb
    └── 05_attack.ipynb
```

## Pipeline des notebooks

- **NB1 — `01_data_prep`** : chargement, découpage 30 min → journalier (48 créneaux
  `t00..t47`), construction du panel public, de la cible et de l'agrégat, statistiques
  descriptives (répartition Power × ToU, thermosensibilité).
- **NB2 — `02_pointwise_mechanism`** : mécanismes ponctuels (Laplace et gaussien) sur
  l'agrégat brut, budgets ε/jour et ε/T, effet de la méthode de clipping sur la
  sensibilité (`clip_quantile=0.95`).
- **NB3 — `03_projections`** : bases SVD / NMF matchées à la taille d'agrégat,
  erreur de projection vs `p`, compromis projection / perturbation (choix de `p*`),
  sensibilité des coefficients, baseline + résidus GAM.
- **NB4 — `04_wpa_vfast`** : WPA en ligne (Laplace, gaussien isotrope et anisotrope,
  budgets ε/jour et ε/T) et VFAST (Kalman vectoriel + PID, un par mécanisme),
  tuning de `q` sur agrégats publics, benchmark utilité / confidentialité,
  sensibilité aux hyperparamètres (`M_ratio`, `θ`, `ξ`).
- **NB5 — `05_attack`** : attaque de reconstruction d'agrégat par MILP (optimum exact
  L1, cardinalité contrainte), en espace brut R⁴⁸ ou projeté R^p, avec ou sans bruit
  DP sur la cible ; courbes de recovery, fréquences de sélection, courbes ROC / AUC
  et matrices de confusion.


