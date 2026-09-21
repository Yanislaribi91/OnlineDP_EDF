"""
online_dp : pipeline online Differential Privacy pour courbes de charge.

Modules :
    config      Config dataclass (chemins, seeds, params) + clés de cache
    cache       compute_or_load (joblib / npz / parquet)
    data        chargement, découpage journalier, split panel/cible, agrégation (NB 1)
    metrics     métriques canoniques (per_day_metrics, acf, compute_wpa_metrics)
    basis       bases NMF+QR / SVD matchées, projection, sensibilité, pools (NB 3)
    gam         GAMResidualModel : baseline additive + résidus (NB 3)
    mechanisms  LPA ponctuel (NB 2), WPA Laplace & gaussien + calibration RDP (NB 4)
    vfast       Kalman vectoriel + PID (VFAST), tuning de q, sensibilité (NB 4)
    viz         couleurs, libellés FR, helpers de figures
"""
from . import config, cache, data, metrics, basis, gam, mechanisms, vfast, viz  

__all__ = ["config", "cache", "data", "metrics", "basis", "gam", "mechanisms", "vfast", "viz"]
__version__ = "0.1.0"
