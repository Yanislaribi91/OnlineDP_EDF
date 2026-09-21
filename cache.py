"""
Cache disque pour ne calculer qu'une fois les objets coûteux
(bases d'ondelettes, modèles GAM, tenseurs VFAST...) et les partager entre
notebooks.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

CACHE = Path("cache")


def _path(key: str, ext: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{key}.{ext}"


def compute_or_load(key: str, fn, force: bool = False, backend: str | None = None):
    """
    Renvoie l'objet associé à `key`, depuis le cache s'il existe, sinon en
    appelant `fn()` (et en mettant en cache le résultat).

    Args:
        key     : identifiant (versionné par les params, cf. Config.key).
        fn      : callable sans argument produisant l'objet à mettre en cache.
        force   : recalcule et écrase le cache même s'il existe.
        backend : 'joblib' | 'npz' | 'parquet' (sinon auto selon le type renvoyé).
    """
    # auto-détection du backend depuis un éventuel suffixe explicite
    for b, ext in (("npz", "npz"), ("parquet", "parquet"), ("joblib", "joblib")):
        if key.endswith("." + ext):
            backend, key = b, key[: -(len(ext) + 1)]

    # lecture 
    if not force:
        for b, ext, loader in (
            ("joblib",  "joblib",  joblib.load),
            ("npz",     "npz",     lambda p: dict(np.load(p, allow_pickle=False))),
            ("parquet", "parquet", pd.read_parquet),
        ):
            p = _path(key, ext)
            if p.exists():
                obj = loader(p)
                # un .npz à clé unique 'arr' est rendu comme un simple ndarray
                if ext == "npz" and set(obj) == {"arr"}:
                    obj = obj["arr"]
                print(f"[cache] '{key}' rechargé ({ext}).")
                return obj

    #  calcul 
    obj = fn()

    if backend is None:
        if isinstance(obj, np.ndarray):
            backend = "npz"
        elif isinstance(obj, pd.DataFrame):
            backend = "parquet"
        else:
            backend = "joblib"

    if backend == "npz":
        arr = obj if isinstance(obj, np.ndarray) else None
        if arr is not None:
            np.savez_compressed(_path(key, "npz"), arr=arr)
        else:  # dict d'arrays
            np.savez_compressed(_path(key, "npz"), **obj)
    elif backend == "parquet":
        obj.to_parquet(_path(key, "parquet"))
    else:
        joblib.dump(obj, _path(key, "joblib"))

    print(f"[cache] '{key}' calculé et mis en cache ({backend}).")
    return obj


def clear(key: str | None = None):
    """Supprime un artefact (toutes extensions) ou vide tout le cache si key=None."""
    if key is None:
        for p in CACHE.glob("*"):
            p.unlink()
        print("[cache] vidé.")
        return
    for ext in ("joblib", "npz", "parquet"):
        p = _path(key, ext)
        if p.exists():
            p.unlink()
            print(f"[cache] '{key}.{ext}' supprimé.")
