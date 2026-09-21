"""
Chargement et mise en forme des données DeepCourboGen.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def load_raw(cfg):
    """Charge labels / température / courbes de charge, aligne le calendrier,
       renvoie (df_label, df_temp, df_load)."""
    d = cfg.data_dir
    df_label = pd.read_feather(d + "labels_export.feather")
    df_temp = pd.read_feather(d + "temperature_export.feather").T
    df_load = pd.read_feather(d + "load_curve_export.feather").T

    calendar = pd.date_range(start=cfg.calendar_start, periods=len(df_load), freq="30min")
    df_load.index = calendar
    df_temp.index = calendar
    df_load.index.names = ["date_time"]
    df_temp.index.names = ["date_time"]

    first = (df_load.index.hour == 0) & (df_load.index.minute == 0)
    last = (df_load.index.hour == 23) & (df_load.index.minute == 30)
    start_ts, end_ts = df_load.index[first][0], df_load.index[last][-1]
    df_load = df_load.loc[start_ts:end_ts]
    df_temp = df_temp.loc[start_ts:end_ts]
    assert len(df_load) % cfg.slots_per_day == 0, "Pas un multiple de 48 après trim"
    return df_label, df_temp, df_load


def sample_households(df_load, cfg):
    """Sous-échantillonne N foyers de façon reproductible."""
    rng = np.random.default_rng(cfg.seed)
    cols = rng.choice(np.asarray(df_load.columns), size=cfg.N, replace=False)
    return df_load[list(cols)]


def split_daily_profiles(df_load):
    """
    Découpe les chroniques en profils journaliers de 48 valeurs.
    Renvoie df_daily : MultiIndex (household_id, date), colonnes t00..t47.
    """
    df_long = df_load.stack().rename("load").reset_index()
    df_long.columns = ["date_time", "household_id", "load"]
    df_long["date"] = df_long["date_time"].dt.date
    df_long["slot"] = df_long["date_time"].dt.hour * 2 + df_long["date_time"].dt.minute // 30
    df_daily = df_long.pivot_table(index=["household_id", "date"], columns="slot", values="load")
    df_daily.columns = [f"t{c:02d}" for c in df_daily.columns]
    return df_daily.dropna(axis=0, how="any")


def panel_target_split(df_daily, cfg):
    """Tirage reproductible ET disjoint panel / cible. Renvoie (panel_users, target_users)."""
    all_users = df_daily.index.get_level_values("household_id").unique()
    assert len(all_users) >= cfg.n_panel + cfg.n_target, (
        f"{len(all_users)} foyers dispo, {cfg.n_panel + cfg.n_target} requis — augmente cfg.N."
    )
    perm = np.random.default_rng(cfg.seed).permutation(all_users)
    return perm[: cfg.n_panel], perm[cfg.n_panel : cfg.n_panel + cfg.n_target]


def daily_aggregate(df_daily, users):
    """Agrégat journalier (moyenne) sur `users` : DataFrame (T, 48) indexé par date triée."""
    agg = df_daily.loc[list(users)].groupby(level="date").mean()
    agg.index = pd.to_datetime(agg.index)
    return agg.sort_index()


def household_tensor(df_daily, users):
    """Tenseur (n_hh, T, 48) + liste des household_id (foyers de longueur égale)."""
    sub = df_daily.loc[list(users)]
    ids = sub.index.get_level_values("household_id").unique().tolist()
    lengths = {len(sub.loc[h]) for h in ids}
    assert len(lengths) == 1, f"Foyers de longueurs différentes : {lengths}"
    T = lengths.pop()
    tensor = np.empty((len(ids), T, 48), dtype=np.float32)
    for k, h in enumerate(ids):
        tensor[k] = sub.loc[h].sort_index().values
    return tensor, ids


# --------------------------------------------------------------------------- #
def build_dataset(cfg) -> dict:
    """
    Pipeline notebook 1 complet, renvoie un dict cache-able contenant tout ce dont
    les notebooks 2 à 4 ont besoin. À envelopper dans compute_or_load(cfg.key('dataset'), ...).
    """
    df_label, df_temp, df_load = load_raw(cfg)
    df_load_sub = sample_households(df_load, cfg)
    df_daily = split_daily_profiles(df_load_sub)
    panel_users, target_users = panel_target_split(df_daily, cfg)
    df_agg = daily_aggregate(df_daily, target_users)
    panel_tensor, panel_ids = household_tensor(df_daily, panel_users)
    return dict(
        df_label=df_label,
        df_temp=df_temp,
        df_daily=df_daily,
        df_agg=df_agg,
        panel_users=np.asarray(panel_users),
        target_users=np.asarray(target_users),
        panel_profiles=df_daily.loc[list(panel_users)].values,   # (n_panel*T, 48)
        panel_tensor=panel_tensor,                               # (n_panel, T, 48)
        panel_ids=panel_ids,
        dates=df_agg.index,
    )


def label_distribution(df_label):
    """Tableau croisé Power × ToU (effectifs) pour les stats descriptives."""
    return pd.crosstab(df_label["Power"], df_label["ToU"])
