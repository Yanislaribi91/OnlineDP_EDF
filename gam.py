"""

GAM "fait main" (baseline additive + résidus) pour le conditionnement par
température / calendrier / classe (Power × ToU), encapsulé dans une classe pour
supprimer toute dépendance à des variables globales (dates_canon, _panel_tensor…).

Modèle, par foyer i, jour t, créneau k :
    L = f_k(temp) + g_k(saison) + w_k·weekend + a_k·Power + b_k·ToU + ε
soit UN Ridge par créneau (48 modèles) : l'interaction température×heure est
portée par le découpage en créneaux ; les covariables entrent additivement.

Usage :
    gam = GAMResidualModel(df_temp, df_label).fit(panel_ids, panel_tensor, dates)
    baseline = gam.baseline_agg(target_users, df_temp, dates)
    residual = gam.residual(panel_tensor, panel_ids, dates)   # pour la base SVD des résidus
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge

POWER_CATS = [6, 9, 12]
TOU_CATS = [0, 1, 2]


def seasonal_features(doy):
    a = 2 * np.pi * np.asarray(doy) / 365.25
    return np.column_stack([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)])


def onehot(vals, cats):
    return np.stack([(np.asarray(vals) == c).astype(float) for c in cats], axis=1)


def build_temp_daily(df_temp, users, dates_index):
    """Température régionale (moyenne sur `users`) pivotée en (T, 48), réindexée sur dates_index."""
    s = df_temp[list(users)].mean(axis=1)
    d = s.index.normalize()
    slot = s.index.hour * 2 + s.index.minute // 30
    piv = (pd.DataFrame({"temp": s.values, "date": d, "slot": slot})
           .pivot_table(index="date", columns="slot", values="temp").sort_index())
    piv.index = pd.to_datetime(piv.index)
    return piv.reindex(dates_index).values


class GAMResidualModel:
    """Baseline GAM par foyer (Ridge par créneau) + calcul des résidus."""

    def __init__(self, df_temp, df_label, ridge_alpha=1.0, n_knots=5, degree=3):
        self.df_temp = df_temp
        self.df_label = df_label
        self.ridge_alpha = ridge_alpha
        self.n_knots = n_knots
        self.degree = degree
        self.slot_models = None
        self.temp_scaler = None
        self.temp_spline = None

    # ------------------------------------------------------------------ #
    def _temp_feat(self, col):
        return self.temp_spline.transform(self.temp_scaler.transform(col))

    def fit(self, panel_ids, panel_tensor, dates_index):
        """Ajuste les 48 Ridge sur le panel (empilement foyer-majeur / jour-mineur)."""
        dates = pd.to_datetime(dates_index)
        doy = np.asarray(dates.dayofyear)
        wknd = (np.asarray(dates.dayofweek) >= 5).astype(float)
        temp_daily = build_temp_daily(self.df_temp, panel_ids, dates_index)
        assert not np.isnan(temp_daily).any(), "NaN température panel"

        n_hh, T = len(panel_ids), temp_daily.shape[0]
        power_oh = onehot(self.df_label.loc[panel_ids, "Power"].values, POWER_CATS)[:, 1:]
        tou_oh = onehot(self.df_label.loc[panel_ids, "ToU"].values, TOU_CATS)[:, 1:]

        self.temp_scaler = StandardScaler().fit(temp_daily.reshape(-1, 1))
        self.temp_spline = SplineTransformer(
            n_knots=self.n_knots, degree=self.degree,
            include_bias=False, extrapolation="constant"
        ).fit(self.temp_scaler.transform(temp_daily.reshape(-1, 1)))

        const_block = np.hstack([
            np.tile(seasonal_features(doy), (n_hh, 1)),
            np.tile(wknd, n_hh)[:, None],
            np.repeat(power_oh, T, axis=0),
            np.repeat(tou_oh, T, axis=0),
        ])
        self.slot_models = []
        for k in range(48):
            Xk = np.hstack([self._temp_feat(np.tile(temp_daily[:, k], n_hh)[:, None]), const_block])
            self.slot_models.append(Ridge(alpha=self.ridge_alpha).fit(Xk, panel_tensor[:, :, k].reshape(-1)))
        return self

    # ------------------------------------------------------------------ #
    def baseline_class(self, power, tou, df_temp, dates_index, temp_users=None):
        """
        Baseline d'une classe homogène (power, tou) : (T, 48).
        `temp_users` = foyers servant à la température régionale (défaut : toutes les colonnes).
        """
        dates = pd.to_datetime(dates_index)
        doy, wknd = np.asarray(dates.dayofyear), (np.asarray(dates.dayofweek) >= 5).astype(float)
        users = list(temp_users) if temp_users is not None else list(df_temp.columns)
        temp_daily = build_temp_daily(df_temp, users, dates_index)
        cls = np.hstack([onehot([power], POWER_CATS)[:, 1:], onehot([tou], TOU_CATS)[:, 1:]])[0]
        seas, wk = seasonal_features(doy), wknd[:, None]
        out = np.zeros((len(dates), 48))
        for k in range(48):
            Xk = np.hstack([self._temp_feat(temp_daily[:, k:k + 1]), seas, wk, np.tile(cls, (len(dates), 1))])
            out[:, k] = self.slot_models[k].predict(Xk)
        return out

    def baseline_agg(self, users, df_temp, dates_index):
        """Baseline d'un agrégat : moyenne des baselines par foyer (chacun avec sa classe). (T, 48)."""
        dates = pd.to_datetime(dates_index)
        doy, wknd = np.asarray(dates.dayofyear), (np.asarray(dates.dayofweek) >= 5).astype(float)
        temp_daily = build_temp_daily(df_temp, users, dates_index)
        pw = onehot(self.df_label.loc[list(users), "Power"].values, POWER_CATS)[:, 1:]
        tu = onehot(self.df_label.loc[list(users), "ToU"].values, TOU_CATS)[:, 1:]
        cls = np.hstack([pw, tu])
        seas, wk = seasonal_features(doy), wknd[:, None]
        D, out = len(dates), np.zeros((len(dates), 48))
        for k in range(48):
            block = np.hstack([self._temp_feat(temp_daily[:, k:k + 1]), seas, wk])
            acc = np.zeros(D)
            for u in range(len(users)):
                acc += self.slot_models[k].predict(np.hstack([block, np.tile(cls[u], (D, 1))]))
            out[:, k] = acc / len(users)
        return out

    def residual(self, panel_tensor, panel_ids, dates_index):
        """Résidu signé L − baseline par foyer : tenseur (n_hh, T, 48), pour la base SVD des résidus."""
        dates = pd.to_datetime(dates_index)
        doy, wknd = np.asarray(dates.dayofyear), (np.asarray(dates.dayofweek) >= 5).astype(float)
        temp_daily = build_temp_daily(self.df_temp, panel_ids, dates_index)
        n_hh, T = len(panel_ids), temp_daily.shape[0]
        power_oh = onehot(self.df_label.loc[panel_ids, "Power"].values, POWER_CATS)[:, 1:]
        tou_oh = onehot(self.df_label.loc[panel_ids, "ToU"].values, TOU_CATS)[:, 1:]
        const_block = np.hstack([
            np.tile(seasonal_features(doy), (n_hh, 1)),
            np.tile(wknd, n_hh)[:, None],
            np.repeat(power_oh, T, axis=0),
            np.repeat(tou_oh, T, axis=0),
        ])
        baseline = np.zeros((n_hh, T, 48))
        for k in range(48):
            Xk = np.hstack([self._temp_feat(np.tile(temp_daily[:, k], n_hh)[:, None]), const_block])
            baseline[:, :, k] = self.slot_models[k].predict(Xk).reshape(n_hh, T)
        return panel_tensor - baseline

class GAMResidualModelPyGAM:
    """Baseline GAM par creneau via pygam : spline PENALISEE sur la temperature
    (lissage selectionne par gridsearch) + termes lineaires saison / weekend /
    Power / ToU. API identique a GAMResidualModel (fit / baseline_agg / residual).
    Plus couteux : 48 LinearGAM."""

    def __init__(self, df_temp, df_label, n_splines=10, lam_grid=None,
                 fit_rows=40000, seed=0):
        self.df_temp, self.df_label = df_temp, df_label
        self.n_splines = n_splines
        self.lam_grid = np.logspace(-1, 3, 4) if lam_grid is None else np.asarray(lam_grid)
        self.fit_rows, self.seed = fit_rows, seed
        self.gams = None
        self.temp_scaler = None

    def _design(self, temp_col, seas, wk, cls_block):
        # temp brute scalee (pygam construit la spline) + covariables lineaires
        return np.hstack([temp_col, seas, wk, cls_block])

    def fit(self, panel_ids, panel_tensor, dates_index):
        from pygam import LinearGAM, s, l
        from tqdm.auto import tqdm
        dates = pd.to_datetime(dates_index)
        doy = np.asarray(dates.dayofyear)
        wknd = (np.asarray(dates.dayofweek) >= 5).astype(float)
        temp_daily = build_temp_daily(self.df_temp, panel_ids, dates_index)
        assert not np.isnan(temp_daily).any(), "NaN temperature panel"
        n_hh, T = len(panel_ids), temp_daily.shape[0]
        power_oh = onehot(self.df_label.loc[panel_ids, "Power"].values, POWER_CATS)[:, 1:]
        tou_oh   = onehot(self.df_label.loc[panel_ids, "ToU"].values,   TOU_CATS)[:, 1:]
        self.temp_scaler = StandardScaler().fit(temp_daily.reshape(-1, 1))

        seas_rep = np.tile(seasonal_features(doy), (n_hh, 1))
        wknd_rep = np.tile(wknd, n_hh)[:, None]
        cls_rep  = np.hstack([np.repeat(power_oh, T, axis=0), np.repeat(tou_oh, T, axis=0)])
        ncol = 1 + seas_rep.shape[1] + 1 + cls_rep.shape[1]

        rng = np.random.default_rng(self.seed)
        n_rows = n_hh * T
        idx = (rng.choice(n_rows, self.fit_rows, replace=False)
               if n_rows > self.fit_rows else np.arange(n_rows))

        terms = s(0, n_splines=self.n_splines)          # spline penalisee : temperature
        for j in range(1, ncol):                        # lineaire : saison / weekend / classe
            terms = terms + l(j)

        self.gams = []
        for k in tqdm(range(48), desc="pygam (1 GAM / creneau)"):
            tcol = self.temp_scaler.transform(np.tile(temp_daily[:, k], n_hh)[:, None])
            X = self._design(tcol, seas_rep, wknd_rep, cls_rep)[idx]
            y = panel_tensor[:, :, k].reshape(-1)[idx]
            self.gams.append(LinearGAM(terms).gridsearch(X, y, lam=self.lam_grid, progress=False))
        return self

    def baseline_agg(self, users, df_temp, dates_index):
        dates = pd.to_datetime(dates_index)
        doy = np.asarray(dates.dayofyear); wknd = (np.asarray(dates.dayofweek) >= 5).astype(float)
        temp_daily = build_temp_daily(df_temp, users, dates_index)
        pw = onehot(self.df_label.loc[list(users), "Power"].values, POWER_CATS)[:, 1:]
        tu = onehot(self.df_label.loc[list(users), "ToU"].values,   TOU_CATS)[:, 1:]
        cls = np.hstack([pw, tu]); seas = seasonal_features(doy); wk = wknd[:, None]
        D, out = len(dates), np.zeros((len(dates), 48))
        for k in range(48):
            tcol = self.temp_scaler.transform(temp_daily[:, k:k + 1])
            acc = np.zeros(D)
            for u in range(len(users)):
                acc += self.gams[k].predict(self._design(tcol, seas, wk, np.tile(cls[u], (D, 1))))
            out[:, k] = acc / len(users)
        return out

    def residual(self, panel_tensor, panel_ids, dates_index):
        dates = pd.to_datetime(dates_index)
        doy = np.asarray(dates.dayofyear); wknd = (np.asarray(dates.dayofweek) >= 5).astype(float)
        temp_daily = build_temp_daily(self.df_temp, panel_ids, dates_index)
        n_hh, T = len(panel_ids), temp_daily.shape[0]
        power_oh = onehot(self.df_label.loc[panel_ids, "Power"].values, POWER_CATS)[:, 1:]
        tou_oh   = onehot(self.df_label.loc[panel_ids, "ToU"].values,   TOU_CATS)[:, 1:]
        seas_rep = np.tile(seasonal_features(doy), (n_hh, 1))
        wknd_rep = np.tile(wknd, n_hh)[:, None]
        cls_rep  = np.hstack([np.repeat(power_oh, T, axis=0), np.repeat(tou_oh, T, axis=0)])
        baseline = np.zeros((n_hh, T, 48))
        for k in range(48):
            tcol = self.temp_scaler.transform(np.tile(temp_daily[:, k], n_hh)[:, None])
            X = self._design(tcol, seas_rep, wknd_rep, cls_rep)
            baseline[:, :, k] = self.gams[k].predict(X).reshape(n_hh, T)
        return panel_tensor - baseline
