# -*- coding: utf-8 -*-
"""
run_continu_calage_HSM_res_cascade.py
----------------------------------------------------
RUN  : lit la série continue, extrait l'année hydrologique 2022-10-01 -> 2023-09-30,
fait une validation croisée sur 2 splits (H1->H2 et H2->H1) avec :

Warm-up 1 mois (au début de la période de calage FULL), exclu de l'objectif
 Calage en 2 étapes :
   - ETAPE 1 : calage Qinf 
              optimise k_infiltr + k_seepage
              objectif = err_volume relatif + log-RMSE forme
   - ETAPE 2 : calage ruissellement sur fenêtre objectif (après warmup)
              optimise k_runoff1 + k_runoff2 (ki/ks figés)
              objectif = RMSE(Qruiss) + petite pénalité forme Qinf
 Exports :
   - simulation.csv (période hydro complète : warmup/calib/valid)
   - params.xlsx
   - metrics_periods.xlsx (CALIB_FULL, CALIB_OBJ, VALID)
   - metrics_monthly.xlsx (bilans mensuels + annuel)

Dépendances : numpy, pandas, scipy, openpyxl
"""

from __future__ import annotations
from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================================================
# Conversions
# =========================================================
def lh_to_m3s(q_lh):
    return np.asarray(q_lh, float) / 1000.0 / 3600.0


def m3s_to_lh(q_m3s):
    return np.asarray(q_m3s, float) * 1000.0 * 3600.0


def mm_per_step_to_mps(mm_step, dt_s):
    return np.asarray(mm_step, float) * 1e-3 / float(dt_s)


def infil_mm_h_to_m_s(v_mm_h):
    return float(v_mm_h) * 1e-3 / 3600.0


# =========================================================
# ETP synthétique (ET appliquée uniquement au stock Ia)
# =========================================================
def build_constant_daytime_etp_rate(time_index: pd.DatetimeIndex,
                                    etp_mm_per_day=2.0,
                                    start_hour=8,
                                    end_hour=20) -> np.ndarray:
    nt = len(time_index)
    etp = np.zeros(nt, dtype=float)
    active_hours = max(end_hour - start_hour, 1)
    etp_mm_h = etp_mm_per_day / active_hours
    for i, t in enumerate(time_index):
        hour = t.hour + t.minute / 60.0
        if start_hour <= hour < end_hour:
            etp[i] = (etp_mm_h / 1000.0) / 3600.0  # m/s
    return etp


# =========================================================
# Lecture série continue + mapping colonnes
# =========================================================
def read_continuous_csv(csv_path: Path, sep=";"):
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier continu introuvable: {csv_path}")

    df = pd.read_csv(csv_path, sep=sep)

    READ_MAPPING = {
        "date": "Date",
        "P_mm": "Hauteur_de_pluie_mm",
        "Q_inf_LH": "Q_inf_LH",
        "Q_ruiss_LH": "Q_ruiss_LH",
    }
    for _, col in READ_MAPPING.items():
        if col not in df.columns:
            raise ValueError(
                f"Colonne '{col}' manquante dans {csv_path.name}. Colonnes présentes={list(df.columns)}"
            )

    df = df.copy()
    df["date"] = pd.to_datetime(
        df[READ_MAPPING["date"]].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    if df["date"].isna().any():
        bad = df.loc[df["date"].isna(), READ_MAPPING["date"]].head(10).tolist()
        raise ValueError(f"Dates non parsables (exemples): {bad}")

    df = df.sort_values("date").reset_index(drop=True)
    t = pd.DatetimeIndex(df["date"])

    diffs = t.to_series().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        raise ValueError("Série continue trop courte pour inférer dt.")
    dt_s = float(np.median(diffs))

    P_mm = pd.to_numeric(df[READ_MAPPING["P_mm"]], errors="coerce").fillna(0.0).to_numpy(float)
    Q_inf_LH = pd.to_numeric(df[READ_MAPPING["Q_inf_LH"]], errors="coerce").fillna(0.0).to_numpy(float)
    Q_ruiss_LH = pd.to_numeric(df[READ_MAPPING["Q_ruiss_LH"]], errors="coerce").fillna(0.0).to_numpy(float)

    P_mm = np.clip(P_mm, 0.0, None)
    Q_inf_LH = np.clip(Q_inf_LH, 0.0, None)
    Q_ruiss_LH = np.clip(Q_ruiss_LH, 0.0, None)

    return t, P_mm, Q_inf_LH, Q_ruiss_LH, dt_s


# =========================================================
# Choix dt_internal qui DIVISE dt_obs
# =========================================================
def choose_dt_internal(dt_obs: float, preferred: float = 60.0) -> float:
    dt_obs = float(dt_obs)
    candidates = [preferred, 120.0, 60.0, 30.0, 20.0, 15.0, 12.0, 10.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    candidates = [c for c in candidates if 0 < c <= dt_obs + 1e-9]
    for c in candidates:
        nsub = dt_obs / c
        if abs(nsub - round(nsub)) < 1e-9:
            return float(c)
    return float(dt_obs)


# =========================================================
# Modèle SCS-HSM CASCADE r1 -> r2 (substepping)
# =========================================================
def run_scs_hsm_cascade(dt_obs,
                        p_rate, etp_rate,
                        i_a, s,
                        k_infiltr,      # m/s
                        k_seepage,      # s^-1
                        k_runoff1,      # s^-1
                        k_runoff2,      # s^-1
                        dt_internal=None,
                        infil_from_surface=True,
                        h_a0=0.0, h_s0=0.0, h_r10=0.0, h_r20=0.0,
                        store_states=True,
                        store_fluxes=True,
                        store_losses=True):
    """
    Simulation SCS-HSM + cascade r1->r2, substepping.
    Mode objectif rapide:
      store_states=False, store_fluxes=False, store_losses=False
      -> calcule uniquement r_out.
    """
    p_rate = np.nan_to_num(np.asarray(p_rate, float), nan=0.0)
    etp_rate = np.nan_to_num(np.asarray(etp_rate, float), nan=0.0)
    nt = len(p_rate)
    dt_obs = float(dt_obs)

    if dt_internal is None:
        dt_internal = dt_obs
    dt_internal = float(dt_internal)
    if dt_internal <= 0:
        raise ValueError("dt_internal doit être > 0")

    nsub = int(round(dt_obs / dt_internal))
    if nsub < 1:
        nsub = 1
    dt_int = dt_obs / nsub
    if abs(nsub * dt_int - dt_obs) > 1e-6:
        raise ValueError(f"dt_internal doit diviser dt_obs. dt_obs={dt_obs}, dt_internal={dt_internal}")

    r_out = np.zeros(nt, dtype=float)

    q = infil = r_gen = None
    sa_loss = seep_loss = None

    if store_fluxes:
        q = np.zeros(nt, dtype=float)
        infil = np.zeros(nt, dtype=float)
        r_gen = np.zeros(nt, dtype=float)

    if store_losses:
        sa_loss = np.zeros(nt, dtype=float)
        seep_loss = np.zeros(nt, dtype=float)

    if store_states:
        h_a  = np.zeros(nt + 1)
        h_s  = np.zeros(nt + 1)
        h_r1 = np.zeros(nt + 1)
        h_r2 = np.zeros(nt + 1)
        h_a[0]  = float(max(0.0, h_a0))
        h_s[0]  = float(max(0.0, h_s0))
        h_r1[0] = float(max(0.0, h_r10))
        h_r2[0] = float(max(0.0, h_r20))
    else:
        h_a_c  = float(max(0.0, h_a0))
        h_s_c  = float(max(0.0, h_s0))
        h_r1_c = float(max(0.0, h_r10))
        h_r2_c = float(max(0.0, h_r20))

    for n in range(nt):
        if store_states:
            h_a_c, h_s_c, h_r1_c, h_r2_c = h_a[n], h_s[n], h_r1[n], h_r2[n]

        q_vol = infil_vol = rgen_vol = rout_vol = seep_vol = sa_vol = 0.0
        p = float(p_rate[n])
        etp = float(etp_rate[n])

        for _ in range(nsub):
            # 1) ET sur Ia
            etp_pot = etp * dt_int
            etp_eff = min(etp_pot, h_a_c)
            h_a_c -= etp_eff
            sa_vol += etp_eff

            # 2) Pluie nette après Ia
            h_a_tmp = h_a_c + p * dt_int
            if h_a_tmp < i_a:
                q_n = 0.0
                h_a_c = h_a_tmp
            else:
                q_n = (h_a_tmp - i_a) / dt_int
                h_a_c = i_a
            q_vol += q_n * dt_int

            # 3) Infiltration potentielle HSM
            X = max(1e-12, 1.0 - h_s_c / s)
            Xn = 1.0 / (1.0 / X + k_infiltr * dt_int / s)
            h_s_target = (1.0 - Xn) * s
            infil_pot = (h_s_target - h_s_c) / dt_int

            # 4) Limitation par eau dispo
            water_rate = (q_n + h_r1_c / dt_int) if infil_from_surface else q_n
            water_rate = max(0.0, water_rate)

            infil_n = max(0.0, min(infil_pot, water_rate))
            infil_vol += infil_n * dt_int

            infil_from_rain = min(infil_n, q_n)
            infil_from_hr = max(0.0, infil_n - infil_from_rain)
            if infil_from_surface:
                infil_from_hr = min(infil_from_hr, h_r1_c / dt_int)
            else:
                infil_from_hr = 0.0

            # 5) Sol + seepage
            h_s_tmp = h_s_c + infil_n * dt_int
            if k_seepage > 0:
                h_s_new = h_s_tmp * math.exp(-k_seepage * dt_int)
                seep = h_s_tmp - h_s_new
            else:
                h_s_new = h_s_tmp
                seep = 0.0
            h_s_c = h_s_new
            seep_vol += seep

            # 6) Génération de surface
            r_gen_n = max(q_n - infil_from_rain, 0.0)
            rgen_vol += r_gen_n * dt_int

            # 7) Routage r1 -> r2 (solutions exactes)
            b1 = r_gen_n - infil_from_hr
            h10 = h_r1_c
            if k_runoff1 > 0:
                e1 = math.exp(-k_runoff1 * dt_int)
                h11 = h10 * e1 + (b1 / k_runoff1) * (1.0 - e1)
            else:
                h11 = h10 + b1 * dt_int
            h11 = max(0.0, h11)

            Vout1 = max(0.0, b1 * dt_int + h10 - h11)
            h_r1_c = h11

            b2 = Vout1 / dt_int
            h20 = h_r2_c
            if k_runoff2 > 0:
                e2 = math.exp(-k_runoff2 * dt_int)
                h21 = h20 * e2 + (b2 / k_runoff2) * (1.0 - e2)
            else:
                h21 = h20 + b2 * dt_int
            h21 = max(0.0, h21)

            Vout2 = max(0.0, b2 * dt_int + h20 - h21)
            h_r2_c = h21

            rout_vol += Vout2

        if store_states:
            h_a[n + 1], h_s[n + 1], h_r1[n + 1], h_r2[n + 1] = h_a_c, h_s_c, h_r1_c, h_r2_c

        r_out[n] = rout_vol / dt_obs

        if store_fluxes:
            q[n] = q_vol / dt_obs
            infil[n] = infil_vol / dt_obs
            r_gen[n] = rgen_vol / dt_obs

        if store_losses:
            sa_loss[n] = sa_vol
            seep_loss[n] = seep_vol

    out = {"r_out": r_out}
    if store_states:
        out.update({"h_a": h_a, "h_s": h_s, "h_r1": h_r1, "h_r2": h_r2})
    else:
        out.update({"last_state": (h_a_c, h_s_c, h_r1_c, h_r2_c)})
    if store_fluxes:
        out.update({"q": q, "infil": infil, "r_gen": r_gen})
    if store_losses:
        out.update({"sa_loss": sa_loss, "seep_loss": seep_loss})
    return out


# =========================================================
# Metrics (identiques)
# =========================================================
def _mask_xy(x, y, min_n=2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return None, None
    return x[m], y[m]


def compute_rmse(q_obs, q_mod):
    x, y = _mask_xy(q_obs, q_mod, min_n=2)
    if x is None:
        return np.nan
    d = y - x
    return float(np.sqrt(np.mean(d * d)))


def compute_bias(q_obs, q_mod):
    x, y = _mask_xy(q_obs, q_mod, min_n=2)
    if x is None:
        return np.nan
    return float(np.mean(y - x))


def compute_nse(q_obs, q_mod):
    x, y = _mask_xy(q_obs, q_mod, min_n=3)
    if x is None:
        return np.nan
    den = np.sum((x - np.mean(x)) ** 2)
    if den <= 1e-30:
        return np.nan
    return float(1.0 - np.sum((y - x) ** 2) / den)


def compute_kge(q_obs, q_mod):
    x, y = _mask_xy(q_obs, q_mod, min_n=3)
    if x is None:
        return np.nan
    if np.std(x) < 1e-30 or np.std(y) < 1e-30:
        return np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    alpha = float(np.std(y) / np.std(x))
    mx, my = float(np.mean(x)), float(np.mean(y))
    beta = float(my / mx) if abs(mx) > 1e-30 else np.nan
    if not np.isfinite(beta):
        return np.nan
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def slice_by_dates(time_index: pd.DatetimeIndex, t0: pd.Timestamp, t1: pd.Timestamp):
    mask = (time_index >= t0) & (time_index <= t1)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1


def compute_period_metrics(name, t, P_mm, dt_obs, Qobs_lh, Qmod_lh, t0, t1):
    sl = slice_by_dates(t, t0, t1)
    if sl is None:
        return None, None
    i0, i1 = sl

    qobs = np.asarray(Qobs_lh[i0:i1], float)
    qmod = np.asarray(Qmod_lh[i0:i1], float)
    Ptot = float(np.sum(np.asarray(P_mm[i0:i1], float)))

    Vobs = float(np.sum(np.clip(lh_to_m3s(qobs), 0.0, None)) * dt_obs)
    Vmod = float(np.sum(np.clip(lh_to_m3s(qmod), 0.0, None)) * dt_obs)

    out = dict(
        period_name=name,
        t_start=str(t0),
        t_end=str(t1),
        n_steps=int(i1 - i0),
        P_tot_mm=Ptot,
        Vruiss_obs_m3=Vobs,
        Vruiss_mod_m3=Vmod,
        Vruiss_ratio=(Vmod / Vobs) if Vobs > 1e-12 else np.nan,
        Qmax_obs_LH=float(np.nanmax(qobs)) if np.isfinite(np.nanmax(qobs)) else np.nan,
        Qmax_mod_LH=float(np.nanmax(qmod)) if np.isfinite(np.nanmax(qmod)) else np.nan,
        RMSE_Qruiss_LH=compute_rmse(qobs, qmod),
        BIAS_Qruiss_LH=compute_bias(qobs, qmod),
        NSE_Qruiss=compute_nse(qobs, qmod),
        KGE_Qruiss=compute_kge(qobs, qmod),
    )
    out["Qmax_ratio"] = (out["Qmax_mod_LH"] / out["Qmax_obs_LH"]) if (np.isfinite(out["Qmax_obs_LH"]) and out["Qmax_obs_LH"] > 1e-12) else np.nan
    return out, (i0, i1)


def at_bound(val, lo, hi, tol=1e-3):
    if hi <= lo:
        return False
    return (val >= hi * (1.0 - tol)) or (val <= lo * (1.0 + tol))


def bounds_log10_center_rel(center, rel=0.3, min_positive=1e-30):
    c = float(center)
    r = float(rel)
    lo = max(c * (1.0 - r), min_positive)
    hi = max(c * (1.0 + r), lo * 1.0000001)
    return (math.log10(lo), math.log10(hi))


# =========================================================
# Objectifs (2 étapes) - TOP LEVEL (picklable)
# =========================================================
def objective_stage1_qinf_shape_volume_theta2(theta2_log10, data):
    log10_ki, log10_ks = float(theta2_log10[0]), float(theta2_log10[1])
    b = data["bounds_stage1_log10"]
    if not (b[0][0] <= log10_ki <= b[0][1]): return 1e12
    if not (b[1][0] <= log10_ks <= b[1][1]): return 1e12

    k_infiltr = 10 ** log10_ki
    k_seepage = 10 ** log10_ks

    try:
        res = run_scs_hsm_cascade(
            dt_obs=data["dt_obs"],
            dt_internal=data["dt_internal"],
            p_rate=data["p_rate_calib_full"],
            etp_rate=data["etp_rate_calib_full"],
            i_a=data["i_a"],
            s=data["s"],
            k_infiltr=k_infiltr,
            k_seepage=k_seepage,
            k_runoff1=data["k_runoff1_fixed_stage1"],
            k_runoff2=data["k_runoff2_fixed_stage1"],
            infil_from_surface=data["infil_from_surface"],
            h_a0=0.0, h_s0=0.0, h_r10=0.0, h_r20=0.0,
            store_states=False,
            store_fluxes=True,
            store_losses=True,
        )
    except Exception:
        return 1e12

    A = float(data["A_BV_M2"])
    dt = float(data["dt_obs"])
    i_obj0 = int(data["i_obj0"])
    i_obj1 = int(data["i_obj1"])

    qinf_obs_obj = np.asarray(data["qinf_obs_obj_lh"], float)
    proxy = str(data["qinf_proxy"])

    if proxy == "seep":
        seepV = np.asarray(res["seep_loss"], float)                 # m/step
        qinf_mod_full = m3s_to_lh((seepV / dt) * A)                 # L/h
    else:
        infil_rate = np.asarray(res["infil"], float)                # m/s
        qinf_mod_full = m3s_to_lh(infil_rate * A)                   # L/h

    qinf_mod_obj = np.asarray(qinf_mod_full[i_obj0:i_obj1], float)

    Vobs = float(np.sum(np.clip(lh_to_m3s(qinf_obs_obj), 0.0, None)) * dt)
    Vmod = float(np.sum(np.clip(lh_to_m3s(qinf_mod_obj), 0.0, None)) * dt)
    if Vobs <= 1e-12 or (not np.isfinite(Vobs)) or (not np.isfinite(Vmod)):
        return 1e12
    Jvol = abs(Vmod - Vobs) / Vobs

    eps = float(data.get("QINF_LOG_EPS_LH", 0.1))
    x = np.log(np.clip(qinf_obs_obj, 0.0, None) + eps)
    y = np.log(np.clip(qinf_mod_obj, 0.0, None) + eps)
    Jshape = compute_rmse(x, y)
    if not np.isfinite(Jshape):
        return 1e12

    wV = float(data.get("STAGE1_WVOL", 0.35))
    wS = float(data.get("STAGE1_WSHAPE", 0.65))
    return float(wV * Jvol + wS * Jshape)


def objective_stage2_runoff_theta2(theta2_log10, data):
    log10_kr1, log10_kr2 = float(theta2_log10[0]), float(theta2_log10[1])
    b = data["bounds_stage2_log10"]
    if not (b[0][0] <= log10_kr1 <= b[0][1]): return 1e12
    if not (b[1][0] <= log10_kr2 <= b[1][1]): return 1e12

    k_infiltr = 10 ** float(data["log10_ki_opt"])
    k_seepage = 10 ** float(data["log10_ks_opt"])
    k_runoff1 = 10 ** log10_kr1
    k_runoff2 = 10 ** log10_kr2

    try:
        res = run_scs_hsm_cascade(
            dt_obs=data["dt_obs"],
            dt_internal=data["dt_internal"],
            p_rate=data["p_rate_calib_full"],
            etp_rate=data["etp_rate_calib_full"],
            i_a=data["i_a"],
            s=data["s"],
            k_infiltr=k_infiltr,
            k_seepage=k_seepage,
            k_runoff1=k_runoff1,
            k_runoff2=k_runoff2,
            infil_from_surface=data["infil_from_surface"],
            h_a0=0.0, h_s0=0.0, h_r10=0.0, h_r20=0.0,
            store_states=False,
            store_fluxes=False,
            store_losses=True,
        )
    except Exception:
        return 1e12

    A = float(data["A_BV_M2"])
    dt = float(data["dt_obs"])
    i_obj0 = int(data["i_obj0"])
    i_obj1 = int(data["i_obj1"])

    r_out = np.asarray(res["r_out"], float)              # m/s
    qruiss_mod_full = m3s_to_lh(r_out * A)
    qruiss_mod_obj = np.asarray(qruiss_mod_full[i_obj0:i_obj1], float)
    qruiss_obs_obj = np.asarray(data["qruiss_obs_obj_lh"], float)

    Jruiss = compute_rmse(qruiss_obs_obj, qruiss_mod_obj)
    if not np.isfinite(Jruiss):
        return 1e12

    lam = float(data.get("STAGE2_LAMBDA_QINF", 0.10))
    if lam <= 0:
        return float(Jruiss)

    proxy = str(data.get("qinf_proxy", "seep"))
    if proxy != "seep":
        return float(Jruiss)  # support minimal (comme ton code)

    qinf_obs_obj = np.asarray(data["qinf_obs_obj_lh"], float)
    seepV = np.asarray(res["seep_loss"], float)
    qinf_mod_full = m3s_to_lh((seepV / dt) * A)
    qinf_mod_obj = np.asarray(qinf_mod_full[i_obj0:i_obj1], float)

    eps = float(data.get("QINF_LOG_EPS_LH", 0.1))
    Jinf = compute_rmse(np.log(np.clip(qinf_obs_obj, 0.0, None) + eps),
                        np.log(np.clip(qinf_mod_obj, 0.0, None) + eps))
    if not np.isfinite(Jinf):
        return 1e12
    return float(Jruiss + lam * Jinf)


# =========================================================
# Parallélisation multi-start Powell (picklable)
# =========================================================
def _worker_minimize_powell_generic(objfun, x0, data, bounds_log10, maxiter):
    try:
        res = minimize(
            objfun,
            np.array(x0, float),
            args=(data,),
            method="Powell",
            bounds=bounds_log10,
            options={"maxiter": int(maxiter), "disp": False},
        )
        J = float(res.fun) if np.isfinite(res.fun) else np.inf
        x = np.array(res.x, float) if res.x is not None else None
        if x is None or not np.all(np.isfinite(x)):
            return (np.inf, None)
        return (J, x)
    except Exception:
        return (np.inf, None)


def calibrate_multistart_powell_parallel_generic(objfun, data, bounds_log10,
                                                 n_starts=25, maxiter=100,
                                                 x0_forced=None, n_workers=None,
                                                 verbose=False, random_seed=123):
    rng = np.random.RandomState(int(random_seed))

    def _sample():
        return np.array([rng.uniform(lo, hi) for (lo, hi) in bounds_log10], float)

    starts = []
    if x0_forced is not None:
        starts.append(np.array(x0_forced, float))
        while len(starts) < int(n_starts):
            starts.append(_sample())
    else:
        for _ in range(int(n_starts)):
            starts.append(_sample())

    if n_workers is None:
        cpu = os.cpu_count() or 2
        n_workers = max(1, cpu - 1)

    # Séquentiel (debug)
    if int(n_workers) <= 1:
        best_x, best_J = None, np.inf
        for k, x0 in enumerate(starts, start=1):
            J, x = _worker_minimize_powell_generic(objfun, x0, data, bounds_log10, int(maxiter))
            if verbose:
                print(f"  start {k:02d}/{len(starts)} : J={J:.6e}")
            if J < best_J:
                best_J, best_x = J, x
        return best_x, best_J

    # Parallèle
    best_x, best_J = None, np.inf
    with ProcessPoolExecutor(max_workers=int(n_workers)) as ex:
        futs = [ex.submit(_worker_minimize_powell_generic, objfun, x0, data, bounds_log10, int(maxiter))
                for x0 in starts]
        done = 0
        for fut in as_completed(futs):
            done += 1
            J, x = fut.result()
            if verbose:
                print(f"  start {done:02d}/{len(starts)} : J={J:.6e}")
            if J < best_J:
                best_J, best_x = J, x
    return best_x, best_J


# =========================================================
# Bilans mensuels/annuel (identiques)
# =========================================================
def _safe_sum(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sum(x)) if len(x) else np.nan


def build_monthly_balances(sim_df: pd.DataFrame, dt_s: float):
    df = sim_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    def vol_m3_from_lh(series_lh):
        q_m3s = np.clip(lh_to_m3s(np.asarray(series_lh, float)), 0.0, None)
        return float(np.sum(q_m3s) * dt_s)

    rows = []
    for m, g in df.groupby("month"):
        out = {"month": m,
               "n_steps": int(len(g)),
               "P_tot_mm": _safe_sum(g["P_mm"].to_numpy(float))}

        out["Vruiss_obs_m3"] = vol_m3_from_lh(g["Q_ruiss_obs_LH"])
        out["Vruiss_mod_m3"] = vol_m3_from_lh(g["Q_ruiss_mod_LH"])
        out["Vruiss_ratio"] = (out["Vruiss_mod_m3"] / out["Vruiss_obs_m3"]) if (np.isfinite(out["Vruiss_obs_m3"]) and out["Vruiss_obs_m3"] > 1e-12) else np.nan

        out["Qmax_obs_LH"] = float(np.nanmax(g["Q_ruiss_obs_LH"].to_numpy(float))) if np.isfinite(g["Q_ruiss_obs_LH"]).any() else np.nan
        out["Qmax_mod_LH"] = float(np.nanmax(g["Q_ruiss_mod_LH"].to_numpy(float))) if np.isfinite(g["Q_ruiss_mod_LH"]).any() else np.nan

        out["RMSE_Qruiss_LH"] = compute_rmse(g["Q_ruiss_obs_LH"], g["Q_ruiss_mod_LH"])
        out["BIAS_Qruiss_LH"] = compute_bias(g["Q_ruiss_obs_LH"], g["Q_ruiss_mod_LH"])
        out["NSE_Qruiss"] = compute_nse(g["Q_ruiss_obs_LH"], g["Q_ruiss_mod_LH"])
        out["KGE_Qruiss"] = compute_kge(g["Q_ruiss_obs_LH"], g["Q_ruiss_mod_LH"])

        for c in ["infil_mod_LH", "seep_mod_LH", "qnet_mod_LH", "et_ia_mod_LH", "Q_inf_obs_LH"]:
            if c in g.columns:
                out[f"V_{c}_m3"] = vol_m3_from_lh(g[c])
        rows.append(out)

    dfm = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    if len(dfm):
        ann = {"month": "YEAR",
               "n_steps": int(df["P_mm"].shape[0]),
               "P_tot_mm": _safe_sum(df["P_mm"].to_numpy(float))}
        ann["Vruiss_obs_m3"] = float(np.nansum(dfm["Vruiss_obs_m3"]))
        ann["Vruiss_mod_m3"] = float(np.nansum(dfm["Vruiss_mod_m3"]))
        ann["Vruiss_ratio"] = (ann["Vruiss_mod_m3"] / ann["Vruiss_obs_m3"]) if (np.isfinite(ann["Vruiss_obs_m3"]) and ann["Vruiss_obs_m3"] > 1e-12) else np.nan

        ann["Qmax_obs_LH"] = float(np.nanmax(df["Q_ruiss_obs_LH"].to_numpy(float))) if np.isfinite(df["Q_ruiss_obs_LH"]).any() else np.nan
        ann["Qmax_mod_LH"] = float(np.nanmax(df["Q_ruiss_mod_LH"].to_numpy(float))) if np.isfinite(df["Q_ruiss_mod_LH"]).any() else np.nan

        ann["RMSE_Qruiss_LH"] = compute_rmse(df["Q_ruiss_obs_LH"], df["Q_ruiss_mod_LH"])
        ann["BIAS_Qruiss_LH"] = compute_bias(df["Q_ruiss_obs_LH"], df["Q_ruiss_mod_LH"])
        ann["NSE_Qruiss"] = compute_nse(df["Q_ruiss_obs_LH"], df["Q_ruiss_mod_LH"])
        ann["KGE_Qruiss"] = compute_kge(df["Q_ruiss_obs_LH"], df["Q_ruiss_mod_LH"])

        for c in ["infil_mod_LH", "seep_mod_LH", "qnet_mod_LH", "et_ia_mod_LH", "Q_inf_obs_LH"]:
            col = f"V_{c}_m3"
            if col in dfm.columns:
                ann[col] = float(np.nansum(dfm[col]))

        dfm = pd.concat([dfm, pd.DataFrame([ann])], ignore_index=True)

    return dfm


# =========================================================
# CONFIG 
# =========================================================

def get_config():
    try:
        root = Path(__file__).resolve().parents[1]  # CSR_Lyon
    except NameError:
        root = Path.cwd()

    cfg = dict(
        ROOT=root,
        DATA_DIR=root / "02_Data",
        OUT_BASE=root / "03_Plots",

        # exports
        EXPORT_DIAGNOSTICS=True,
        EXPORT_STATES=True,
        EXPORT_INTERNAL_FLUXES=True,

        # pilotage
        RUN_SPLITS=["H1->H2"],
        # UN_SPLITS=["H1->H2", "H2->H1"],
        DO_CALIBRATION=True,
        RANDOM_SEED=123,
        N_WORKERS=none,              # none ou en debug: 1
        VERBOSE_STARTS=False,

        # multi-start
        N_STARTS_STAGE1=25, MAXITER_STAGE1=100,
        N_STARTS_STAGE2=15, MAXITER_STAGE2=100,

        # CSR fixe
        A_BV_M2=94.0,
        I_A_FIXED=0.002,
        S_FIXED=0.13,
        PREFERRED_DT_INTERNAL=60.0,
        INFIL_FROM_SURFACE=True,

        # ETP synth
        ETP_MM_DAY=1.5,
        ETP_START_H=8,
        ETP_END_H=20,

        # proxy Qinf
        QINF_PROXY="seep",  # "seep" ou "infil"

        # pondérations / eps (non param)
        STAGE1_WVOL=0.35,
        STAGE1_WSHAPE=0.65,
        STAGE2_LAMBDA_QINF=0.10,
        QINF_LOG_EPS_LH=0.1,

        # bornes params
        KINF_MIN_MM_H=0.2,
        KINF_MAX_MM_H=5.0,
        KSEEP_CENTER_S_1=9.62e-5,
        KSEEP_REL=0.50,
        KR1_CENTER_S_1=2.89e-3,
        KR2_CENTER_S_1=1.44e-3,
        KR1_REL=0.60,
        KR2_REL=0.60,

        # période hydro
        HYDRO_START=pd.Timestamp("2022-10-01 00:00:00"),
        HYDRO_END=pd.Timestamp("2023-09-30 23:59:59"),

        # splits
        SPLITS={
            "H1->H2": dict(
                name="calib_H1_valid_H2",
                calib_start=pd.Timestamp("2022-10-01 00:00:00"),
                calib_end=pd.Timestamp("2023-03-31 23:59:59"),
                valid_start=pd.Timestamp("2023-04-01 00:00:00"),
                valid_end=pd.Timestamp("2023-09-30 23:59:59"),
            ),
            "H2->H1": dict(
                name="calib_H2_valid_H1",
                calib_start=pd.Timestamp("2023-04-01 00:00:00"),
                calib_end=pd.Timestamp("2023-09-30 23:59:59"),
                valid_start=pd.Timestamp("2022-10-01 00:00:00"),
                valid_end=pd.Timestamp("2023-03-31 23:59:59"),
            ),
        },
        WARMUP_MONTHS=1,

        # fichier continu
        CONTINUOUS_CSV_NAME="Donnees_serie_complete_2022-2024_corrigee_AS.csv",
        CSV_SEP=";",
    )
    return cfg


# =========================================================
# Helpers RUN
# =========================================================
def make_bounds(cfg):
    ki_lo = infil_mm_h_to_m_s(cfg["KINF_MIN_MM_H"])
    ki_hi = infil_mm_h_to_m_s(cfg["KINF_MAX_MM_H"])
    bounds_stage1_log10 = [
        (math.log10(ki_lo), math.log10(ki_hi)),
        bounds_log10_center_rel(cfg["KSEEP_CENTER_S_1"], cfg["KSEEP_REL"]),
    ]
    bounds_stage2_log10 = [
        bounds_log10_center_rel(cfg["KR1_CENTER_S_1"], cfg["KR1_REL"]),
        bounds_log10_center_rel(cfg["KR2_CENTER_S_1"], cfg["KR2_REL"]),
    ]
    return (ki_lo, ki_hi, bounds_stage1_log10, bounds_stage2_log10)


def pack_stage_common(cfg, dt_obs, dt_internal, p_rate, etp_rate,
                      i0, i1, o0, o1, i_obj0, i_obj1,
                      qruiss_obs_lh, qinf_obs_lh):
    return dict(
        dt_obs=float(dt_obs),
        dt_internal=float(dt_internal),
        i_a=float(cfg["I_A_FIXED"]),
        s=float(cfg["S_FIXED"]),
        A_BV_M2=float(cfg["A_BV_M2"]),
        infil_from_surface=bool(cfg["INFIL_FROM_SURFACE"]),
        p_rate_calib_full=np.asarray(p_rate[i0:i1], float),
        etp_rate_calib_full=np.asarray(etp_rate[i0:i1], float),
        qruiss_obs_obj_lh=np.asarray(qruiss_obs_lh[o0:o1], float),
        qinf_obs_obj_lh=np.asarray(qinf_obs_lh[o0:o1], float),
        i_obj0=int(i_obj0),
        i_obj1=int(i_obj1),
        qinf_proxy=str(cfg["QINF_PROXY"]),
        STAGE1_WVOL=float(cfg["STAGE1_WVOL"]),
        STAGE1_WSHAPE=float(cfg["STAGE1_WSHAPE"]),
        STAGE2_LAMBDA_QINF=float(cfg["STAGE2_LAMBDA_QINF"]),
        QINF_LOG_EPS_LH=float(cfg["QINF_LOG_EPS_LH"]),
    )


def assemble_full_sim(t, P_mm, Qruiss_obs_lh, Qinf_obs_lh, dt_obs, A_BV_M2,
                      i0, i1, j0, j1, calib_start, warmup_end, valid_start, valid_end,
                      res_cal_full, res_val,
                      export_diag, export_states, export_fluxes):

    nT = len(t)
    period = np.array(["all"] * nT, dtype=object)

    period[i0:i1] = "calib"
    period[j0:j1] = "valid"
    sl_warm = slice_by_dates(t, calib_start, warmup_end)
    if sl_warm is not None:
        w0, w1 = sl_warm
        period[w0:w1] = "warmup"

    r_out_full = np.full(nT, np.nan, float)
    r_out_full[i0:i1] = res_cal_full["r_out"]
    r_out_full[j0:j1] = res_val["r_out"]

    Qruiss_mod_lh = m3s_to_lh(np.asarray(r_out_full, float) * float(A_BV_M2))

    # internes
    qnet_mod_lh = infil_mod_lh = seep_mod_lh = et_ia_mod_lh = np.full(nT, np.nan, float)

    if export_diag and export_fluxes:
        q_all = np.full(nT, np.nan, float)
        infil_all = np.full(nT, np.nan, float)
        seepV_all = np.full(nT, np.nan, float)
        saV_all = np.full(nT, np.nan, float)

        for (a, b, res) in [(i0, i1, res_cal_full), (j0, j1, res_val)]:
            if "q" in res: q_all[a:b] = res["q"]
            if "infil" in res: infil_all[a:b] = res["infil"]
            if "seep_loss" in res: seepV_all[a:b] = res["seep_loss"]
            if "sa_loss" in res: saV_all[a:b] = res["sa_loss"]

        qnet_mod_lh = m3s_to_lh(q_all * float(A_BV_M2)) if np.isfinite(q_all).any() else qnet_mod_lh
        infil_mod_lh = m3s_to_lh(infil_all * float(A_BV_M2)) if np.isfinite(infil_all).any() else infil_mod_lh

        seep_rate_mps = seepV_all / float(dt_obs)
        seep_mod_lh = m3s_to_lh(seep_rate_mps * float(A_BV_M2)) if np.isfinite(seep_rate_mps).any() else seep_mod_lh

        sa_rate_mps = saV_all / float(dt_obs)
        et_ia_mod_lh = m3s_to_lh(sa_rate_mps * float(A_BV_M2)) if np.isfinite(sa_rate_mps).any() else et_ia_mod_lh

    # états
    states = {}
    if export_diag and export_states and ("h_a" in res_cal_full) and ("h_a" in res_val):
        h_a_all = np.full(nT, np.nan, float)
        h_s_all = np.full(nT, np.nan, float)
        h_r1_all = np.full(nT, np.nan, float)
        h_r2_all = np.full(nT, np.nan, float)

        h_a_all[i0:i1] = res_cal_full["h_a"][:-1]
        h_s_all[i0:i1] = res_cal_full["h_s"][:-1]
        h_r1_all[i0:i1] = res_cal_full["h_r1"][:-1]
        h_r2_all[i0:i1] = res_cal_full["h_r2"][:-1]

        h_a_all[j0:j1] = res_val["h_a"][:-1]
        h_s_all[j0:j1] = res_val["h_s"][:-1]
        h_r1_all[j0:j1] = res_val["h_r1"][:-1]
        h_r2_all[j0:j1] = res_val["h_r2"][:-1]

        states = dict(h_a_m=h_a_all, h_s_m=h_s_all, h_r1_m=h_r1_all, h_r2_m=h_r2_all)

    sim_df = pd.DataFrame({
        "date": t.astype(str),
        "period": period,
        "P_mm": P_mm,
        "Q_ruiss_obs_LH": Qruiss_obs_lh,
        "Q_ruiss_mod_LH": Qruiss_mod_lh,
        "Q_inf_obs_LH": Qinf_obs_lh,
        "qnet_mod_LH": qnet_mod_lh,
        "infil_mod_LH": infil_mod_lh,
        "seep_mod_LH": seep_mod_lh,
        "et_ia_mod_LH": et_ia_mod_lh,
        **states
    })

    return sim_df, Qruiss_mod_lh, seep_mod_lh, infil_mod_lh


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = get_config()
    cfg["OUT_BASE"].mkdir(parents=True, exist_ok=True)

    ki_lo, ki_hi, bounds_stage1_log10, bounds_stage2_log10 = make_bounds(cfg)

    # Fixe pour stage1 (juste proxy)
    KR1_FIXED_STAGE1 = float(cfg["KR1_CENTER_S_1"])
    KR2_FIXED_STAGE1 = float(cfg["KR2_CENTER_S_1"])

    print("=== RUN CONTINU 2022-2023 split 6 mois + warmup 1 mois + 2 étapes ===")
    print(f"ROOT         : {cfg['ROOT']}")
    print(f"Période hydro : {cfg['HYDRO_START']} -> {cfg['HYDRO_END']}")
    print(f"Warm-up       : {cfg['WARMUP_MONTHS']} mois")
    print(f"Proxy Qinf    : {'seep_mod_LH' if cfg['QINF_PROXY']=='seep' else 'infil_mod_LH'}")
    print(f"OUT_BASE      : {cfg['OUT_BASE']}")
    print("=======================================================================\n")

    # 1) Lire continu + filtrer période hydro
    csv_continu = cfg["DATA_DIR"] / cfg["CONTINUOUS_CSV_NAME"]
    t_all, P_all_mm, Qinf_all_lh, Qruiss_all_lh, dt_obs = read_continuous_csv(csv_continu, sep=cfg["CSV_SEP"])

    mask = (t_all >= cfg["HYDRO_START"]) & (t_all <= cfg["HYDRO_END"])
    if mask.sum() < 10:
        raise RuntimeError("Pas assez de données sur la période hydro 2022-2023.")

    t = pd.DatetimeIndex(t_all[mask])
    P_mm = np.asarray(P_all_mm[mask], float)
    Qinf_obs_lh = np.asarray(Qinf_all_lh[mask], float)
    Qruiss_obs_lh = np.asarray(Qruiss_all_lh[mask], float)

    dt_internal = choose_dt_internal(dt_obs, preferred=cfg["PREFERRED_DT_INTERNAL"])
    p_rate = mm_per_step_to_mps(P_mm, dt_obs)
    etp_rate = build_constant_daytime_etp_rate(
        t, etp_mm_per_day=cfg["ETP_MM_DAY"], start_hour=cfg["ETP_START_H"], end_hour=cfg["ETP_END_H"]
    )

    # 2) Boucle splits
    for key in cfg["RUN_SPLITS"]:
        if key not in cfg["SPLITS"]:
            print(f"[WARN] Split inconnu: {key} (skip)")
            continue

        sp = cfg["SPLITS"][key]
        split_name = sp["name"]
        CALIB_START, CALIB_END = sp["calib_start"], sp["calib_end"]
        VALID_START, VALID_END = sp["valid_start"], sp["valid_end"]
        WARMUP_END = (CALIB_START + pd.DateOffset(months=cfg["WARMUP_MONTHS"]))

        out_dir = cfg["OUT_BASE"] / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== SPLIT: {split_name} ===")
        print(f"Calage FULL : {CALIB_START} -> {CALIB_END}")
        print(f"Warm-up     : {CALIB_START} -> {WARMUP_END} (exclu objectif)")
        print(f"Calage OBJ  : {WARMUP_END} -> {CALIB_END}")
        print(f"Validation  : {VALID_START} -> {VALID_END}")
        print(f"out_dir     : {out_dir}\n")

        sl_cal_full = slice_by_dates(t, CALIB_START, CALIB_END)
        sl_val = slice_by_dates(t, VALID_START, VALID_END)
        sl_cal_obj = slice_by_dates(t, WARMUP_END, CALIB_END)
        if sl_cal_full is None or sl_val is None or sl_cal_obj is None:
            print("[SKIP] Fenêtre calage/validation introuvable.")
            continue

        i0, i1 = sl_cal_full
        j0, j1 = sl_val
        o0, o1 = sl_cal_obj

        i_obj0 = int(o0 - i0)
        i_obj1 = int(o1 - i0)
        if i_obj1 - i_obj0 < 3:
            print("[SKIP] Fenêtre objectif trop courte.")
            continue

        if not cfg["DO_CALIBRATION"]:
            print("[SKIP] DO_CALIBRATION=False")
            continue

        # Pack données communes (picklable)
        data_common = pack_stage_common(
            cfg, dt_obs, dt_internal, p_rate, etp_rate,
            i0, i1, o0, o1, i_obj0, i_obj1,
            Qruiss_obs_lh, Qinf_obs_lh
        )

        # =========================================================
        # ETAPE 1
        # =========================================================
        print("--- ETAPE 1 : Qinf (volume + forme) -> optimise ki + ks ---")
        data_stage1 = dict(data_common)
        data_stage1["bounds_stage1_log10"] = bounds_stage1_log10
        data_stage1["k_runoff1_fixed_stage1"] = KR1_FIXED_STAGE1
        data_stage1["k_runoff2_fixed_stage1"] = KR2_FIXED_STAGE1

        theta1_opt, J1 = calibrate_multistart_powell_parallel_generic(
            objective_stage1_qinf_shape_volume_theta2,
            data_stage1,
            bounds_stage1_log10,
            n_starts=int(cfg["N_STARTS_STAGE1"]),
            maxiter=int(cfg["MAXITER_STAGE1"]),
            x0_forced=None,
            n_workers=cfg["N_WORKERS"],
            verbose=cfg["VERBOSE_STARTS"],
            random_seed=int(cfg["RANDOM_SEED"]),
        )

        if theta1_opt is None or not np.all(np.isfinite(theta1_opt)):
            print("[SKIP] ETAPE 1 échouée (theta1_opt invalide).")
            continue

        log10_ki_opt, log10_ks_opt = float(theta1_opt[0]), float(theta1_opt[1])
        k_infiltr_opt = 10 ** log10_ki_opt
        k_seepage_opt = 10 ** log10_ks_opt

        print(f"  [OK] Stage1 J={J1:.6e} | ki={k_infiltr_opt:.3e} m/s ({k_infiltr_opt*3600*1000:.3f} mm/h) | ks={k_seepage_opt:.3e} s^-1")

        # =========================================================
        # ETAPE 2
        # =========================================================
        print("\n--- ETAPE 2 : Qruiss (RMSE) + laisse Qinf -> optimise kr1 + kr2 ---")
        data_stage2 = dict(data_common)
        data_stage2["bounds_stage2_log10"] = bounds_stage2_log10
        data_stage2["log10_ki_opt"] = log10_ki_opt
        data_stage2["log10_ks_opt"] = log10_ks_opt

        theta2_opt, J2 = calibrate_multistart_powell_parallel_generic(
            objective_stage2_runoff_theta2,
            data_stage2,
            bounds_stage2_log10,
            n_starts=int(cfg["N_STARTS_STAGE2"]),
            maxiter=int(cfg["MAXITER_STAGE2"]),
            x0_forced=None,
            n_workers=cfg["N_WORKERS"],
            verbose=cfg["VERBOSE_STARTS"],
            random_seed=int(cfg["RANDOM_SEED"]) + 7,
        )

        if theta2_opt is None or not np.all(np.isfinite(theta2_opt)):
            print("[SKIP] ETAPE 2 échouée (theta2_opt invalide).")
            continue

        log10_kr1_opt, log10_kr2_opt = float(theta2_opt[0]), float(theta2_opt[1])
        k_runoff1_opt = 10 ** log10_kr1_opt
        k_runoff2_opt = 10 ** log10_kr2_opt
        print(f"  [OK] Stage2 J={J2:.6e} | kr1={k_runoff1_opt:.3e} s^-1 | kr2={k_runoff2_opt:.3e} s^-1\n")

        # =========================================================
        # Resimulation calage FULL (CI=0) + CI fin calage -> validation
        # =========================================================
        res_cal_full = run_scs_hsm_cascade(
            dt_obs=dt_obs,
            dt_internal=dt_internal,
            p_rate=p_rate[i0:i1],
            etp_rate=etp_rate[i0:i1],
            i_a=cfg["I_A_FIXED"],
            s=cfg["S_FIXED"],
            k_infiltr=k_infiltr_opt,
            k_seepage=k_seepage_opt,
            k_runoff1=k_runoff1_opt,
            k_runoff2=k_runoff2_opt,
            infil_from_surface=cfg["INFIL_FROM_SURFACE"],
            h_a0=0.0, h_s0=0.0, h_r10=0.0, h_r20=0.0,
            store_states=bool(cfg["EXPORT_DIAGNOSTICS"] and cfg["EXPORT_STATES"]),
            store_fluxes=bool(cfg["EXPORT_DIAGNOSTICS"] and cfg["EXPORT_INTERNAL_FLUXES"]),
            store_losses=bool(cfg["EXPORT_DIAGNOSTICS"] and cfg["EXPORT_INTERNAL_FLUXES"]),
        )

        if "h_a" in res_cal_full:
            h_a_end = float(res_cal_full["h_a"][-1])
            h_s_end = float(res_cal_full["h_s"][-1])
            h_r1_end = float(res_cal_full["h_r1"][-1])
            h_r2_end = float(res_cal_full["h_r2"][-1])
        else:
            (h_a_end, h_s_end, h_r1_end, h_r2_end) = res_cal_full["last_state"]

        res_val = run_scs_hsm_cascade(
            dt_obs=dt_obs,
            dt_internal=dt_internal,
            p_rate=p_rate[j0:j1],
            etp_rate=etp_rate[j0:j1],
            i_a=cfg["I_A_FIXED"],
            s=cfg["S_FIXED"],
            k_infiltr=k_infiltr_opt,
            k_seepage=k_seepage_opt,
            k_runoff1=k_runoff1_opt,
            k_runoff2=k_runoff2_opt,
            infil_from_surface=cfg["INFIL_FROM_SURFACE"],
            h_a0=h_a_end, h_s0=h_s_end, h_r10=h_r1_end, h_r20=h_r2_end,
            store_states=bool(cfg["EXPORT_DIAGNOSTICS"] and cfg["EXPORT_STATES"]),
            store_fluxes=bool(cfg["EXPORT_DIAGNOSTICS"] and cfg["EXPORT_INTERNAL_FLUXES"]),
            store_losses=bool(cfg["EXPORT_DIAGNOSTICS"] and cfg["EXPORT_INTERNAL_FLUXES"]),
        )

        # =========================================================
        # Assemble full simulation + exports
        # =========================================================
        sim_df, Qruiss_mod_lh, seep_mod_lh, infil_mod_lh = assemble_full_sim(
            t=t, P_mm=P_mm, Qruiss_obs_lh=Qruiss_obs_lh, Qinf_obs_lh=Qinf_obs_lh,
            dt_obs=dt_obs, A_BV_M2=cfg["A_BV_M2"],
            i0=i0, i1=i1, j0=j0, j1=j1,
            calib_start=CALIB_START, warmup_end=WARMUP_END,
            valid_start=VALID_START, valid_end=VALID_END,
            res_cal_full=res_cal_full, res_val=res_val,
            export_diag=cfg["EXPORT_DIAGNOSTICS"],
            export_states=cfg["EXPORT_STATES"],
            export_fluxes=cfg["EXPORT_INTERNAL_FLUXES"],
        )

        # Metrics périodes
        metrics_cal_full, _ = compute_period_metrics("CALIB_FULL", t, P_mm, dt_obs, Qruiss_obs_lh, Qruiss_mod_lh, CALIB_START, CALIB_END)
        metrics_cal_obj, _ = compute_period_metrics("CALIB_OBJ", t, P_mm, dt_obs, Qruiss_obs_lh, Qruiss_mod_lh, WARMUP_END, CALIB_END)
        metrics_val, _ = compute_period_metrics("VALID", t, P_mm, dt_obs, Qruiss_obs_lh, Qruiss_mod_lh, VALID_START, VALID_END)

        # Diagnostic volume Qinf final sur OBJ (comme ton code)
        qinf_mod_lh_final = seep_mod_lh if cfg["QINF_PROXY"] == "seep" else infil_mod_lh
        Vinf_obs = float(np.sum(np.clip(lh_to_m3s(Qinf_obs_lh[o0:o1]), 0.0, None)) * dt_obs)
        Vinf_mod = float(np.sum(np.clip(lh_to_m3s(qinf_mod_lh_final[o0:o1]), 0.0, None)) * dt_obs)
        vol_rel_err = abs(Vinf_mod - Vinf_obs) / Vinf_obs if Vinf_obs > 1e-12 else np.nan

        # Export params
        ki_at_bound = at_bound(k_infiltr_opt, 10 ** bounds_stage1_log10[0][0], 10 ** bounds_stage1_log10[0][1])
        ks_at_bound = at_bound(k_seepage_opt, 10 ** bounds_stage1_log10[1][0], 10 ** bounds_stage1_log10[1][1])
        kr1_at_bound = at_bound(k_runoff1_opt, 10 ** bounds_stage2_log10[0][0], 10 ** bounds_stage2_log10[0][1])
        kr2_at_bound = at_bound(k_runoff2_opt, 10 ** bounds_stage2_log10[1][0], 10 ** bounds_stage2_log10[1][1])

        params_df = pd.DataFrame([{
            "split_name": split_name,
            "hydro_start": str(cfg["HYDRO_START"]),
            "hydro_end": str(cfg["HYDRO_END"]),
            "calib_start": str(CALIB_START),
            "calib_end": str(CALIB_END),
            "warmup_end": str(WARMUP_END),
            "valid_start": str(VALID_START),
            "valid_end": str(VALID_END),
            "dt_obs_s": float(dt_obs),
            "dt_internal_s": float(dt_internal),
            "A_BV_M2": float(cfg["A_BV_M2"]),
            "Ia_m": float(cfg["I_A_FIXED"]),
            "S_m": float(cfg["S_FIXED"]),
            "infil_from_surface": bool(cfg["INFIL_FROM_SURFACE"]),
            "qinf_proxy": "seep_mod_LH" if cfg["QINF_PROXY"] == "seep" else "infil_mod_LH",
            "k_infiltr_m_s": float(k_infiltr_opt),
            "k_infiltr_mm_h": float(k_infiltr_opt * 3600 * 1000),
            "k_seepage_s_1": float(k_seepage_opt),
            "k_runoff1_s_1": float(k_runoff1_opt),
            "k_runoff2_s_1": float(k_runoff2_opt),
            "Stage1_J": float(J1) if np.isfinite(J1) else np.nan,
            "Stage1_vol_rel_err_final": float(vol_rel_err) if np.isfinite(vol_rel_err) else np.nan,
            "Stage2_J": float(J2) if np.isfinite(J2) else np.nan,
            "STAGE1_WVOL": float(cfg["STAGE1_WVOL"]),
            "STAGE1_WSHAPE": float(cfg["STAGE1_WSHAPE"]),
            "STAGE2_LAMBDA_QINF": float(cfg["STAGE2_LAMBDA_QINF"]),
            "QINF_LOG_EPS_LH": float(cfg["QINF_LOG_EPS_LH"]),
            "bounds_k_infiltr_mm_h_lo": float(ki_lo * 3600 * 1000),
            "bounds_k_infiltr_mm_h_hi": float(ki_hi * 3600 * 1000),
            "bounds_log10_ks_lo": float(bounds_stage1_log10[1][0]),
            "bounds_log10_ks_hi": float(bounds_stage1_log10[1][1]),
            "bounds_log10_kr1_lo": float(bounds_stage2_log10[0][0]),
            "bounds_log10_kr1_hi": float(bounds_stage2_log10[0][1]),
            "bounds_log10_kr2_lo": float(bounds_stage2_log10[1][0]),
            "bounds_log10_kr2_hi": float(bounds_stage2_log10[1][1]),
            "ki_at_bound": bool(ki_at_bound),
            "ks_at_bound": bool(ks_at_bound),
            "kr1_at_bound": bool(kr1_at_bound),
            "kr2_at_bound": bool(kr2_at_bound),
            "RANDOM_SEED": int(cfg["RANDOM_SEED"]),
            "N_WORKERS": int(cfg["N_WORKERS"]) if cfg["N_WORKERS"] is not None else -1,
            "N_STARTS_STAGE1": int(cfg["N_STARTS_STAGE1"]),
            "MAXITER_STAGE1": int(cfg["MAXITER_STAGE1"]),
            "N_STARTS_STAGE2": int(cfg["N_STARTS_STAGE2"]),
            "MAXITER_STAGE2": int(cfg["MAXITER_STAGE2"]),
            "EXPORT_DIAGNOSTICS": bool(cfg["EXPORT_DIAGNOSTICS"]),
            "EXPORT_STATES": bool(cfg["EXPORT_STATES"]),
            "EXPORT_INTERNAL_FLUXES": bool(cfg["EXPORT_INTERNAL_FLUXES"]),
        }])
        (out_dir / "params.xlsx").write_bytes(b"")  # ensure file unlocked on some FS (safe no-op)
        params_df.to_excel(out_dir / "params.xlsx", index=False)

        # Export metrics periods
        pd.DataFrame([m for m in [metrics_cal_full, metrics_cal_obj, metrics_val] if m is not None]) \
          .to_excel(out_dir / "metrics_periods.xlsx", index=False)

        # Export simulation
        sim_df.to_csv(out_dir / "simulation.csv", sep=";", index=False)

        # Export monthly
        monthly = build_monthly_balances(sim_df, dt_s=float(dt_obs))
        monthly.to_excel(out_dir / "metrics_monthly.xlsx", index=False)

        # Logs
        print(f"[OK] params          : {out_dir / 'params.xlsx'}")
        print(f"[OK] metrics_periods : {out_dir / 'metrics_periods.xlsx'}")
        print(f"[OK] metrics_monthly : {out_dir / 'metrics_monthly.xlsx'}")
        print(f"[OK] simulation      : {out_dir / 'simulation.csv'}")
        if metrics_cal_obj is not None and metrics_val is not None:
            print("  -- Résumé métriques périodes --")
            print(f"  CALIB_OBJ: RMSE={metrics_cal_obj['RMSE_Qruiss_LH']:.2f} | NSE={metrics_cal_obj['NSE_Qruiss']:.2f} | KGE={metrics_cal_obj['KGE_Qruiss']:.2f} | Vratio={metrics_cal_obj['Vruiss_ratio']:.2f}")
            print(f"  VALID    : RMSE={metrics_val['RMSE_Qruiss_LH']:.2f} | NSE={metrics_val['NSE_Qruiss']:.2f} | KGE={metrics_val['KGE_Qruiss']:.2f} | Vratio={metrics_val['Vruiss_ratio']:.2f}")

    print("\n[FIN] RUN terminé pour tous les splits demandés.")


if __name__ == "__main__":
    # IMPORTANT Windows: indispensable pour éviter un spawn infini lors du ProcessPool.
    main()
