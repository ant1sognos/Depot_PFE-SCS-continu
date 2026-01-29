# -*- coding: utf-8 -*-
"""
SCS-HSM continu — VERSION CASCADE (r1 -> r2) + SUBSTEPPING
APPLIQUÉ AU PARKING CSR (calage évènementiel)
-----------------------------------------------------------

Entrée canonique CSR :
    - Date                : datetime
    - Hauteur_de_pluie_mm : pluie (mm/pas)
    - Q_inf_LH            : débit infiltré (L/h) (optionnel)
    - Q_ruiss_LH          : débit ruisselé (L/h) (Q_obs)

Compatibilité anciens exports :
    - date / dateP à la place de Date
    - P_mm à la place de Hauteur_de_pluie_mm
    - Q_ls (L/s) à la place de Q_ruiss_LH (L/h)

Modèle :
    - h_a  : abstraction Ia
    - h_s  : sol (capacité S)
    - h_r1 : réservoir ruissellement 1
    - h_r2 : réservoir ruissellement 2
    - substepping dt_int : intégration interne plus fine
    - cascade r1->r2 : solution exacte du réservoir linéaire à entrée constante sur dt_int

Calage :
    - theta = [log10(k_infiltr), log10(k_runoff1), log10(k_runoff2), log10(k_seepage)]
    - critère : RMSE sur Q_ruiss (L/h) 

Sorties :
    - Figures dans ../03_Plots/Parking_CSR_CASCADE/<event_name>/
"""

from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["agg.path.chunksize"] = 10000


# =========================================================
# Conversions (unités)
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
# ETP synthétique 
# =========================================================
def build_constant_daytime_etp_rate(
    time_index: pd.DatetimeIndex,
    etp_mm_per_day: float = 2.0,
    start_hour: int = 8,
    end_hour: int = 20,
) -> np.ndarray:
    """
    ETP journalière répartie uniformément entre start_hour et end_hour (0 sinon).
    Sortie en m/s.
    """
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
# Lecture event CSV
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]

def _infer_dayfirst(date_series: pd.Series) -> bool:
    s = date_series.astype(str).dropna().head(25).tolist()
    if not s:
        return False
    slash_like = sum(bool(re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", x)) for x in s)
    iso_like   = sum(bool(re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", x)) for x in s)
    return slash_like > iso_like

def read_parking_event_csv(csv_event_rel: str, sep: str = ";"):
    """
    Retourne :
        time_index, P_mm, Q_inf_LH_or_None, Q_ruiss_LH, dt_sec
    """
    csv_path = BASE_DIR / "02_Data" / Path(csv_event_rel)
    if not csv_path.exists():
        raise FileNotFoundError(f"Event CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path, sep=sep, na_values=["NA", "NaN", "", -9999, -9999.0])

    # Date
    date_col = next((c for c in ["Date", "date", "dateP"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"Colonne date introuvable dans {csv_path.name}. Colonnes présentes={list(df.columns)}")

    dayfirst = _infer_dayfirst(df[date_col])
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="raise")
    df = df.sort_values(date_col).reset_index(drop=True)
    time_index = pd.DatetimeIndex(df[date_col])

    diffs = time_index.to_series().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        raise ValueError(f"Série trop courte pour inférer dt: {csv_path.name}")
    dt_sec = float(np.median(diffs))

    # Pluie
    p_col = "Hauteur_de_pluie_mm" if "Hauteur_de_pluie_mm" in df.columns else ("P_mm" if "P_mm" in df.columns else None)
    if p_col is None:
        raise ValueError(f"Colonne pluie introuvable dans {csv_path.name}. Attendu Hauteur_de_pluie_mm ou P_mm.")
    P_mm = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0).to_numpy(float)
    P_mm = np.clip(P_mm, 0.0, None)

    # Q ruiss obs
    if "Q_ruiss_LH" in df.columns:
        Q_ruiss_LH = pd.to_numeric(df["Q_ruiss_LH"], errors="coerce").fillna(0.0).to_numpy(float)
        Q_ruiss_LH = np.clip(Q_ruiss_LH, 0.0, None)
    elif "Q_ls" in df.columns:
        q_ls = pd.to_numeric(df["Q_ls"], errors="coerce").fillna(0.0).to_numpy(float)
        q_ls = np.clip(q_ls, 0.0, None)
        Q_ruiss_LH = m3s_to_lh(q_ls / 1000.0)  # L/s -> m³/s -> L/h
    else:
        raise ValueError(f"Colonne Q obs introuvable dans {csv_path.name}. Attendu Q_ruiss_LH ou Q_ls.")

    # Q inf obs (optionnel)
    Q_inf_LH = None
    if "Q_inf_LH" in df.columns:
        Q_inf_LH = pd.to_numeric(df["Q_inf_LH"], errors="coerce").fillna(0.0).to_numpy(float)
        Q_inf_LH = np.clip(Q_inf_LH, 0.0, None)

    return time_index, P_mm, Q_inf_LH, Q_ruiss_LH, dt_sec


# =========================================================
# Modèle SCS-HSM CASCADE + SUBSTEPPING
# =========================================================
def run_scs_hsm_cascade(
    dt_obs: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray,
    i_a: float,
    s: float,
    k_infiltr: float,   # m/s
    k_seepage: float,   # s^-1
    k_runoff1: float,   # s^-1
    k_runoff2: float,   # s^-1
    dt_internal: float | None = None,
    infil_from_surface: bool = True,
) -> dict:
    """
    Substepping sur dt_int, cascade r1->r2 (réservoirs linéaires exacts).
    Sorties au pas dt_obs (m/s pour q/infil/r_gen/r_out ; m/pas pour sa_loss/seep_loss).
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
        raise ValueError(f"dt_internal doit diviser dt_obs. Ici dt_obs={dt_obs}, dt_internal={dt_internal}")

    # Etats (m)
    h_a  = np.zeros(nt + 1)
    h_s  = np.zeros(nt + 1)
    h_r1 = np.zeros(nt + 1)
    h_r2 = np.zeros(nt + 1)

    # Flux au pas obs
    q = np.zeros(nt)          # m/s
    infil = np.zeros(nt)      # m/s
    r_gen = np.zeros(nt)      # m/s
    r_out = np.zeros(nt)      # m/s (sortie r2)
    sa_loss = np.zeros(nt)    # m / pas obs (ET Ia)
    seep_loss = np.zeros(nt)  # m / pas obs

    for n in range(nt):
        h_a_c  = h_a[n]
        h_s_c  = h_s[n]
        h_r1_c = h_r1[n]
        h_r2_c = h_r2[n]

        q_vol = infil_vol = rgen_vol = rout_vol = seep_vol = sa_vol = 0.0

        p = float(p_rate[n])
        etp = float(etp_rate[n])

        for _ in range(nsub):
            # 1) ET sur Ia
            etp_eff = min(etp * dt_int, h_a_c)
            h_a_c -= etp_eff
            sa_vol += etp_eff

            # 2) pluie nette après Ia
            h_a_tmp = h_a_c + p * dt_int
            if h_a_tmp < i_a:
                q_n = 0.0
                h_a_c = h_a_tmp
            else:
                q_n = (h_a_tmp - i_a) / dt_int
                h_a_c = i_a
            q_vol += q_n * dt_int

            # 3) infiltration potentielle HSM
            X = max(1e-12, 1.0 - h_s_c / s)
            Xn = 1.0 / (1.0 / X + k_infiltr * dt_int / s)
            h_s_target = (1.0 - Xn) * s
            infil_pot = (h_s_target - h_s_c) / dt_int  # m/s

            # 4) limitation par eau dispo
            water_rate = q_n + (h_r1_c / dt_int if infil_from_surface else 0.0)
            water_rate = max(water_rate, 0.0)

            infil_n = max(0.0, min(infil_pot, water_rate))
            infil_vol += infil_n * dt_int

            infil_from_rain = min(infil_n, q_n)
            infil_from_hr = max(0.0, infil_n - infil_from_rain)
            if infil_from_surface:
                infil_from_hr = min(infil_from_hr, h_r1_c / dt_int)
            else:
                infil_from_hr = 0.0

            # 5) sol + seepage
            h_s_tmp = h_s_c + infil_n * dt_int
            if k_seepage > 0:
                h_s_new = h_s_tmp * math.exp(-k_seepage * dt_int)
                seep = h_s_tmp - h_s_new
            else:
                h_s_new = h_s_tmp
                seep = 0.0
            h_s_c = h_s_new
            seep_vol += seep

            # 6) génération surface
            r_gen_n = max(q_n - infil_from_rain, 0.0)
            rgen_vol += r_gen_n * dt_int

            # 7) cascade r1 -> r2 (solution exacte)

            # r1: entrée = r_gen_n - infil_from_hr
            b1 = r_gen_n - infil_from_hr
            h10 = h_r1_c
            if k_runoff1 > 0:
                e1 = math.exp(-k_runoff1 * dt_int)
                h11 = h10 * e1 + (b1 / k_runoff1) * (1.0 - e1)
            else:
                h11 = h10 + b1 * dt_int
            h11 = max(0.0, h11)

            Vout1 = b1 * dt_int + h10 - h11
            Vout1 = max(0.0, Vout1)
            h_r1_c = h11

            # r2: entrée = Vout1/dt
            b2 = Vout1 / dt_int
            h20 = h_r2_c
            if k_runoff2 > 0:
                e2 = math.exp(-k_runoff2 * dt_int)
                h21 = h20 * e2 + (b2 / k_runoff2) * (1.0 - e2)
            else:
                h21 = h20 + b2 * dt_int
            h21 = max(0.0, h21)

            Vout2 = b2 * dt_int + h20 - h21
            Vout2 = max(0.0, Vout2)
            h_r2_c = h21

            rout_vol += Vout2

        # stockage états
        h_a[n + 1]  = h_a_c
        h_s[n + 1]  = h_s_c
        h_r1[n + 1] = h_r1_c
        h_r2[n + 1] = h_r2_c

        # débits surfaciques moyens sur dt_obs
        q[n]         = q_vol / dt_obs
        infil[n]     = infil_vol / dt_obs
        r_gen[n]     = rgen_vol / dt_obs
        r_out[n]     = rout_vol / dt_obs

        # pertes en m/pas obs
        sa_loss[n]   = sa_vol
        seep_loss[n] = seep_vol

    return {
        "h_a": h_a, "h_s": h_s, "h_r1": h_r1, "h_r2": h_r2,
        "q": q, "infil": infil, "r_gen": r_gen, "r_out": r_out,
        "sa_loss": sa_loss, "seep_loss": seep_loss
    }


# =========================================================
# Objectif calage
# =========================================================

def compute_rmse(q_obs, q_mod):
    q_obs = np.asarray(q_obs, float)
    q_mod = np.asarray(q_mod, float)
    mask = np.isfinite(q_obs) & np.isfinite(q_mod)
    if mask.sum() < 2:
        return 1e9
    d = q_mod[mask] - q_obs[mask]
    return float(np.sqrt(np.mean(d * d)))

def _bounds_pm20pct(x):
    lo = 0.7 * float(x)
    hi = 1.3 * float(x)
    return max(lo, 1e-30), hi

def sample_uniform(bounds_log10):
    return np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds_log10], float)

def objective_rmse_theta_log10(theta_log10, data):
    """
    theta = [log10(k_infiltr), log10(k_runoff1), log10(k_runoff2), log10(k_seepage)]
    RMSE sur Q_ruiss (L/h).
    """
    log10_ki, log10_kr1, log10_kr2, log10_ks = theta_log10
    b = data["bounds_log10"]

    if not (b[0][0] <= log10_ki  <= b[0][1]): return 1e9
    if not (b[1][0] <= log10_kr1 <= b[1][1]): return 1e9
    if not (b[2][0] <= log10_kr2 <= b[2][1]): return 1e9
    if not (b[3][0] <= log10_ks  <= b[3][1]): return 1e9

    k_infiltr = 10**log10_ki
    k_runoff1 = 10**log10_kr1
    k_runoff2 = 10**log10_kr2
    k_seepage = 10**log10_ks

    try:
        res = run_scs_hsm_cascade(
            dt_obs=data["dt_obs"],
            dt_internal=data["dt_internal"],
            p_rate=data["p_rate"],
            etp_rate=data["etp_rate"],
            i_a=data["i_a"],
            s=data["s"],
            k_infiltr=k_infiltr,
            k_seepage=k_seepage,
            k_runoff1=k_runoff1,
            k_runoff2=k_runoff2,
            infil_from_surface=data["infil_from_surface"],
        )
    except Exception:
        return 1e9

    q_mod_m3s = np.asarray(res["r_out"], float) * data["A_BV_M2"]
    q_mod_lh  = m3s_to_lh(q_mod_m3s)

    q_obs_lh  = data["qruiss_obs_lh"]
    return compute_rmse(q_obs_lh, q_mod_lh)

def calibrate_multistart_powell(data, bounds_log10, n_starts=20):
    best_x, best_J = None, np.inf
    for i in range(n_starts):
        x0 = sample_uniform(bounds_log10)
        res = minimize(
            objective_rmse_theta_log10, x0, args=(data,),
            method="Powell", bounds=bounds_log10,
            options={"maxiter": 300, "disp": False}
        )
        J = float(res.fun) if np.isfinite(res.fun) else 1e9
        print(f"Essai {i+1}/{n_starts} : RMSE = {J:.6e}")
        if J < best_J:
            best_J = J
            best_x = np.array(res.x, float)
    return best_x, best_J


# =========================================================
# Bilan de masse
# =========================================================

def compute_mass_balance_cascade(
    res: dict,
    p_rate: np.ndarray,
    etp_rate: np.ndarray,
    dt_obs: float,
    h_a0: float = 0.0,
    h_s0: float = 0.0,
    h_r10: float = 0.0,
    h_r20: float = 0.0,
) -> dict:
    """
    Convention :
      - p_rate, etp_rate : m/s
      - res["r_out"]     : m/s (sortie vers exutoire, sortie du r2)
      - res["seep_loss"] : m par pas obs
      - res["sa_loss"]   : m par pas obs
      - états en m
    """
    p_rate   = np.nan_to_num(np.asarray(p_rate, float), nan=0.0)
    etp_rate = np.nan_to_num(np.asarray(etp_rate, float), nan=0.0)
    dt = float(dt_obs)

    P_tot     = float(np.nansum(p_rate) * dt)
    R_tot     = float(np.nansum(np.asarray(res["r_out"], float)) * dt)
    Seep_tot  = float(np.nansum(np.asarray(res["seep_loss"], float)))
    ET_Ia_tot = float(np.nansum(np.asarray(res["sa_loss"], float)))

    ET_Soil_tot = 0.0
    ET_tot = ET_Ia_tot + ET_Soil_tot

    h_a  = np.asarray(res["h_a"], float)
    h_s  = np.asarray(res["h_s"], float)
    h_r1 = np.asarray(res["h_r1"], float)
    h_r2 = np.asarray(res["h_r2"], float)

    delta_storage = (h_a[-1] - h_a0) + (h_s[-1] - h_s0) + (h_r1[-1] - h_r10) + (h_r2[-1] - h_r20)
    closure = P_tot - (Seep_tot + ET_tot + R_tot + delta_storage)

    return {
        "P_tot_m": P_tot,
        "R_tot_m": R_tot,
        "Seep_tot_m": Seep_tot,
        "ET_Ia_tot_m": ET_Ia_tot,
        "ET_Soil_tot_m": ET_Soil_tot,
        "ET_tot_m": ET_tot,
        "Delta_storage_m": delta_storage,
        "Closure_error_m": closure,
        "Closure_error_mm": closure * 1000.0,
        "Relative_error_%": 100.0 * closure / P_tot if P_tot > 0 else np.nan,
    }

def print_mass_balance(mb: dict):
    print("\n=== Bilan de masse sur la période ===")
    print(f"P_tot          = {mb['P_tot_m']*1000:.3f} mm")
    print(f"Ruissellement  = {mb['R_tot_m']*1000:.3f} mm (vers exutoire)")
    print(f"Seepage/drain  = {mb['Seep_tot_m']*1000:.3f} mm")
    print(f"ETP Ia         = {mb['ET_Ia_tot_m']*1000:.3f} mm")
    print(f"ETP Sol        = {mb['ET_Soil_tot_m']*1000:.3f} mm")
    print(f"ETP totale     = {mb['ET_tot_m']*1000:.3f} mm")
    print(f"ΔStock (Ia+sol+r1+r2) = {mb['Delta_storage_m']*1000:.3f} mm")
    print(f"Erreur fermeture     = {mb['Closure_error_mm']:.6f} mm ({mb['Relative_error_%']:.6f} %)")


# =========================================================
# MAIN
# =========================================================
def main():
    base_dir = Path(__file__).resolve().parent

    # ---- event à tester
    csv_event_rel = "all_events1/2024/event_2024_004.csv"
    event_name = Path(csv_event_rel).stem

    # ---- surface parking
    A_BV_M2 = 94.0

    # ---- paramètres fixés
    I_A_FIXED = 0.0002  # m
    S_FIXED   = 0.13    # m

    # ---- init (centres)
    k_runoff1_init  = 1.7e-3
    k_runoff2_init  = 1.7e-3
    k_seepage_init  = 6.5e-05

    print("=== PARAMÈTRES INIT (centres) ===")
    print(f"A_BV_M2      = {A_BV_M2:.1f} m²")
    print(f"k_runoff1    = {k_runoff1_init:.3e} s^-1 (t1/2 ≈ {math.log(2)/k_runoff1_init/3600:.3f} h)")
    print(f"k_runoff2    = {k_runoff2_init:.3e} s^-1 (t1/2 ≈ {math.log(2)/k_runoff2_init/3600:.3f} h)")
    print(f"k_seepage    = {k_seepage_init:.3e} s^-1 (t1/2 ≈ {math.log(2)/k_seepage_init/3600:.3f} h)")
    print("=================================\n")

    # ---- lecture event
    time_index, P_mm_event, qinf_obs_lh, qruiss_obs_lh, dt_obs = read_parking_event_csv(csv_event_rel)
    p_rate = mm_per_step_to_mps(P_mm_event, dt_obs)

    # ---- ETP (Ia uniquement)
    ETP_MM_PER_DAY = 5.0
    etp_rate = build_constant_daytime_etp_rate(time_index, etp_mm_per_day=ETP_MM_PER_DAY, start_hour=8, end_hour=20)
    print(f"[INFO] ETP synthétique = {ETP_MM_PER_DAY:.2f} mm/j entre 8h et 20h (0 sinon)")

    # ---- substepping
    DT_INTERNAL = 15.0
    print(f"[INFO] dt obs = {dt_obs:.1f} s | dt interne = {DT_INTERNAL:.1f} s")

    # ---- option : infiltration peut pomper h_r1 ?
    INFIL_FROM_SURFACE = False

    # ---- bornes k_infiltr en mm/h (calage)
    KINF_MIN_MM_H = 0.03
    KINF_MAX_MM_H = 6
    ki_lo = infil_mm_h_to_m_s(KINF_MIN_MM_H)
    ki_hi = infil_mm_h_to_m_s(KINF_MAX_MM_H)

    # ---- calage
    DO_CALIBRATION = True
    N_STARTS = 15

    if DO_CALIBRATION:
        kr1_lo, kr1_hi = _bounds_pm20pct(k_runoff1_init)
        kr2_lo, kr2_hi = _bounds_pm20pct(k_runoff2_init)
        ks_lo,  ks_hi  = _bounds_pm20pct(k_seepage_init)

        bounds_log10 = [
            (math.log10(ki_lo),  math.log10(ki_hi)),
            (math.log10(kr1_lo), math.log10(kr1_hi)),
            (math.log10(kr2_lo), math.log10(kr2_hi)),
            (math.log10(ks_lo),  math.log10(ks_hi)),
        ]

        print("=== Bornes calibration ===")
        print(f"k_infiltr  ∈ [{ki_lo:.3e}, {ki_hi:.3e}] m/s   ({KINF_MIN_MM_H:.2f}–{KINF_MAX_MM_H:.2f} mm/h)")
        print(f"k_runoff1  ∈ [{kr1_lo:.3e}, {kr1_hi:.3e}] s^-1")
        print(f"k_runoff2  ∈ [{kr2_lo:.3e}, {kr2_hi:.3e}] s^-1")
        print(f"k_seepage  ∈ [{ks_lo:.3e}, {ks_hi:.3e}] s^-1")
        print("==========================\n")

        data = dict(
            dt_obs=dt_obs,
            dt_internal=DT_INTERNAL,
            p_rate=p_rate,
            etp_rate=etp_rate,
            i_a=I_A_FIXED,
            s=S_FIXED,
            A_BV_M2=A_BV_M2,
            qruiss_obs_lh=np.asarray(qruiss_obs_lh, float),
            bounds_log10=bounds_log10,
            infil_from_surface=INFIL_FROM_SURFACE,
        )

        print(f"Lancement du calage (multistart={N_STARTS} + Powell) sur RMSE(Q_ruiss en L/h)...")
        theta_opt, J_opt = calibrate_multistart_powell(data, bounds_log10, n_starts=N_STARTS)

        log10_ki, log10_kr1, log10_kr2, log10_ks = theta_opt
        k_infiltr = 10**log10_ki
        k_runoff1 = 10**log10_kr1
        k_runoff2 = 10**log10_kr2
        k_seepage = 10**log10_ks

        print("\n=== Résultats calage (RMSE) ===")
        print(f"RMSE_opt     = {J_opt:.6e} (L/h)")
        print(f"k_infiltr    = {k_infiltr:.6e} m/s   ({k_infiltr*3600*1000:.3f} mm/h)")
        print(f"k_runoff1    = {k_runoff1:.6e} s^-1  (t1/2 ≈ {math.log(2)/k_runoff1/60:.3f} min)")
        print(f"k_runoff2    = {k_runoff2:.6e} s^-1  (t1/2 ≈ {math.log(2)/k_runoff2/60:.3f} min)")
        print(f"k_seepage    = {k_seepage:.6e} s^-1  (t1/2 ≈ {math.log(2)/k_seepage/3600:.3f} h)")
        print("==============================\n")

    else:
        # fallback (comme ton script)
        k_infiltr = 2.747694e-07
        k_runoff1 = k_runoff1_init
        k_runoff2 = k_runoff2_init
        k_seepage = k_seepage_init

    # ---- simulation finale
    res = run_scs_hsm_cascade(
        dt_obs=dt_obs,
        dt_internal=DT_INTERNAL,
        p_rate=p_rate,
        etp_rate=etp_rate,
        i_a=I_A_FIXED,
        s=S_FIXED,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
        k_runoff1=k_runoff1,
        k_runoff2=k_runoff2,
        infil_from_surface=INFIL_FROM_SURFACE,
    )

    # ---- bilan de masse
    mb = compute_mass_balance_cascade(
        res=res,
        p_rate=p_rate,
        etp_rate=etp_rate,
        dt_obs=dt_obs,
        h_a0=0.0, h_s0=0.0, h_r10=0.0, h_r20=0.0
    )
    print_mass_balance(mb)

    # =========================================================
    # Conversions débits + volumes cumulés (m³)
    # =========================================================
    qruiss_obs_lh = np.asarray(qruiss_obs_lh, float)
    qruiss_obs_m3s = lh_to_m3s(qruiss_obs_lh)

    qruiss_mod_m3s = np.asarray(res["r_out"], float) * A_BV_M2
    qruiss_mod_lh  = m3s_to_lh(qruiss_mod_m3s)

    # Proxy débit "inf/drain" mod : seep_loss (m/pas) -> m/s -> m3/s -> L/h
    qinf_mod_m3s = (np.asarray(res["seep_loss"], float) / dt_obs) * A_BV_M2
    qinf_mod_lh  = m3s_to_lh(qinf_mod_m3s)

    qinf_obs_lh = None if qinf_obs_lh is None else np.asarray(qinf_obs_lh, float)

    V_obs_cum = np.cumsum(np.clip(qruiss_obs_m3s, 0.0, None) * dt_obs)
    V_mod_cum = np.cumsum(np.clip(qruiss_mod_m3s, 0.0, None) * dt_obs)

    print("\n===== BILAN VOLUMES (RUISSELLEMENT) =====")
    print(f"Volume observé V_obs     = {V_obs_cum[-1]:.6f} m³")
    print(f"Volume modélisé V_mod    = {V_mod_cum[-1]:.6f} m³")
    if V_obs_cum[-1] > 0:
        print(f"Rapport V_mod / V_obs    = {V_mod_cum[-1] / V_obs_cum[-1]:.3f}")
    print("=========================================\n")

    # =========================================================
    # Séries mm/pas + cumuls
    # =========================================================
    factor_mm = dt_obs * 1000.0
    P_mm      = p_rate * factor_mm
    infil_mm  = np.asarray(res["infil"], float) * factor_mm
    runoff_mm = np.asarray(res["r_out"], float) * factor_mm
    seep_mm   = np.asarray(res["seep_loss"], float) * 1000.0
    ET_Ia_mm  = np.asarray(res["sa_loss"], float) * 1000.0

    P_cum      = np.cumsum(P_mm)
    infil_cum  = np.cumsum(infil_mm)
    runoff_cum = np.cumsum(runoff_mm)
    seep_cum   = np.cumsum(seep_mm)
    ET_Ia_cum  = np.cumsum(ET_Ia_mm)

    # =========================================================
    # Dossier plots
    # =========================================================
    plots_dir = base_dir.parent / "03_Plots" / "Parking_CSR_CASCADE" / event_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    dt_days = dt_obs / 86400.0
    maxP = float(np.nanmax(P_mm)) if np.nanmax(P_mm) > 0 else 1.0

    # FIG 0: volumes cumulés
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_index, V_obs_cum, label="V_obs cumulé (ruissellement)", linewidth=1.4)
    ax.plot(time_index, V_mod_cum, label="V_mod cumulé (ruissellement)", linewidth=1.4, linestyle="--")
    ax.set_ylabel("Volume cumulé (m³)")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "cumuls_V_obs_V_mod.png", dpi=200)
    plt.close(fig)

    # FIG 1: Q_ruiss + pluie
    fig, axQ = plt.subplots(figsize=(10, 4))
    axQ.plot(time_index, qruiss_obs_lh, label="Q_ruiss_obs (L/h)", linewidth=1.0, alpha=0.7)
    axQ.plot(time_index, qruiss_mod_lh, label="Q_ruiss_mod (L/h)", linewidth=1.2)
    axQ.set_xlabel("Date")
    axQ.set_ylabel("Débit ruisselé (L/h)")
    axQ.grid(True, linewidth=0.4, alpha=0.6)

    axP = axQ.twinx()
    axP.bar(time_index, P_mm, width=dt_days * 0.8, align="center", alpha=0.35, label="P (mm/pas)")
    axP.set_ylabel("Pluie (mm/pas)")
    axP.invert_yaxis()
    axP.set_ylim(maxP * 1.05, 0.0)

    l1, lab1 = axQ.get_legend_handles_labels()
    l2, lab2 = axP.get_legend_handles_labels()
    axQ.legend(l1 + l2, lab1 + lab2, loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "Qruiss_obs_vs_mod_LH_P.png", dpi=200)
    plt.close(fig)

    # FIG 2: Q_inf (si dispo) + proxy mod (seep)
    fig, axD = plt.subplots(figsize=(10, 4))
    if qinf_obs_lh is not None:
        axD.plot(time_index, qinf_obs_lh, label="Q_inf_obs (L/h)", linewidth=1.0, alpha=0.7)
    axD.plot(time_index, qinf_mod_lh, label="Q_inf_mod proxy (seep) (L/h)", linewidth=1.2)
    axD.set_xlabel("Date")
    axD.set_ylabel("Débit drain (L/h)")
    axD.grid(True, linewidth=0.4, alpha=0.6)

    axP2 = axD.twinx()
    axP2.bar(time_index, P_mm, width=dt_days * 0.8, align="center", alpha=0.35, label="P (mm/pas)")
    axP2.set_ylabel("Pluie (mm/pas)")
    axP2.invert_yaxis()
    axP2.set_ylim(maxP * 1.05, 0.0)

    l1, lab1 = axD.get_legend_handles_labels()
    l2, lab2 = axP2.get_legend_handles_labels()
    axD.legend(l1 + l2, lab1 + lab2, loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "Qinf_obs_vs_mod_LH_P.png", dpi=200)
    plt.close(fig)

    # FIG 3: états
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_index, np.asarray(res["h_a"],  float)[:-1], label="h_a (Ia)")
    ax.plot(time_index, np.asarray(res["h_s"],  float)[:-1], label="h_s (sol)")
    ax.plot(time_index, np.asarray(res["h_r1"], float)[:-1], label="h_r1")
    ax.plot(time_index, np.asarray(res["h_r2"], float)[:-1], label="h_r2")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hauteur (m)")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "etats_reservoirs.png", dpi=200)
    plt.close(fig)

    # FIG 4: cumuls mm
    fig, axc = plt.subplots(figsize=(10, 4))
    axc.plot(time_index, P_cum,      label="P cumulée", linewidth=1.3)
    axc.plot(time_index, infil_cum,  label="Infiltration cumulée", linewidth=1.1)
    axc.plot(time_index, seep_cum,   label="Drain/Seepage cumulé", linestyle="--", linewidth=1.1)
    axc.plot(time_index, runoff_cum, label="Ruissellement cumulé", linewidth=1.3)
    axc.plot(time_index, ET_Ia_cum,  label="ET Ia cumulée", linewidth=1.0, alpha=0.8)
    axc.set_xlabel("Date")
    axc.set_ylabel("Lame cumulée (mm)")
    axc.grid(True, linewidth=0.4, alpha=0.6)
    axc.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "cumuls_mm.png", dpi=200)
    plt.close(fig)

    print(f"[OK] Figures sauvegardées dans : {plots_dir}")


if __name__ == "__main__":
    main()
