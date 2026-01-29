# -*- coding: utf-8 -*-
"""
SCS-HSM continu — VERSION AVEC ROUTAGE (réservoir de surface vers exutoire)
APPLIQUÉ AU PARKING CSR (calage évènementiel)
---------------------------------------------------------------

Objectif :
    - Lire un CSV d'évènement CSR 
    - Caler (multistart + Powell) :
          k_infiltr  (m/s)  : capacité d'infiltration
          k_seepage  (s^-1) : vidange/percolation du sol
          k_runoff   (s^-1) : vidange du stock de surface vers l'exutoire
    - Simuler Q_mod et produire bilans + figures

Modèle A + routage simple du stock de surface :
    - h_a : réservoir d'abstraction Ia
    - h_s : réservoir de sol (capacité S)
    - h_r : stock de surface
    - r_gen = max(q - infil_from_rain, 0) : ruissellement généré par la pluie nette
    - r_out = k_runoff * h_r (borné)      : ruissellement sortant vers exutoire
    - Q_mod = r_out * A_BV_M2             : débit modélisé à l'exutoire (m3/s)

Entrées :
    - Date                : datetime
    - Hauteur_de_pluie_mm : pluie (mm / pas)
    - Q_inf_LH            : débit infiltré (L/h) [optionnel, non calé ici]
    - Q_ruiss_LH          : débit ruisselé (L/h) -> utilisé comme Q_obs


Sorties :
    - Figures dans ../03_Plots/Parking_CSR_Avec_routage/<event_name>/
"""

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl

# Optimisation affichage
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["agg.path.chunksize"] = 10000


# =======================
# PARAMÈTRES UTILISATEUR
# =======================
CSV_EVENT_REL = "all_events1/2024/event_2024_004.csv"  # chemin relatif depuis 02_Data
A_BV_M2 = 94                                           # surface du parking (m²)
N_STARTS = 25                                          # multistart
I_A_FIXED = 0.002                                      # m
S_FIXED = 0.15                                         # m
DO_CALIBRATION = True                                  # True = calage, False = run "à la main"


# =======================
# BORNES PHYSIQUES
# =======================
def half_life_bounds_to_log10k(t_half_min_h: float, t_half_max_h: float) -> tuple[float, float]:
    """Convertit demi-vie (h) -> bornes log10(k) (s^-1)."""
    t1 = min(t_half_min_h, t_half_max_h)
    t2 = max(t_half_min_h, t_half_max_h)
    k_max = math.log(2.0) / (t1 * 3600.0)
    k_min = math.log(2.0) / (t2 * 3600.0)
    return math.log10(k_min), math.log10(k_max)

def infil_bounds_mm_h_to_log10k(v_min_mm_h: float, v_max_mm_h: float) -> tuple[float, float]:
    """Convertit capacité d'infiltration (mm/h) -> bornes log10(k_infiltr) (m/s)."""
    v1 = min(v_min_mm_h, v_max_mm_h)
    v2 = max(v_min_mm_h, v_max_mm_h)
    k_min = (v1 / 1000.0) / 3600.0
    k_max = (v2 / 1000.0) / 3600.0
    return math.log10(k_min), math.log10(k_max)

# 1) Infiltration : mm/h
INFIL_MIN_MM_H = 0.1
INFIL_MAX_MM_H = 7.0
LOG10_KINF_MIN, LOG10_KINF_MAX = infil_bounds_mm_h_to_log10k(INFIL_MIN_MM_H, INFIL_MAX_MM_H)

# 2) Routage surface : demi-vie (h)
T_HALF_RUNOFF_MIN_H = 0.5
T_HALF_RUNOFF_MAX_H = 6.0
LOG10_KRUN_MIN, LOG10_KRUN_MAX = half_life_bounds_to_log10k(T_HALF_RUNOFF_MIN_H, T_HALF_RUNOFF_MAX_H)

# 3) Seepage sol : demi-vie (h)
T_HALF_SEEP_MIN_H = 0.5
T_HALF_SEEP_MAX_H = 12.0
LOG10_KSEEP_MIN, LOG10_KSEEP_MAX = half_life_bounds_to_log10k(T_HALF_SEEP_MIN_H, T_HALF_SEEP_MAX_H)


# ======================================================================
# 1) Modèle SCS-HSM + routage du stock de surface vers exutoire
# ======================================================================

def run_scs_hsm(
    dt: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray | None = None,
    i_a: float = 2e-3,
    s: float = 0.02,
    k_infiltr: float = 1e-6,
    k_seepage: float = 1e-5,
    k_runoff: float = 1e-4,
    h_a_init: float = 0.0,
    h_s_init: float = 0.0,
    h_r_init: float = 0.0,
) -> dict:
  
    p_rate = np.nan_to_num(np.asarray(p_rate, dtype=float), nan=0.0)
    nt = len(p_rate)

    if etp_rate is None:
        etp_rate = np.zeros(nt, dtype=float)
    else:
        etp_rate = np.nan_to_num(np.asarray(etp_rate, dtype=float), nan=0.0)
        if len(etp_rate) != nt:
            raise ValueError("etp_rate doit avoir la même longueur que p_rate")

    # Etats
    h_a = np.zeros(nt + 1, dtype=float)
    h_s = np.zeros(nt + 1, dtype=float)
    h_r = np.zeros(nt + 1, dtype=float)
    h_a[0] = float(h_a_init)
    h_s[0] = float(h_s_init)
    h_r[0] = float(h_r_init)

    # Flux
    q         = np.zeros(nt, dtype=float)  # pluie nette [m/s]
    infil     = np.zeros(nt, dtype=float)  # infiltration [m/s]
    r_gen     = np.zeros(nt, dtype=float)  # ruissellement généré [m/s]
    r_out     = np.zeros(nt, dtype=float)  # ruissellement vers exutoire [m/s]
    sa_loss   = np.zeros(nt, dtype=float)  # ETP sur Ia [m/pas]
    seep_loss = np.zeros(nt, dtype=float)  # seepage [m/pas]

    for n in range(nt):
        p   = p_rate[n]
        etp = etp_rate[n]

        # 1) ETP sur h_a
        etp_eff = min(etp * dt, h_a[n])
        sa_loss[n] = etp_eff
        h_a_after = h_a[n] - etp_eff

        # 2) Ia -> pluie nette q_n
        h_temp = h_a_after + p * dt
        if h_temp < i_a:
            q_n = 0.0
            h_a[n + 1] = h_temp
        else:
            q_n = (h_temp - i_a) / dt
            h_a[n + 1] = i_a
        q[n] = q_n

        # 3) Infiltration potentielle HSM
        h_s_b = h_s[n]
        Xb = 1.0 - h_s_b / s
        if Xb <= 0.0:
            Xb = 1e-12
        Xe = 1.0 / (1.0 / Xb + k_infiltr * dt / s)
        h_s_end = (1.0 - Xe) * s
        infil_pot = (h_s_end - h_s_b) / dt  # [m/s]

        # 4) Limitation par eau dispo (q + h_r/dt)
        h_r_b = h_r[n]
        water_avail_rate = q_n + h_r_b / dt
        water_avail_rate = max(water_avail_rate, 0.0)

        infil_n = max(0.0, min(infil_pot, water_avail_rate))
        infil[n] = infil_n

        # Décomposition : ce qui vient de la pluie vs du stock de surface
        infil_from_rain = min(infil_n, q_n)
        infil_from_hr_rate = max(0.0, infil_n - infil_from_rain)
        infil_from_hr_rate = min(infil_from_hr_rate, h_r_b / dt)

        # 5) Mise à jour sol + seepage
        hs_temp = h_s_b + infil_n * dt
        if k_seepage > 0.0:
            h_s_after = hs_temp * math.exp(-k_seepage * dt)
            seep = hs_temp - h_s_after
        else:
            h_s_after = hs_temp
            seep = 0.0
        h_s[n + 1] = h_s_after
        seep_loss[n] = seep

        # 6) Ruissellement généré (depuis la pluie nette)
        r_gen_n = max(q_n - infil_from_rain, 0.0)
        r_gen[n] = r_gen_n

        # 7) Routage vers exutoire : r_out = k_runoff * h_r_b, borné
        if k_runoff > 0.0:
            r_out_raw = k_runoff * h_r_b  # [m/s]

            # flux max dispo à la surface sur le pas
            water_surface_rate = h_r_b / dt + r_gen_n - infil_from_hr_rate
            water_surface_rate = max(water_surface_rate, 0.0)

            r_out_n = min(r_out_raw, water_surface_rate)
            r_out_n = max(r_out_n, 0.0)
        else:
            r_out_n = 0.0

        r_out[n] = r_out_n

        # Mise à jour h_r
        h_r_next = h_r_b + (r_gen_n - r_out_n - infil_from_hr_rate) * dt
        h_r[n + 1] = max(h_r_next, 0.0)

    # Bilan de masse (m)
    P_tot    = float(np.nansum(p_rate) * dt)
    Seep_tot = float(np.nansum(seep_loss))
    ET_tot   = float(np.nansum(sa_loss))
    R_tot    = float(np.nansum(r_out) * dt)  # flux sortant vers exutoire

    d_ha = h_a[-1] - h_a[0]
    d_hs = h_s[-1] - h_s[0]
    d_hr = h_r[-1] - h_r[0]
    delta_storage = d_ha + d_hs + d_hr

    closure = P_tot - (Seep_tot + ET_tot + R_tot + delta_storage)

    mass_balance = {
        "P_tot_m": P_tot,
        "R_tot_m": R_tot,
        "Seep_tot_m": Seep_tot,
        "ET_tot_m": ET_tot,
        "Delta_storage_m": delta_storage,
        "Closure_error_m": closure,
        "Closure_error_mm": closure * 1000.0,
        "Relative_error_pct": 100.0 * closure / P_tot if P_tot > 0 else np.nan,
    }

    return {
        "h_a": h_a,
        "h_s": h_s,
        "h_r": h_r,
        "q": q,
        "infil": infil,
        "r_gen": r_gen,
        "r_out": r_out,
        "sa_loss": sa_loss,
        "seep_loss": seep_loss,
        "mass_balance": mass_balance,
    }


# ======================================================================
# 2) Lecture CSV CSR (canonique + compat)
# ======================================================================

def _infer_dayfirst(date_series: pd.Series) -> bool:
    """Heuristique : dd/mm -> dayfirst True, ISO yyyy-mm-dd -> False."""
    s = date_series.astype(str).dropna().head(20).tolist()
    if not s:
        return False
    slash_like = sum(bool(re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", x)) for x in s)
    iso_like   = sum(bool(re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", x)) for x in s)
    return slash_like > iso_like

def read_parking_event_csv(csv_rel_path: str):
    """
    Retour :
      time_index, P_mm, p_rate, q_obs_m3s, dt_seconds, q_inf_m3s
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "02_Data"
    csv_path = data_dir / csv_rel_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV évènement introuvable : {csv_path}")

    df = pd.read_csv(csv_path, sep=";", na_values=["NA", "NaN", "", -9999, -9999.0])

    # Date
    date_col = next((c for c in ["Date", "date", "dateP"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            "Colonne date introuvable. Attendu: Date (ou date/dateP). "
            f"Colonnes présentes: {list(df.columns)}"
        )

    dayfirst = _infer_dayfirst(df[date_col])
    time_series = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="raise")
    time_index = pd.DatetimeIndex(time_series)

    if len(time_index) < 2:
        raise ValueError("Série trop courte pour estimer dt.")
    dt_seconds = float(time_index.to_series().diff().dropna().dt.total_seconds().median())

    # Pluie
    if "Hauteur_de_pluie_mm" in df.columns:
        p_col = "Hauteur_de_pluie_mm"
    elif "P_mm" in df.columns:
        p_col = "P_mm"
    else:
        raise ValueError(
            "Colonne pluie absente. Attendu: Hauteur_de_pluie_mm (ou P_mm). "
            f"Colonnes présentes: {list(df.columns)}"
        )

    P_mm = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0).to_numpy(float)
    P_mm = np.clip(P_mm, 0.0, None)
    p_rate = P_mm * 1e-3 / dt_seconds  # mm/pas -> m/s

    # Débit observé : priorité CSR canonique
    q_inf_m3s = None

    if "Q_ruiss_LH" in df.columns:
        q_ruiss_lh = pd.to_numeric(df["Q_ruiss_LH"], errors="coerce").fillna(0.0).to_numpy(float)
        q_ruiss_lh = np.clip(q_ruiss_lh, 0.0, None)
        q_obs_m3s = q_ruiss_lh / 1000.0 / 3600.0

        if "Q_inf_LH" in df.columns:
            q_inf_lh = pd.to_numeric(df["Q_inf_LH"], errors="coerce").fillna(0.0).to_numpy(float)
            q_inf_lh = np.clip(q_inf_lh, 0.0, None)
            q_inf_m3s = q_inf_lh / 1000.0 / 3600.0

    elif "Q_ls" in df.columns:
        q_ls = pd.to_numeric(df["Q_ls"], errors="coerce").fillna(0.0).to_numpy(float)
        q_ls = np.clip(q_ls, 0.0, None)
        q_obs_m3s = q_ls / 1000.0

    else:
        raise ValueError(
            "Colonne débit observé absente. Attendu: Q_ruiss_LH (ou Q_ls). "
            f"Colonnes présentes: {list(df.columns)}"
        )

    return time_index, P_mm, p_rate, q_obs_m3s, dt_seconds, q_inf_m3s


# ======================================================================
# 3) Critères
# ======================================================================

def compute_log_rmse(q_obs: np.ndarray, q_mod: np.ndarray, eps: float = 1e-9) -> float:
    q_obs = np.asarray(q_obs, dtype=float)
    q_mod = np.asarray(q_mod, dtype=float)
    mask = np.isfinite(q_obs) & np.isfinite(q_mod) & (q_obs > 0.0)
    if mask.sum() == 0:
        return 1e6
    o = q_obs[mask]
    m = np.maximum(q_mod[mask], eps)
    diff = np.log(m) - np.log(o + eps)
    return float(np.sqrt(np.mean(diff**2)))


# ======================================================================
# 4) Objectif + multistart
# ======================================================================

def objective(theta: np.ndarray, data: dict) -> float:
    """
    theta = [log10(k_infiltr), log10(k_seepage), log10(k_runoff)]
    J = log-RMSE(Q_obs, Q_mod) sur instants où Q_obs > 0.
    """
    log10_ki, log10_ks, log10_kr = theta

    # Gardes-fous (bornes physiques)
    if not (LOG10_KINF_MIN <= log10_ki <= LOG10_KINF_MAX):
        return 1e6
    if not (LOG10_KSEEP_MIN <= log10_ks <= LOG10_KSEEP_MAX):
        return 1e6
    if not (LOG10_KRUN_MIN <= log10_kr <= LOG10_KRUN_MAX):
        return 1e6

    k_infiltr = 10.0 ** log10_ki
    k_seepage = 10.0 ** log10_ks
    k_runoff  = 10.0 ** log10_kr

    dt       = data["dt"]
    p_rate   = data["p_rate"]
    etp_rate = data["etp_rate"]
    q_obs    = data["q_obs_m3s"]
    A_BV_M2  = data["A_BV_M2"]
    i_a      = data["i_a_fixed"]
    s        = data["S_FIXED"]

    try:
        res = run_scs_hsm(
            dt=dt,
            p_rate=p_rate,
            etp_rate=etp_rate,
            i_a=i_a,
            s=s,
            k_infiltr=k_infiltr,
            k_seepage=k_seepage,
            k_runoff=k_runoff,
            h_a_init=0.0,
            h_s_init=0.0,
            h_r_init=0.0,
        )
    except Exception:
        return 1e6

    q_mod = res["r_out"] * A_BV_M2
    J = compute_log_rmse(q_obs, q_mod, eps=1e-9)
    if (J is None) or (not np.isfinite(J)):
        return 1e6
    return float(J)

def sample_random_theta(bounds: list[tuple[float, float]]) -> np.ndarray:
    return np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)

def calibrate_multistart(data: dict, bounds: list[tuple[float, float]], n_starts: int = 15) -> tuple[np.ndarray, float]:
    best_theta = None
    best_J = np.inf

    for k in range(n_starts):
        theta0 = sample_random_theta(bounds)
        res = minimize(
            objective,
            theta0,
            args=(data,),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 200, "disp": False},
        )
        print(f"Essai {k+1}/{n_starts} : J = {res.fun:.4f}")
        if res.fun < best_J:
            best_J = float(res.fun)
            best_theta = np.array(res.x, dtype=float)

    return best_theta, best_J


# ======================================================================
# 5) Affichage bilans
# ======================================================================

def print_mass_balance(mb: dict):
    print("\n=== Bilan de masse sur la période ===")
    print(f"P_tot               = {mb['P_tot_m']*1000:.1f} mm")
    print(f"Ruissellement (sort)= {mb['R_tot_m']*1000:.1f} mm (vers exutoire)")
    print(f"Seepage profond     = {mb['Seep_tot_m']*1000:.1f} mm")
    print(f"ETP effective       = {mb['ET_tot_m']*1000:.1f} mm")
    print(f"ΔStock (Ia+sol+surf)= {mb['Delta_storage_m']*1000:.2f} mm")
    print(f"Erreur de fermeture = {mb['Closure_error_mm']:.3f} mm ({mb['Relative_error_pct']:.3f} %)")


# ======================================================================
# 6) MAIN
# ======================================================================

def main():
    base_dir = Path(__file__).resolve().parent
    csv_event_rel = CSV_EVENT_REL
    event_name = Path(csv_event_rel).stem

    print("=== Bornes physiques retenues (CSR avec routage) ===")
    print(f"k_infiltr  : {INFIL_MIN_MM_H:.0f}–{INFIL_MAX_MM_H:.0f} mm/h"
          f"  => log10(k) ∈ [{LOG10_KINF_MIN:.2f}, {LOG10_KINF_MAX:.2f}]")
    print(f"k_runoff   : t1/2 ∈ [{T_HALF_RUNOFF_MIN_H:.1f}, {T_HALF_RUNOFF_MAX_H:.1f}] h"
          f" => log10(k) ∈ [{LOG10_KRUN_MIN:.2f}, {LOG10_KRUN_MAX:.2f}]")
    print(f"k_seepage  : t1/2 ∈ [{T_HALF_SEEP_MIN_H:.1f}, {T_HALF_SEEP_MAX_H:.1f}] h"
          f" => log10(k) ∈ [{LOG10_KSEEP_MIN:.2f}, {LOG10_KSEEP_MAX:.2f}]")
    print("====================================================\n")

    # 1) Lecture événement CSR
    (time_index, P_mm_event, p_rate_input, q_obs_m3s, dt, q_inf_m3s) = read_parking_event_csv(csv_event_rel)
    print(f"[INFO] dt estimé pour l'évènement = {dt:.1f} s")

    # Parking : ETP = 0
    etp_rate = np.zeros_like(p_rate_input)

    q_obs_for_vol = np.where(np.isfinite(q_obs_m3s), np.clip(q_obs_m3s, 0.0, None), 0.0)

    # 2) Calage
    if DO_CALIBRATION:
        bounds = [
            (LOG10_KINF_MIN,  LOG10_KINF_MAX),
            (LOG10_KSEEP_MIN, LOG10_KSEEP_MAX),
            (LOG10_KRUN_MIN,  LOG10_KRUN_MAX),
        ]

        data = {
            "dt": dt,
            "p_rate": p_rate_input,
            "etp_rate": etp_rate,
            "q_obs_m3s": q_obs_m3s,
            "A_BV_M2": A_BV_M2,
            "i_a_fixed": I_A_FIXED,
            "S_FIXED": S_FIXED,
        }

        print("Lancement du calage (multistart + Powell) sur J = log-RMSE(Q)...")
        theta_opt, J_opt = calibrate_multistart(data, bounds, N_STARTS)

        log10_ki, log10_ks, log10_kr = theta_opt
        k_infiltr = 10.0 ** log10_ki
        k_seepage = 10.0 ** log10_ks
        k_runoff  = 10.0 ** log10_kr

        infil_mm_h   = k_infiltr * 3600.0 * 1000.0
        t12_seep_h   = math.log(2.0) / k_seepage / 3600.0
        t12_runoff_h = math.log(2.0) / k_runoff  / 3600.0

        print("\n=== Résultats du calage (avec routage) ===")
        print(f"J_opt        = {J_opt:.4f} (-)")
        print(f"  i_a        = {I_A_FIXED:.6f} m")
        print(f"  s          = {S_FIXED:.6f} m")
        print(f"  k_infiltr  = {k_infiltr:.3e} m/s ({infil_mm_h:.1f} mm/h)")
        print(f"  k_seepage  = {k_seepage:.3e} s^-1 (t1/2 ≈ {t12_seep_h:.2f} h)")
        print(f"  k_runoff   = {k_runoff:.3e} s^-1 (t1/2 ≈ {t12_runoff_h:.2f} h)")

        i_a = I_A_FIXED
        s   = S_FIXED

    else:
        # Valeurs "à la main"
        i_a = I_A_FIXED
        s   = S_FIXED
        k_infiltr = 2.0e-06
        k_seepage = 2.0e-05
        k_runoff  = 2.0e-04

    # 3) Simulation finale
    res = run_scs_hsm(
        dt=dt,
        p_rate=p_rate_input,
        etp_rate=etp_rate,
        i_a=i_a,
        s=s,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
        k_runoff=k_runoff,
        h_a_init=0.0,
        h_s_init=0.0,
        h_r_init=0.0,
    )

    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]

    r_out   = res["r_out"]
    r_gen   = res["r_gen"]
    infil   = res["infil"]
    sa_loss = res["sa_loss"]
    seep    = res["seep_loss"]
    p_rate  = p_rate_input

    q_mod_m3s = r_out * A_BV_M2
    q_mod_m3s_safe = np.clip(q_mod_m3s, 0.0, None)

    # 4) Volumes
    V_mod_m3 = float(np.nansum(q_mod_m3s_safe) * dt)
    V_obs_m3 = float(np.nansum(q_obs_for_vol) * dt)
    V_diff   = float(np.nansum((q_mod_m3s_safe - q_obs_m3s)) * dt)

    print(f"Volume (mod - obs) = {V_diff:.2f} m³")
    print("\n===== BILAN VOLUMES SUR LA PÉRIODE =====")
    print(f"Volume observé V_obs     = {V_obs_m3:.2f} m³")
    print(f"Volume modélisé V_mod    = {V_mod_m3:.2f} m³")
    if V_obs_m3 > 0:
        print(f"Rapport V_mod / V_obs    = {V_mod_m3 / V_obs_m3:.3f}")
    print("=========================================\n")

    # 5) Bilan de masse
    print_mass_balance(res["mass_balance"])

    # 6) Lames et cumuls (mm)
    factor_mm = dt * 1000.0
    P_mm      = p_rate * factor_mm
    ET_mm     = sa_loss * 1000.0
    Infil_mm  = infil * factor_mm
    Seep_mm   = seep * 1000.0
    Runoff_mm = r_out * factor_mm

    P_cum      = np.cumsum(P_mm)
    ET_cum     = np.cumsum(ET_mm)
    Infil_cum  = np.cumsum(Infil_mm)
    Seep_cum   = np.cumsum(Seep_mm)
    Runoff_cum = np.cumsum(Runoff_mm)

    # 7) Sorties
    plots_dir = base_dir.parent / "03_Plots" / "Parking_CSR_Avec_routage" / event_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 7.0 Volumes cumulés
    V_obs_cum = np.cumsum(q_obs_for_vol * dt)
    V_mod_cum = np.cumsum(q_mod_m3s_safe * dt)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_index, V_obs_cum, label="V_obs cumulé", linewidth=1.4)
    ax.plot(time_index, V_mod_cum, label="V_mod cumulé (avec routage)", linewidth=1.4, linestyle="--")
    ax.scatter(time_index[-1], V_obs_cum[-1], s=30, zorder=5)
    ax.scatter(time_index[-1], V_mod_cum[-1], s=30, zorder=5)
    ax.set_ylabel("Volume cumulé (m³)")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "cumuls_V_obs_V_mod_avec_routage.png", dpi=200)
    plt.close(fig)

    # 7.1 Hydrogramme Q_obs / Q_mod + pluie
    fig, axQ = plt.subplots(figsize=(10, 4))
    axQ.plot(time_index, q_obs_m3s, label="Q_obs (m³/s)", linewidth=1.0, alpha=0.7)
    axQ.plot(time_index, q_mod_m3s_safe, label="Q_mod (m³/s)", linewidth=1.2)
    axQ.set_xlabel("Date")
    axQ.set_ylabel("Débit (m³/s)")
    axQ.grid(True, linewidth=0.4, alpha=0.6)

    axP = axQ.twinx()
    dt_days = dt / 86400.0
    axP.bar(time_index, P_mm, width=dt_days * 0.8, align="center", alpha=0.4, label="P (mm/pas)")
    axP.set_ylabel("Pluie (mm/pas)")
    axP.invert_yaxis()
    maxP = np.nanmax(P_mm) if np.nanmax(P_mm) > 0 else 1.0
    axP.set_ylim(maxP * 1.05, 0.0)

    lines1, labels1 = axQ.get_legend_handles_labels()
    lines2, labels2 = axP.get_legend_handles_labels()
    axQ.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle("Q_obs et Q_mod (avec routage) – évènement parking CSR")
    fig.tight_layout()
    fig.savefig(plots_dir / "Q_mod_vs_Q_obs_P_haut_avec_routage.png", dpi=200)
    plt.close(fig)

    # 7.2 États
    fig2, axr = plt.subplots(figsize=(10, 4))
    axr.plot(time_index, h_a, label="h_a (Ia)", linewidth=1.0)
    axr.plot(time_index, h_s, label="h_s (sol)", linewidth=1.0)
    axr.plot(time_index, h_r, label="h_r (surface)", linewidth=1.0)
    axr.set_xlabel("Date")
    axr.set_ylabel("Hauteur d'eau (m)")
    axr.grid(True, linewidth=0.4, alpha=0.6)
    axr.legend(loc="upper left")
    fig2.suptitle("États des réservoirs – avec routage, parking CSR")
    fig2.tight_layout()
    fig2.savefig(plots_dir / "etats_reservoirs_avec_routage.png", dpi=200)
    plt.close(fig2)

    # 7.3 Cumuls
    fig3, axc = plt.subplots(figsize=(10, 4))
    axc.plot(time_index, P_cum,      label="P cumulée", linewidth=1.3)
    axc.plot(time_index, ET_cum,     label="ETP cumulée", linestyle=":", linewidth=1.1)
    axc.plot(time_index, Infil_cum,  label="Infiltration cumulée", linewidth=1.1)
    axc.plot(time_index, Seep_cum,   label="Percolation cumulée", linestyle="--", linewidth=1.1)
    axc.plot(time_index, Runoff_cum, label="Ruissellement cumulé (exutoire)", linewidth=1.3)
    axc.set_xlabel("Date")
    axc.set_ylabel("Lame cumulée (mm)")
    axc.grid(True, linewidth=0.4, alpha=0.6)
    axc.legend(loc="upper left")
    fig3.suptitle("Cumuls – avec routage, parking CSR")
    fig3.tight_layout()
    fig3.savefig(plots_dir / "cumuls_P_ET_infil_seep_runoff_avec_routage.png", dpi=200)
    plt.close(fig3)

    print(f"[OK] Figures sauvegardées dans : {plots_dir}")


if __name__ == "__main__":
    main()
