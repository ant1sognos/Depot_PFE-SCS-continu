# -*- coding: utf-8 -*-
"""
SCS-HSM continu avec réservoir de ruissellement vers l'exutoire
AVEC ROUTAGE 
---------------------------------------------------------------

- Structure des réservoirs :
    * h_a : réservoir d'abstraction Ia
    * h_s : réservoir de sol
    * h_r : réservoir de surface 

    * Une fraction de h_r est évacuée vers l'exutoire à chaque pas de temps
      selon une loi de réservoir linéaire :
            r_out = k_runoff * h_r
      => Q_mod = r_out * A_BV_M2   (m3/s)

    * On limite r_out pour ne pas vider plus d'eau que ce qui est disponible :
            r_out <= (h_r/dt + r_gen)
      où r_gen = max(q - infil, 0) est le flux de ruissellement généré.

- Paramètres à caler :
    theta = [log10(k_infiltr), log10(k_seepage), log10(k_runoff)]

  avec i_a fixé (valeur définie dans main via I_A_FIXED).

- Fichiers attendus :
    ../02_Data/PQ_BV_Cloutasse.csv
       - colonnes : dateP, P_mm, Q_ls
    ../02_Data/ETP_SAFRAN_J.csv
       - colonnes : DATE (AAAAMMJJ), ETP (mm/jour)
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["agg.path.chunksize"] = 10000


# ======================================================================
# 1. Modèle SCS-HSM + réservoir de ruissellement vers exutoire
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
    """
    Modèle SCS-HSM continu avec réservoir de surface connecté à l'exutoire.

    Schéma :
        1) ETP sur h_a
        2) Ia -> pluie nette q
        3) Infiltration potentielle dans le sol (loi HSM)
        4) Infiltration limitée par l'eau dispo (q + h_r/dt)
        5) Mise à jour du sol + seepage
        6) Génération de ruissellement : r_gen = max(q - infil, 0)
        7) Réservoir de surface h_r :
              - reçoit r_gen
              - perd un flux vers l'exutoire : r_out = k_runoff * h_r
                (limité pour ne pas vider plus que dispo)
        8) Q_mod = r_out * A_BV    (A_BV est géré en dehors de cette fonction)

    Sortie : dict avec
        - t, p
        - h_a, h_s, h_r
        - q, infil
        - r_gen : ruissellement généré (m/s)
        - r_out : ruissellement à l'exutoire (m/s) -> utilisé pour Q_mod
        - sa_loss, seep_loss
        - mass_balance
    """

    # --- Prétraitement ---
    p_rate = np.nan_to_num(np.asarray(p_rate, dtype=float), nan=0.0)
    nt = len(p_rate)

    if etp_rate is None:
        etp_rate = np.zeros(nt, dtype=float)
    else:
        etp_rate = np.nan_to_num(np.asarray(etp_rate, dtype=float), nan=0.0)
        if len(etp_rate) != nt:
            raise ValueError("etp_rate doit avoir la même longueur que p_rate")

    # --- Temps ---
    t = np.array([i * dt for i in range(nt + 1)], dtype=float)

    # --- États (nt+1) ---
    h_a = np.zeros(nt + 1, dtype=float)
    h_s = np.zeros(nt + 1, dtype=float)
    h_r = np.zeros(nt + 1, dtype=float)

    h_a[0] = float(h_a_init)
    h_s[0] = float(h_s_init)
    h_r[0] = float(h_r_init)

    # --- Flux (nt) ---
    p_store   = np.zeros(nt, dtype=float)
    q         = np.zeros(nt, dtype=float)   # pluie nette après Ia
    infil     = np.zeros(nt, dtype=float)   # infiltration sol
    r_gen     = np.zeros(nt, dtype=float)   # ruissellement généré à la surface
    r_out     = np.zeros(nt, dtype=float)   # ruissellement à l'exutoire
    sa_loss   = np.zeros(nt, dtype=float)   # ETP effective sur h_a (m par pas)
    seep_loss = np.zeros(nt, dtype=float)   # seepage profond (m par pas)

    # ========================
    #  BOUCLE TEMPORELLE
    # ========================
    for n in range(nt):
        p   = p_rate[n]     # [m/s]
        etp = etp_rate[n]   # [m/s]
        p_store[n] = p

        # ----------------------------------------------------------
        # 1) ETP sur h_a (abstraction)
        # ----------------------------------------------------------
        h_a_0   = h_a[n]
        etp_pot = etp * dt              # [m]
        etp_eff = min(etp_pot, h_a_0)   # ne pas dépasser ce qu'il y a dans Ia
        h_a_after_etp = h_a_0 - etp_eff
        sa_loss[n] = etp_eff

        # ----------------------------------------------------------
        # 2) Réservoir Ia -> pluie nette q_n
        # ----------------------------------------------------------
        h_a_temp = h_a_after_etp + p * dt   # h_a après ajout de P 
        if h_a_temp < i_a:
            q_n = 0.0
            h_a_next = h_a_temp
        else:
            # excès = pluie nette
            q_n = (h_a_temp - i_a) / dt   # [m/s]
            h_a_next = i_a

        q[n] = q_n
        h_a[n + 1] = h_a_next

        # ----------------------------------------------------------
        # 3) Infiltration potentielle HSM (sur h_s)
        # ----------------------------------------------------------
        h_s_begin = h_s[n]
        X_begin = 1.0 - h_s_begin / s
        if X_begin <= 0.0:
            X_begin = 1e-12

        X_end = 1.0 / (1.0 / X_begin + k_infiltr * dt / s)
        h_s_end = (1.0 - X_end) * s
        infil_pot = (h_s_end - h_s_begin) / dt   # [m/s]

        # ----------------------------------------------------------
        # 4) Limitation par l'eau dispo en surface
        #    (pluie nette q + stock de surface h_r)
        # ----------------------------------------------------------
        h_r_begin = h_r[n]
        water_avail_rate = q_n + h_r_begin / dt   # [m/s]
        if water_avail_rate < 0.0:
            water_avail_rate = 0.0

        infil_n = max(0.0, min(infil_pot, water_avail_rate))
        infil[n] = infil_n

        # Part d'infiltration qui vient de la pluie nette
        infil_from_rain = min(infil_n, q_n)
        # Part d'infiltration qui vient du stock de surface h_r
        infil_from_hr_rate = max(0.0, infil_n - infil_from_rain)
        # On ne peut pas pomper plus que ce qui est dispo dans h_r
        infil_from_hr_rate = min(infil_from_hr_rate, h_r_begin / dt)

        # ----------------------------------------------------------
        # 5) Mise à jour du sol + seepage
        # ----------------------------------------------------------
        h_s_temp = h_s_begin + infil_n * dt
        if k_seepage > 0.0:
            h_s_after_seep = h_s_temp * math.exp(-k_seepage * dt)
            seep = h_s_temp - h_s_after_seep
        else:
            h_s_after_seep = h_s_temp
            seep = 0.0

        h_s[n + 1] = h_s_after_seep
        seep_loss[n] = seep

        # ----------------------------------------------------------
        # 6) Ruissellement généré à la surface
        # ----------------------------------------------------------
        #    on ne retire à la pluie que l'infiltration venant de la pluie
        r_gen_n = max(q_n - infil_from_rain, 0.0)   # [m/s]
        r_gen[n] = r_gen_n

        # ----------------------------------------------------------
        # 7) Réservoir de surface h_r
        #    - reçoit r_gen
        #    - perd r_out = k_runoff * h_r_begin vers l'exutoire
        #      (limité pour ne pas vider plus que dispo)
        # ----------------------------------------------------------
        if k_runoff > 0.0:
            r_out_raw = k_runoff * h_r_begin     # [m/s]
            
            # flux max disponible à la surface pendant le pas
            water_surface_rate = h_r_begin / dt + r_gen_n - infil_from_hr_rate
            if water_surface_rate < 0.0:
                water_surface_rate = 0.0
           
            r_out_n = min(r_out_raw, water_surface_rate)
            r_out_n = max(r_out_n, 0.0)
        else:
            r_out_n = 0.0

        r_out[n] = r_out_n

        # Mise à jour de h_r : on enlève aussi l'infiltration pompée dans h_r
        h_r_next = h_r_begin + (r_gen_n - r_out_n - infil_from_hr_rate) * dt
        if h_r_next < 0.0:
            h_r_next = 0.0

        h_r[n + 1] = h_r_next

    # ========================
    #  BILAN DE MASSE
    # ========================
    P_tot    = float(np.nansum(p_rate) * dt)   # [m]
    Seep_tot = float(np.nansum(seep_loss))     # [m]
    ET_tot   = float(np.nansum(sa_loss))       # [m]
    R_tot    = float(np.nansum(r_out) * dt)    # [m] ruissellement VERS EXUTOIRE

    d_h_a = h_a[-1] - h_a[0]
    d_h_s = h_s[-1] - h_s[0]
    d_h_r = h_r[-1] - h_r[0]
    delta_storage = d_h_a + d_h_s + d_h_r

    # fermeture = P_tot - (ET + Seep + R + Δstocks)
    closure = P_tot - (Seep_tot + ET_tot + R_tot + delta_storage)

    mass_balance = {
        "P_tot_m": P_tot,
        "R_tot_m": R_tot,
        "Seep_tot_m": Seep_tot,
        "ET_tot_m": ET_tot,
        "Delta_storage_m": delta_storage,
        "Closure_error_m": closure,
        "Closure_error_mm": closure * 1000.0,
        "Relative_error_%": 100.0 * closure / P_tot if P_tot > 0 else np.nan,
    }

    return {
        "t": t,
        "p": p_store,
        "h_a": h_a,
        "h_s": h_s,
        "h_r": h_r,
        "q": q,
        "infil": infil,
        "r_gen": r_gen,
        "r_out": r_out,          # <-- flux vers exutoire (m/s)
        "sa_loss": sa_loss,
        "seep_loss": seep_loss,
        "mass_balance": mass_balance,
    }


# ======================================================================
# 2. Lecture des séries P & Q
# ======================================================================

def read_rain_series_from_csv(csv_name: str, dt: float = 300.0):
    """
    Lit ../02_Data/csv_name (séparateur ';'), extrait :
      - dateP   : datetimes
      - P_mm    : pluie (mm / 5 min)  (NA -> 0)
      - Q_ls    : débit mesuré (L/s)

    Renvoie : (time_index, P_mm_5, p_rate_m_per_s, q_ls)
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "02_Data"
    csv_path = data_dir / csv_name

    df = pd.read_csv(
        csv_path,
        sep=";",
        na_values=["NA", "NaN", "", -9999, -9999.0],
    )

    time_series = pd.to_datetime(df["dateP"])
    time_index = pd.DatetimeIndex(time_series)

    rain_5min_mm = df["P_mm"].astype(float).fillna(0.0).to_numpy()
    p_rate = rain_5min_mm * 1e-3 / dt  # mm/5min -> m/s

    q_ls = None
    if "Q_ls" in df.columns:
        q_raw = df["Q_ls"].astype(float).to_numpy()
        q_ls = q_raw   # en L/s (conversion en m³/s plus loin)

    return time_index, rain_5min_mm, p_rate, q_ls


# ======================================================================
# 3. Lecture ETP SAFRAN journalière + projection sur la grille 5 min
# ======================================================================

def read_etp_series_for_time_index(
    etp_csv_name: str,
    time_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Lit ../02_Data/etp_csv_name (séparateur ';') contenant :
      - DATE : AAAAMMJJ
      - ETP  : mm/jour

    Renvoie etp_rate (m/s) sur la grille temporelle time_index (NA -> 0),
    avec une répartition sinusoïdale intra-journalière.
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "02_Data"  
    etp_path = data_dir / etp_csv_name

    df_etp = pd.read_csv(
        etp_path,
        sep=";",
        na_values=["NA", "NaN", "", -9999, -9999.0],
    )

    date_col = None
    for c in ["DATE", "Date", "date"]:
        if c in df_etp.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("Colonne date absente dans le fichier ETP.")

    etp_col = None
    for c in ["ETP", "etp", "Etp"]:
        if c in df_etp.columns:
            etp_col = c
            break
    if etp_col is None:
        raise ValueError("Colonne ETP absente dans le fichier ETP.")

    df_etp[date_col] = pd.to_datetime(df_etp[date_col].astype(str), format="%Y%m%d")
    df_etp[date_col] = df_etp[date_col].dt.normalize()
    df_etp[etp_col] = df_etp[etp_col].astype(float).fillna(0.0)

    etp_dict = dict(zip(df_etp[date_col], df_etp[etp_col]))

    dates_only = time_index.normalize()
    etp_mm_per_day = np.array(
        [etp_dict.get(d, 0.0) for d in dates_only],
        dtype=float
    )

    # Pas de temps en secondes
    if len(time_index) >= 2:
        dt_seconds = (time_index[1] - time_index[0]).total_seconds()
    else:
        dt_seconds = 300.0

    nt = len(time_index)
    etp_rate = np.zeros(nt, dtype=float)

    unique_dates, inverse = np.unique(dates_only, return_inverse=True)

    for j, d in enumerate(unique_dates):
        mask = (inverse == j)
        n_steps = mask.sum()
        if n_steps == 0:
            continue

        mm_day = etp_mm_per_day[mask][0]
        if mm_day <= 0.0:
            continue

        t = np.linspace(0.0, 2.0 * np.pi, n_steps, endpoint=False)
        w = np.maximum(np.sin(t - np.pi / 2.0), 0.0)
        w_sum = w.sum()
        if w_sum <= 0.0:
            per_step_mm = np.full(n_steps, mm_day / n_steps)
        else:
            per_step_mm = mm_day * w / w_sum

        etp_rate[mask] = per_step_mm * 1e-3 / dt_seconds   # mm -> m, /s

    return etp_rate


# ======================================================================
# 4. Fonctions de calage 
# ======================================================================

def compute_rmse(q_obs: np.ndarray, q_mod: np.ndarray) -> float:
    q_obs = np.asarray(q_obs, dtype=float)
    q_mod = np.asarray(q_mod, dtype=float)

    mask = np.isfinite(q_obs) & np.isfinite(q_mod)
    if mask.sum() == 0:
        return 1e6

    diff = q_mod[mask] - q_obs[mask]
    rmse = np.sqrt(np.mean(diff**2))
    return float(rmse)

def compute_log_rmse(q_obs: np.ndarray, q_mod: np.ndarray, eps: float = 1e-9) -> float:

    q_obs = np.asarray(q_obs, dtype=float)
    q_mod = np.asarray(q_mod, dtype=float)

    # Masque : là où on a une observation positive
    mask = np.isfinite(q_obs) & np.isfinite(q_mod) & (q_obs > 0.0)

    if mask.sum() == 0:
        return 1e6

    q_obs_m = q_obs[mask]
    q_mod_m = q_mod[mask]

    # On évite que q_mod tombe à 0 => s'il le fait, ça fait log(eps) très négatif.
    q_mod_m_safe = np.maximum(q_mod_m, eps)

    log_q_obs = np.log(q_obs_m + eps)
    log_q_mod = np.log(q_mod_m_safe)

    diff = log_q_mod - log_q_obs
    rmse_log = np.sqrt(np.mean(diff**2))
    return float(rmse_log)


def compute_nash(q_obs: np.ndarray, q_mod: np.ndarray, eps: float = 1e-9) -> float:
    """
    NSE entre Q_obs et Q_mod.
    """
    q_obs = np.asarray(q_obs, dtype=float)
    q_mod = np.asarray(q_mod, dtype=float)

    mask = np.isfinite(q_obs) & np.isfinite(q_mod)
    if mask.sum() < 2:
        return -np.inf

    q_obs_m = q_obs[mask]
    q_mod_m = q_mod[mask]

    denom = np.sum((q_obs_m - np.mean(q_obs_m))**2)
    if denom < eps:
        return -np.inf

    num = np.sum((q_mod_m - q_obs_m)**2)
    nse = 1.0 - num / (denom + eps)
    return float(nse)


# ======================================================================
# 4 bis. Définition physique des bornes de calibration (comme dans A)
# ======================================================================

def half_life_bounds_to_log10k(t_half_min_h: float, t_half_max_h: float) -> tuple[float, float]:
    """
    Convertit un intervalle de demi-vie [heures] en bornes log10(k) [s^-1].

    On fournit t_half_min_h et t_half_max_h (en heures) et on renvoie :

        (log10(k_min), log10(k_max))

    avec :
        k_min = ln(2)/t_half_max  (vidange la plus lente)
        k_max = ln(2)/t_half_min  (vidange la plus rapide)
    """
    t1 = min(t_half_min_h, t_half_max_h)
    t2 = max(t_half_min_h, t_half_max_h)

    # conversion heures -> secondes
    k_max = math.log(2.0) / (t1 * 3600.0)  # s^-1
    k_min = math.log(2.0) / (t2 * 3600.0)  # s^-1

    return math.log10(k_min), math.log10(k_max)


def infil_bounds_mm_h_to_log10k(v_min_mm_h: float, v_max_mm_h: float) -> tuple[float, float]:
    """
    Convertit un intervalle de capacité d'infiltration [mm/h] en log10(k_infiltr) [m/s].

    v_min_mm_h, v_max_mm_h : bornes en mm/h
    k_infiltr = v_mm_h / (1000 * 3600)  (mm -> m, h -> s)
    """
    v1 = min(v_min_mm_h, v_max_mm_h)
    v2 = max(v_min_mm_h, v_max_mm_h)

    k_min = (v1 / 1000.0) / 3600.0  # m/s
    k_max = (v2 / 1000.0) / 3600.0  # m/s

    return math.log10(k_min), math.log10(k_max)


# ======================================================================
# 4 ter. Fonction objectif avec gardes-fous via bornes physiques
# ======================================================================

def objective(theta: np.ndarray, data: dict) -> float:
    """
    Fonction objectif à MINIMISER pour le modèle AVEC_ROUTAGE (sans réservoir lent).

    Paramètres calibrés :
        theta = [log10_k_infiltr, log10_k_seepage, log10_k_runoff]

    Critère d'ajustement :
        J = RMSE( log(Q_mod) - log(Q_obs) )
        où Q_mod = r_out * A_BV_M2
        (calculé uniquement sur les instants où Q_obs et Q_mod > 0 et finis)
    """

    # -------------------------
    # 1. Déballage des params
    # -------------------------
    log10_k_infiltr, log10_k_seepage, log10_k_runoff = theta

    s         = data["S_FIXED"]
    i_a       = data["i_a_fixed"]
    A_BV_M2   = data["A_BV_M2"]
    dt        = data["dt"]
    p_rate    = data["p_rate"]
    etp_rate  = data["etp_rate"]
    q_obs     = data["q_obs_m3s"]

    # -------------------------
    # 2. Gardes-fous sur theta
    #    (utilise les bornes physiques définies plus haut)
    # -------------------------
    if not (LOG10_KINF_MIN  < log10_k_infiltr < LOG10_KINF_MAX):
        return 1e6

    if not (LOG10_KSEEP_MIN < log10_k_seepage < LOG10_KSEEP_MAX):
        return 1e6

    if not (LOG10_KRUN_MIN  < log10_k_runoff  < LOG10_KRUN_MAX):
        return 1e6

    # Passage en espace physique
    k_infiltr = 10.0**log10_k_infiltr
    k_runoff  = 10.0**log10_k_runoff
    k_seepage = 10.0**log10_k_seepage

    # -------------------------
    # 3. Simulation du modèle
    # -------------------------
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
        # Le modèle diverge ou plante -> grosse pénalité
        return 1e6

    # Débit modélisé vers l'exutoire [m³/s]
    r_out = res["r_out"]        # [m/s]
    q_mod = r_out * A_BV_M2     # [m³/s]

    # -------------------------
    # 4. Critère : RMSE du log(Q)
    # -------------------------
    J = compute_log_rmse(q_obs, q_mod, eps=1e-9)

    # Filet de sécurité
    if (J is None) or (not np.isfinite(J)):
        return 1e6

    return float(J)


def sample_random_theta(bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    Tire un vecteur de paramètres aléatoires dans les bornes.
    """
    return np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)


def calibrate_multistart(
    data: dict,
    bounds: list[tuple[float, float]],
    n_starts: int = 15,
) -> tuple[np.ndarray, float]:
    """
    Stratégie multistart + optimisation locale (Powell).
    """
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
# 5. Impression du bilan de masse
# ======================================================================

def print_mass_balance(mb: dict):
    print("\n=== Bilan de masse sur la période ===")
    print(f"P_tot          = {mb['P_tot_m']*1000:.1f} mm")
    print(f"Ruissellement  = {mb['R_tot_m']*1000:.1f} mm (vers exutoire)")
    print(f"Seepage profond= {mb['Seep_tot_m']*1000:.1f} mm")
    print(f"ETP effective  = {mb['ET_tot_m']*1000:.1f} mm")
    print(f"ΔStock (Ia+sol+surf) = {mb['Delta_storage_m']*1000:.2f} mm")
    print(
        f"Erreur de fermeture   = {mb['Closure_error_mm']:.3f} mm "
        f"({mb['Relative_error_%']:.3f} %)"
    )
    
def k_from_tau(tau_hours):

    tau_seconds = tau_hours * 3600.0
    return np.log(2.0) / tau_seconds

def infil_mm_h_to_m_s(v_mm_h):
    return v_mm_h * 1e-3 / 3600.0


# 1) Infiltration : capacité en mm/h (issue mesures double anneau / pluie simulée)
INFIL_MIN_MM_H = 10.0
INFIL_MAX_MM_H = 200.0
LOG10_KINF_MIN, LOG10_KINF_MAX = infil_bounds_mm_h_to_log10k(
    INFIL_MIN_MM_H, INFIL_MAX_MM_H
)

# 2) Ruissellement rapide (k_runoff) : demi-vie en heures
T_HALF_RUNOFF_MIN_H = 1.0
T_HALF_RUNOFF_MAX_H = 5.0
LOG10_KRUN_MIN, LOG10_KRUN_MAX = half_life_bounds_to_log10k(
    T_HALF_RUNOFF_MIN_H, T_HALF_RUNOFF_MAX_H
)

# 3) Seepage profond (k_seepage) : demi-vie en heures (lent, plusieurs dizaines d'heures)
T_HALF_SEEP_MIN_H = 48.0
T_HALF_SEEP_MAX_H = 96.0
LOG10_KSEEP_MIN, LOG10_KSEEP_MAX = half_life_bounds_to_log10k(
    T_HALF_SEEP_MIN_H, T_HALF_SEEP_MAX_H
)



# ======================================================================
# 6. Main : lecture, calage, simulation, figures
# ======================================================================
def main():
    dt = 300.0  # pas de temps = 5 min
    csv_event = "all_events1/2024/event_2024_005.csv"
    csv_etp = "ETP_SAFRAN_J.csv"
    A_BV_M2 = 810000.0
    n_starts = 15
    event_name = Path(csv_event).stem


    # Valeurs FIXES
    I_A_FIXED = 0.003  # m
    S_FIXED   = 0.15   # m

    # Affichage des bornes physiques (pour toi / pour le rapport)
    print("=== Bornes physiques retenues pour la calibration ===")
    print(f"k_infiltr  : {INFIL_MIN_MM_H:.0f}–{INFIL_MAX_MM_H:.0f} mm/h"
          f"  => log10(k) ∈ [{LOG10_KINF_MIN:.2f}, {LOG10_KINF_MAX:.2f}]")
    print(f"k_runoff   : t1/2 ∈ [{T_HALF_RUNOFF_MIN_H:.1f}, {T_HALF_RUNOFF_MAX_H:.1f}] h"
          f" => log10(k) ∈ [{LOG10_KRUN_MIN:.2f}, {LOG10_KRUN_MAX:.2f}]")
    print(f"k_seepage  : t1/2 ∈ [{T_HALF_SEEP_MIN_H:.1f}, {T_HALF_SEEP_MAX_H:.1f}] h"
          f" => log10(k) ∈ [{LOG10_KSEEP_MIN:.2f}, {LOG10_KSEEP_MAX:.2f}]")
    print("====================================================\n")

    # 1) Lecture des données
    time_index, rain_5min_mm, p_rate_input, q_obs = read_rain_series_from_csv(csv_event, dt)
    etp_rate = read_etp_series_for_time_index(csv_etp, time_index)

    if q_obs is None:
        raise RuntimeError("Pas de colonne Q_ls dans le CSV, impossible de caler le modèle.")
        
    # On garde les NaN pour le tracé et la fonction objectif
    q_obs_m3s = np.asarray(q_obs, dtype=float) / 1000.0  # L/s -> m³/s

    # Version dédiée aux volumes : NaN -> 0, et on force les négatifs à 0
    q_obs_for_vol = np.where(np.isfinite(q_obs_m3s), np.clip(q_obs_m3s, 0.0, None), 0.0)

    # 2) Calage
    DO_CALIBRATION = True

    if DO_CALIBRATION:
        # theta = [log10_k_infiltr, log10_k_seepage, log10_k_runoff]
        bounds = [
            (LOG10_KINF_MIN,  LOG10_KINF_MAX),   # log10(k_infiltr)  [m/s]
            (LOG10_KSEEP_MIN, LOG10_KSEEP_MAX),  # log10(k_seepage) [s^-1]
            (LOG10_KRUN_MIN,  LOG10_KRUN_MAX),   # log10(k_runoff)  [s^-1]
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

        print("Lancement du calage (multistart + Powell) sur J log-RMSE...")
        theta_opt, J_opt = calibrate_multistart(data, bounds, n_starts)

        log10_ki_opt, log10_ks_opt, log10_kr_opt = theta_opt

        k_infiltr_opt = 10.0 ** log10_ki_opt
        k_seepage_opt = 10.0 ** log10_ks_opt      
        k_runoff_opt  = 10.0 ** log10_kr_opt

        # Interprétation physique : demi-vies et capacité d'infiltration
        t12_runoff_h = math.log(2.0) / k_runoff_opt  / 3600.0
        t12_seep_h   = math.log(2.0) / k_seepage_opt / 3600.0
        infil_mm_h   = k_infiltr_opt * 3600.0 * 1000.0

        print("\n=== Résultats du calage ===")
        print(f"J_opt        = {J_opt:.4f} (-)")
        print(f"  i_a        = {I_A_FIXED:.6f} m")
        print(f"  s          = {S_FIXED:.6f} m")
        print(f"  k_infiltr  = {k_infiltr_opt:.3e} m/s "
              f"({infil_mm_h:.1f} mm/h)")
        print(f"  k_seepage  = {k_seepage_opt:.3e} s^-1 "
              f"(t1/2 ≈ {t12_seep_h:.1f} h)")
        print(f"  k_runoff   = {k_runoff_opt:.3e} s^-1 "
              f"(t1/2 ≈ {t12_runoff_h:.2f} h)")

        i_a       = I_A_FIXED
        s         = S_FIXED
        k_infiltr = k_infiltr_opt
        k_seepage = k_seepage_opt
        k_runoff  = k_runoff_opt
    else:
        i_a       = I_A_FIXED
        s         = S_FIXED
        k_infiltr = 2.778e-06
        k_seepage = 4.023e-06 
        k_runoff  = 1.034e-05

    # 3) Simulation avec les paramètres choisis
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

    # États (on enlève le dernier point pour aligner sur time_index)
    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]

    # Flux
    r_out   = res["r_out"]        # [m/s] vers exutoire
    r_gen   = res["r_gen"]        # [m/s] généré à la surface
    infil   = res["infil"]        # [m/s]
    sa_loss = res["sa_loss"]      # [m] par pas (ETP sur Ia)
    seep    = res["seep_loss"]    # [m] par pas
    p_rate  = p_rate_input        # [m/s]

    # Débit modélisé
    q_mod_m3s = r_out * A_BV_M2   # m³/s

    # 4) Bilan volumique (pour info)
    V_mod_m3 = float(np.nansum(q_mod_m3s) * dt)
    V_obs_m3 = float(np.sum(q_obs_for_vol) * dt) 

    diff   = q_mod_m3s - q_obs_m3s
    V_diff = float(np.nansum(diff) * dt)
    print(f"Volume (mod - obs) = {V_diff:.1f} m³")

    print("\n===== BILAN VOLUMES SUR LA PÉRIODE =====")
    print(f"Volume observé V_obs     = {V_obs_m3:.1f} m³")
    print(f"Volume modélisé V_mod    = {V_mod_m3:.1f} m³")
    if V_obs_m3 > 0:
        print(f"Rapport V_mod / V_obs    = {V_mod_m3 / V_obs_m3:.3f}")
    print("=========================================\n")

    # 5) Bilan de masse
    print_mass_balance(res["mass_balance"])

    # 6) Conversion en mm / pas et cumuls (P, ETP, infil, seep, ruissellement)
    factor_mm_5min = dt * 1000.0  # m/s * dt -> m, *1000 -> mm

    P_mm_5      = p_rate * factor_mm_5min          # pluie (mm / pas)
    ET_mm_5     = sa_loss * 1000.0                 # ETP effective (mm / pas)
    Infil_mm_5  = infil * factor_mm_5min           # infiltration (mm / pas)
    Seep_mm_5   = seep * 1000.0                    # percolation (mm / pas)
    Runoff_mm_5 = r_out * factor_mm_5min           # ruissellement à l’exutoire (mm / pas)

    P_cum_mm      = np.cumsum(P_mm_5)
    ET_cum_mm     = np.cumsum(ET_mm_5)
    Infil_cum_mm  = np.cumsum(Infil_mm_5)
    Seep_cum_mm   = np.cumsum(Seep_mm_5)
    Runoff_cum_mm = np.cumsum(Runoff_mm_5)

    # 7) Figures
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir.parent / "03_Plots" / "Avec Routage" / event_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 7.0 Volume cumulé observé / modélisé
    V_obs_cum = np.cumsum(q_obs_for_vol * dt)                 # [m³]
    V_mod_cum = np.cumsum(np.clip(q_mod_m3s, 0.0, None) * dt) # [m³]

    print("\n=== Check volumes cumulés ===")
    print(f"  V_obs_cum final = {V_obs_cum[-1]:.1f} m3")
    print(f"  V_mod_cum final = {V_mod_cum[-1]:.1f} m3")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(time_index, V_obs_cum,
            label="V_obs cumulé", linewidth=1.6)
    ax.plot(time_index, V_mod_cum,
            label="V_mod cumulé", linewidth=1.6, linestyle="--")

    # petit marqueur sur le dernier point pour bien voir les deux
    ax.scatter(time_index[-1], V_obs_cum[-1], s=30, zorder=5)
    ax.scatter(time_index[-1], V_mod_cum[-1], s=30, zorder=5)

    ax.set_ylabel("Volume cumulé (m³)")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "cumuls_V_obs_V_mod.png", dpi=200)
    plt.close(fig)

    # 7.1 Hydrogramme Q_mod / Q_obs + pluie "pendue" depuis le haut
    fig, axQ = plt.subplots(figsize=(12, 4))

    # Débits
    axQ.plot(time_index, q_obs_m3s, label="Q_obs (m³/s)", linewidth=1.0, alpha=0.7)
    axQ.plot(time_index, q_mod_m3s, label="Q_mod (m³/s)", linewidth=1.2)

    axQ.set_xlabel("Date")
    axQ.set_ylabel("Débit (m³/s)")
    axQ.grid(True, linewidth=0.4, alpha=0.6)

    # Axe secondaire pour la pluie
    axP = axQ.twinx()
    dt_days = dt / 86400.0

    axP.bar(
        time_index,
        P_mm_5,
        width=dt_days * 0.8,
        align="center",
        alpha=0.4,
        label="P (mm / 5 min)",
    )
    axP.set_ylabel("Pluie (mm / 5 min)")

    # Pluie "accrochée" en haut : inversion de l'axe Y
    axP.invert_yaxis()
    maxP = np.nanmax(P_mm_5) if np.nanmax(P_mm_5) > 0 else 1.0
    axP.set_ylim(maxP * 1.05, 0.0)

    # Légende combinée
    lines1, labels1 = axQ.get_legend_handles_labels()
    lines2, labels2 = axP.get_legend_handles_labels()
    axQ.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle("Q_obs et Q_mod")
    fig.tight_layout()
    fig.savefig(plots_dir / "Q_mod_vs_Q_obs_P_haut.png", dpi=200)
    plt.close(fig)

    # 7.2 États des réservoirs
    fig2, axr = plt.subplots(figsize=(12, 4))
    axr.plot(time_index, h_a, label="h_a (Ia)",  linewidth=1.0)
    axr.plot(time_index, h_s, label="h_s (sol)", linewidth=1.0)
    axr.plot(time_index, h_r, label="h_r (surface)", linewidth=1.0)

    axr.set_xlabel("Date")
    axr.set_ylabel("Hauteur d'eau (m)")
    axr.grid(True, linewidth=0.4, alpha=0.6)
    axr.legend(loc="upper left")
    fig2.suptitle("États des réservoirs (Ia, sol, surface)")
    fig2.tight_layout()
    fig2.savefig(plots_dir / "etats_reservoirs_runoff.png", dpi=200)
    plt.close(fig2)

    # 7.3 Cumuls P / ETP / infiltration / seepage / ruissellement
    fig3, axc = plt.subplots(figsize=(12, 4))

    axc.plot(time_index, P_cum_mm,      label="P cumulée",               linewidth=1.3)
    axc.plot(time_index, ET_cum_mm,     label="ETP effective cumulée",   linestyle=":",  linewidth=1.1)
    axc.plot(time_index, Infil_cum_mm,  label="Infiltration cumulée",    linewidth=1.1)
    axc.plot(time_index, Seep_cum_mm,   label="Percolation cumulée",     linestyle="--", linewidth=1.1)
    axc.plot(time_index, Runoff_cum_mm, label="Ruissellement cumulé",    linewidth=1.3)

    axc.set_xlabel("Date")
    axc.set_ylabel("Lame cumulée (mm)")
    axc.grid(True, linewidth=0.4, alpha=0.6)
    axc.legend(loc="upper left")

    fig3.suptitle("Cumuls P / ETP / infiltration / percolation / ruissellement (évènement)")
    fig3.tight_layout()
    fig3.savefig(plots_dir / "cumuls_P_ETP_infil_seep_runoff_event2.png", dpi=200)
    plt.close(fig3)

    print(f"[OK] Figures sauvegardées dans : {plots_dir}")


if __name__ == "__main__":
    main()
