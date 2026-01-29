# -*- coding: utf-8 -*-
"""
SCS-HSM continu avec réservoir de ruissellement vers l'exutoire
+ réservoir souterrain à vidange lente
---------------------------------------------------------------

- Structure des réservoirs :
    * h_a   : réservoir d'abstraction Ia
    * h_s   : réservoir de sol
    * h_r   : réservoir de surface (stock de ruissellement rapide)
    * h_sub : réservoir souterrain (écoulement lent / baseflow)

- Ruissellement de surface :
    * Une fraction de h_r est évacuée vers l'exutoire à chaque pas de temps
      selon une loi de réservoir linéaire :
            r_out = k_runoff * h_r
      => Q_mod_surface = r_out * A_BV_M2   (m³/s)

    * On limite r_out pour ne pas vider plus d'eau que ce qui est disponible :
            r_out <= (h_r/dt + r_gen)
      où r_gen = max(q - infil_from_rain, 0) est le flux de ruissellement généré.

- Partie souterraine :
    * Le réservoir de sol (h_s) perd de l'eau par seepage (perte profonde).
    * Une fraction alpha_sub de l'infiltration alimente le réservoir lent h_sub.
    * Le réservoir h_sub se vide vers l'exutoire selon :
            q_sub = k_sub * h_sub
      => Q_mod_total = (r_out + q_sub) * A_BV_M2

- Paramètres à caler (i_a FIXE, s FIXE) :
    theta = [
        log10(k_infiltr),   # m/s (lié à une capacité d'infiltration en mm/h)
        log10(k_seepage),   # s^-1 (profonde, demi-vie longue)
        log10(k_runoff),    # s^-1 (ruissellement rapide, demi-vie courte)
        log10(k_sub),       # s^-1 (réservoir lent, demi-vie intermédiaire)
        alpha_sub           # [-] fraction de l'infiltration alimentant h_sub
    ]
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl

# Un peu d'optimisation pour l'affichage
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["agg.path.chunksize"] = 10000


# ======================================================================
# 1. Modèle SCS-HSM + réservoir de surface + réservoir souterrain lent
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
    k_sub: float = 1e-6,
    h_a_init: float = 0.0,
    h_s_init: float = 0.0,
    h_r_init: float = 0.0,
    h_sub_init: float = 0.0,
    alpha_sub: float = 0.7,  # fraction de l'infiltration envoyée vers le réservoir lent h_sub
) -> dict:

    # clamp alpha_sub dans [0, 1]
    alpha_sub = max(0.0, min(alpha_sub, 1.0))

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
    h_sub = np.zeros(nt + 1, dtype=float)

    h_a[0] = float(h_a_init)
    h_s[0] = float(h_s_init)
    h_r[0] = float(h_r_init)
    h_sub[0] = float(h_sub_init)

    # --- Flux (nt) ---
    p_store     = np.zeros(nt, dtype=float)
    q           = np.zeros(nt, dtype=float)   # pluie nette après Ia
    infil       = np.zeros(nt, dtype=float)   # infiltration totale (m/s)
    r_gen       = np.zeros(nt, dtype=float)   # ruissellement généré à la surface
    r_out       = np.zeros(nt, dtype=float)   # ruissellement de surface vers exutoire
    q_sub       = np.zeros(nt, dtype=float)   # flux lent vers exutoire
    seep_to_sub = np.zeros(nt, dtype=float)   # infiltration envoyée vers h_sub (m/pas)
    sa_loss     = np.zeros(nt, dtype=float)   # ETP effective sur h_a (m/pas)
    seep_loss   = np.zeros(nt, dtype=float)   # perte profonde depuis h_s (m/pas)

    # ========================
    #  BOUCLE TEMPORELLE
    # ========================
    for n in range(nt):
        p   = p_rate[n]     # [m/s]
        etp = etp_rate[n]   # [m/s]
        p_store[n] = p

        # ----------------------------------------------------------
        # 1) ETP sur h_a (abstraction Ia)
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
        # ----------------------------------------------------------
        h_r_begin = h_r[n]

        # eau disponible (flux) durant le pas :
        water_avail_rate = q_n + max(h_r_begin, 0.0) / dt   # [m/s]
        if water_avail_rate < 0.0:
            water_avail_rate = 0.0

        # infiltration totale (limitée par dispo et potentiel)
        infil_n = max(0.0, min(infil_pot, water_avail_rate))

        # part d'infiltration venant de la pluie nette
        infil_from_rain_rate = min(infil_n, q_n)

        # part d'infiltration venant du réservoir de surface h_r
        infil_from_hr_rate = max(0.0, infil_n - infil_from_rain_rate)

        # ne pas pomper plus que ce qui est dispo dans h_r
        max_from_hr = max(h_r_begin, 0.0) / dt
        if infil_from_hr_rate > max_from_hr:
            infil_from_hr_rate = max_from_hr
            infil_n = infil_from_rain_rate + infil_from_hr_rate

        infil[n] = infil_n

        # ----------------------------------------------------------
        # 5) Décomposition de l'infiltration :
        #    - alpha_sub * infil -> réservoir lent h_sub
        #    - (1 - alpha_sub) * infil -> réservoir de sol h_s
        # ----------------------------------------------------------
        h_sub_begin = h_sub[n]
        h_s_begin   = h_s[n]

        infil_vol = infil_n * dt           # [m]
        infil_to_sub  = alpha_sub * infil_vol
        infil_to_soil = (1.0 - alpha_sub) * infil_vol

        # mise à jour intermédiaire
        h_sub_temp = h_sub_begin + infil_to_sub
        h_s_temp   = h_s_begin   + infil_to_soil

        # seepage profond depuis h_s
        if k_seepage > 0.0:
            h_s_after_seep = h_s_temp * math.exp(-k_seepage * dt)
            seep_total = h_s_temp - h_s_after_seep   # [m]
        else:
            h_s_after_seep = h_s_temp
            seep_total = 0.0

        h_s[n + 1]   = h_s_after_seep
        seep_loss[n] = seep_total
        seep_to_sub[n] = infil_to_sub

        # ----------------------------------------------------------
        # 6) Réservoir lent h_sub -> exutoire
        # ----------------------------------------------------------
        if k_sub > 0.0:
            q_sub_raw = k_sub * max(h_sub_temp, 0.0)   # [m/s]
            q_sub_max = max(h_sub_temp, 0.0) / dt
            q_sub_n = max(0.0, min(q_sub_raw, q_sub_max))
        else:
            q_sub_n = 0.0

        q_sub[n] = q_sub_n

        h_sub_next = h_sub_temp - q_sub_n * dt
        if h_sub_next < 0.0:
            h_sub_next = 0.0
        h_sub[n + 1] = h_sub_next

        # ----------------------------------------------------------
        # 7) Ruissellement généré à la surface
        # ----------------------------------------------------------
        r_gen_n = max(q_n - infil_from_rain_rate, 0.0)   # [m/s]
        r_gen[n] = r_gen_n

        # ----------------------------------------------------------
        # 8) Réservoir de surface h_r
        # ----------------------------------------------------------
        if k_runoff > 0.0:
            r_out_raw = k_runoff * max(h_r_begin, 0.0)   # [m/s]

            water_surface_rate = (
                max(h_r_begin, 0.0) / dt + r_gen_n - infil_from_hr_rate
            )
            if water_surface_rate < 0.0:
                water_surface_rate = 0.0

            r_out_n = min(r_out_raw, water_surface_rate)
            r_out_n = max(r_out_n, 0.0)
        else:
            r_out_n = 0.0

        r_out[n] = r_out_n

        h_r_next = h_r_begin + (r_gen_n - r_out_n - infil_from_hr_rate) * dt
        if h_r_next < 0.0:
            h_r_next = 0.0

        h_r[n + 1] = h_r_next

    # ========================
    #  BILAN DE MASSE
    # ========================
    P_tot    = float(np.nansum(p_rate) * dt)                  # [m]
    Seep_tot = float(np.nansum(seep_loss))                    # [m]
    ET_tot   = float(np.nansum(sa_loss))                      # [m]
    R_tot    = float(np.nansum((r_out + q_sub) * dt))         # [m]

    d_h_a   = h_a[-1]   - h_a[0]
    d_h_s   = h_s[-1]   - h_s[0]
    d_h_r   = h_r[-1]   - h_r[0]
    d_h_sub = h_sub[-1] - h_sub[0]
    delta_storage = d_h_a + d_h_s + d_h_r + d_h_sub

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
        "h_sub": h_sub,
        "q": q,
        "infil": infil,
        "r_gen": r_gen,
        "r_out": r_out,
        "q_sub": q_sub,
        "sa_loss": sa_loss,
        "seep_loss": seep_loss,
        "seep_to_sub": seep_to_sub,
        "mass_balance": mass_balance,
    }


# ======================================================================
# 2. Lecture des séries P & Q
# ======================================================================

def read_rain_series_from_csv(csv_name: str, dt: float = 300.0):
    """
    Lit ../02_Data/csv_name (séparateur ';'), extrait :
      - dateP   : datetimes
      - P_mm    : pluie (mm / pas de dt, typiquement 5 min)
      - Q_ls    : débit mesuré (L/s)

    Renvoie : (time_index, P_mm, p_rate_m_per_s, q_ls)
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

    rain_mm = df["P_mm"].astype(float).fillna(0.0).to_numpy()
    p_rate = rain_mm * 1e-3 / dt  # mm -> m, /dt -> m/s

    q_ls = None
    if "Q_ls" in df.columns:
        q_raw = df["Q_ls"].astype(float).to_numpy()
        q_ls = q_raw   # L/s

    return time_index, rain_mm, p_rate, q_ls


# ======================================================================
# 3. Lecture ETP SAFRAN journalière + projection sur la grille
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

        etp_rate[mask] = per_step_mm * 1e-3 / dt_seconds

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
    return float(np.sqrt(np.mean(diff**2)))


def compute_nash(q_obs: np.ndarray, q_mod: np.ndarray, eps: float = 1e-9) -> float:
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


def compute_log_rmse(q_obs: np.ndarray, q_mod: np.ndarray, eps: float = 1e-9) -> float:
    q_obs = np.asarray(q_obs, dtype=float)
    q_mod = np.asarray(q_mod, dtype=float)

    mask = (
        np.isfinite(q_obs) & np.isfinite(q_mod) &
        (q_obs > 0.0) & (q_mod > 0.0)
    )
    if mask.sum() == 0:
        return 1e6

    log_q_obs = np.log(q_obs[mask] + eps)
    log_q_mod = np.log(q_mod[mask] + eps)

    diff = log_q_mod - log_q_obs
    return float(np.sqrt(np.mean(diff**2)))


def objective(theta: np.ndarray, data: dict) -> float:
    """
    Fonction objectif à MINIMISER pour le modèle avec réservoir de surface
    + réservoir lent h_sub.

    Paramètres calibrés :
        theta = [ log10_k_infiltr, log10_k_seepage, log10_k_runoff, log10_k_sub, alpha_sub ]

    Critère d'ajustement :
        J = RMSE( log(Q_mod_total) - log(Q_obs) )
        où Q_mod_total = (r_out + q_sub) * A_BV_M2
    """
    log10_k_infiltr, log10_k_seepage, log10_k_runoff, log10_k_sub, alpha_sub = theta

    s         = data["s_fixed"]
    i_a       = data["i_a_fixed"]
    A_BV_M2   = data["A_BV_M2"]
    dt        = data["dt"]
    p_rate    = data["p_rate"]
    etp_rate  = data["etp_rate"]
    q_obs     = data["q_obs_m3s"]

    # --- Garde-fous basés sur les bornes physiques définies plus haut ---
    if not (LOG10_KINF_MIN  < log10_k_infiltr < LOG10_KINF_MAX):
        return 1e6

    if not (LOG10_KSEEP_MIN < log10_k_seepage < LOG10_KSEEP_MAX):
        return 1e6

    if not (LOG10_KRUN_MIN  < log10_k_runoff  < LOG10_KRUN_MAX):
        return 1e6

    if not (LOG10_KSUB_MIN  < log10_k_sub     < LOG10_KSUB_MAX):
        return 1e6

    if (alpha_sub <= 0.0) or (alpha_sub >= 1.0):
        return 1e6

    # Passage en espace physique
    k_seepage = 10.0**log10_k_seepage
    k_infiltr = 10.0**log10_k_infiltr
    k_runoff  = 10.0**log10_k_runoff
    k_sub     = 10.0**log10_k_sub

    # Simulation
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
            k_sub=k_sub,
            h_a_init=0.0,
            h_s_init=0.0,
            h_r_init=0.0,
            h_sub_init=0.0,
            alpha_sub=alpha_sub,
        )
    except Exception:
        return 1e6

    r_out = res["r_out"]        # [m/s]
    q_sub = res["q_sub"]        # [m/s]
    q_mod = (r_out + q_sub) * A_BV_M2     # [m³/s]

    J = compute_log_rmse(q_obs, q_mod, eps=1e-9)
    if (J is None) or (not np.isfinite(J)):
        return 1e6

    return float(J)


def sample_random_theta(bounds: list[tuple[float, float]]) -> np.ndarray:
    return np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)


def calibrate_multistart(
    data: dict,
    bounds: list[tuple[float, float]],
    n_starts: int = 15,
) -> tuple[np.ndarray, float]:
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
# 5. Bilan de masse
# ======================================================================

def print_mass_balance(mb: dict):
    print("\n=== Bilan de masse sur la période ===")
    print(f"P_tot          = {mb['P_tot_m']*1000:.1f} mm")
    print(f"Ruissellement  = {mb['R_tot_m']*1000:.1f} mm (total vers exutoire)")
    print(f"Seepage profond= {mb['Seep_tot_m']*1000:.1f} mm")
    print(f"ETP effective  = {mb['ET_tot_m']*1000:.1f} mm")
    print(f"ΔStock (Ia+sol+surf+sub) = {mb['Delta_storage_m']*1000:.2f} mm")
    print(
        f"Erreur de fermeture   = {mb['Closure_error_mm']:.3f} mm "
        f"({mb['Relative_error_%']:.3f} %)"
    )

# ======================================================================
#  Bornes physiques (mm/h et demi-vies en heures)
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
# bis. Définition physique des bornes de calibration
# ======================================================================

# 1) Infiltration : vitesse d'infiltration en mm/h 
INFIL_MIN_MM_H = 20.0    
INFIL_MAX_MM_H = 200.0   
LOG10_KINF_MIN, LOG10_KINF_MAX = infil_bounds_mm_h_to_log10k(
    INFIL_MIN_MM_H, INFIL_MAX_MM_H
)

# 2) Ruissellement rapide (k_runoff) : demi-vie en heures

T_HALF_RUNOFF_MIN_H = 0.5
T_HALF_RUNOFF_MAX_H = 1.5
LOG10_KRUN_MIN, LOG10_KRUN_MAX = half_life_bounds_to_log10k(
    T_HALF_RUNOFF_MIN_H, T_HALF_RUNOFF_MAX_H
)

# 3) Réservoir lent (k_sub) : demi-vie en heures
T_HALF_SUB_MIN_H = 1.5
T_HALF_SUB_MAX_H = 12
LOG10_KSUB_MIN, LOG10_KSUB_MAX = half_life_bounds_to_log10k(
    T_HALF_SUB_MIN_H, T_HALF_SUB_MAX_H
)

# 4) Seepage profond (k_seepage) : demi-vie en heures
#    Encore plus lent, typiquement plusieurs jours / semaines.
T_HALF_SEEP_MIN_H = 18
T_HALF_SEEP_MAX_H = 64   
LOG10_KSEEP_MIN, LOG10_KSEEP_MAX = half_life_bounds_to_log10k(
    T_HALF_SEEP_MIN_H, T_HALF_SEEP_MAX_H
)


def k_from_tau(tau_hours):
    tau_seconds = tau_hours * 3600.0
    return np.log(2.0) / tau_seconds

def infil_mm_h_to_m_s(v_mm_h):
    return v_mm_h * 1e-3 / 3600.0

def tau_from_k_seconds(k):
    if k <= 0:
        return np.inf

    tau_s = 1.0 / k      # secondes
    tau_min = tau_s / 60
    tau_h = tau_s / 3600

    return tau_s, tau_min, tau_h


# ======================================================================
# 6. MAIN
# ======================================================================

def main():
    dt = 300.0  # pas de temps = 5 min
    csv_event = "all_events1/2024/event_2024_003.csv"
    csv_etp = "ETP_SAFRAN_J.csv"
    A_BV_M2 = 810000.0
    n_starts = 50
    I_A_FIXED = 0.003  # m
    S_FIXED = 0.15     # m
    event_name = Path(csv_event).stem

    # Affichage des bornes physiques (pour ta compréhension / ton rapport)
    print("=== Bornes physiques retenues pour la calibration ===")
    print(f"k_infiltr  : {INFIL_MIN_MM_H:.0f}–{INFIL_MAX_MM_H:.0f} mm/h"
          f"  => log10(k) ∈ [{LOG10_KINF_MIN:.2f}, {LOG10_KINF_MAX:.2f}]")
    print(f"k_runoff   : t1/2 ∈ [{T_HALF_RUNOFF_MIN_H:.1f}, {T_HALF_RUNOFF_MAX_H:.1f}] h"
          f" => log10(k) ∈ [{LOG10_KRUN_MIN:.2f}, {LOG10_KRUN_MAX:.2f}]")
    print(f"k_sub      : t1/2 ∈ [{T_HALF_SUB_MIN_H:.1f}, {T_HALF_SUB_MAX_H:.1f}] h"
          f" => log10(k) ∈ [{LOG10_KSUB_MIN:.2f}, {LOG10_KSUB_MAX:.2f}]")
    print(f"k_seepage  : t1/2 ∈ [{T_HALF_SEEP_MIN_H:.1f}, {T_HALF_SEEP_MAX_H:.1f}] h"
          f" => log10(k) ∈ [{LOG10_KSEEP_MIN:.2f}, {LOG10_KSEEP_MAX:.2f}]")
    print("====================================================\n")

    # 1) Lecture des données
    time_index, rain_mm, p_rate_input, q_obs = read_rain_series_from_csv(csv_event, dt)
    etp_rate = read_etp_series_for_time_index(csv_etp, time_index)

    if q_obs is None:
        raise RuntimeError("Pas de colonne Q_ls dans le CSV, impossible de caler le modèle.")

    q_obs_m3s = np.asarray(q_obs, dtype=float) / 1000.0  # L/s -> m³/s

    # pour les volumes : NaN -> 0, négatifs clampés à 0
    q_obs_for_vol = np.where(
        np.isfinite(q_obs_m3s),
        np.clip(q_obs_m3s, 0.0, None),
        0.0
    )

    # 2) Calage
    DO_CALIBRATION = False

    if DO_CALIBRATION:
        bounds = [
            (LOG10_KINF_MIN,  LOG10_KINF_MAX),   # log10(k_infiltr) [m/s]
            (LOG10_KSEEP_MIN, LOG10_KSEEP_MAX),  # log10(k_seepage) [s^-1]
            (LOG10_KRUN_MIN,  LOG10_KRUN_MAX),   # log10(k_runoff)  [s^-1]
            (LOG10_KSUB_MIN,  LOG10_KSUB_MAX),   # log10(k_sub)     [s^-1]
            (0.2, 0.8),                         # alpha_sub        [-]
        ]

        data = {
            "dt": dt,
            "p_rate": p_rate_input,
            "etp_rate": etp_rate,
            "q_obs_m3s": q_obs_m3s,
            "A_BV_M2": A_BV_M2,
            "i_a_fixed": I_A_FIXED,
            "s_fixed": S_FIXED,
        }

        print("Lancement du calage (multistart + Powell) sur J = log-RMSE...")
        theta_opt, J_opt = calibrate_multistart(data, bounds, n_starts)

        (
            log10_ki_opt,
            log10_ks_opt,
            log10_kr_opt,
            log10_kb_opt,
            alpha_sub_opt,
        ) = theta_opt

        k_seepage_opt = 10.0 ** log10_ks_opt
        k_infiltr_opt = 10.0 ** log10_ki_opt
        k_runoff_opt  = 10.0 ** log10_kr_opt
        k_sub_opt     = 10.0 ** log10_kb_opt

        # Durées caractéristiques en heures
        t12_runoff_h = math.log(2.0) / k_runoff_opt   / 3600.0
        t12_sub_h    = math.log(2.0) / k_sub_opt      / 3600.0
        t12_seep_h   = math.log(2.0) / k_seepage_opt  / 3600.0

        # Capacité d'infiltration en mm/h
        infil_mm_h   = k_infiltr_opt * 3600.0 * 1000.0

        print("\n=== Résultats du calage ===")
        print(f"J_opt          = {J_opt:.4f} (-)")
        print(f"  i_a          = {I_A_FIXED:.6f} m")
        print(f"  s            = {S_FIXED:.6f} m")
        print(f"  k_infiltr    = {k_infiltr_opt:.3e} m/s "
              f"({infil_mm_h:.1f} mm/h)")
        print(f"  k_seepage    = {k_seepage_opt:.3e} s^-1 "
              f"(t1/2 ≈ {t12_seep_h:.1f} h)")
        print(f"  k_runoff     = {k_runoff_opt:.3e} s^-1 "
              f"(t1/2 ≈ {t12_runoff_h:.2f} h)")
        print(f"  k_sub        = {k_sub_opt:.3e} s^-1 "
              f"(t1/2 ≈ {t12_sub_h:.1f} h)")
        print(f"  alpha_sub    = {alpha_sub_opt:.3f} (-)")

        i_a = I_A_FIXED
        s = S_FIXED
        k_infiltr = k_infiltr_opt
        k_seepage = k_seepage_opt
        k_runoff  = k_runoff_opt
        k_sub     = k_sub_opt
        alpha_sub = float(max(0.0, min(alpha_sub_opt, 1.0)))
    else:
        # Valeurs par défaut si tu ne veux pas caler à chaque fois
        i_a = I_A_FIXED
        s = S_FIXED
        k_infiltr = 3.22222222222223e-04
        k_seepage = 3.697e-06
        k_runoff  = 6.418e-4
        k_sub     = 1.605e-05
        alpha_sub = 0.1

    # 3) Simulation avec les paramètres retenus
    res = run_scs_hsm(
        dt=dt,
        p_rate=p_rate_input,
        etp_rate=etp_rate,
        i_a=i_a,
        s=s,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
        k_runoff=k_runoff,
        k_sub=k_sub,
        h_a_init=0.0,
        h_s_init=0.0,
        h_r_init=0.0,
        h_sub_init=0.0,
        alpha_sub=alpha_sub,
    )

    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]
    h_sub = res["h_sub"][:-1]

    r_out   = res["r_out"]
    q_sub   = res["q_sub"]
    r_gen   = res["r_gen"]
    infil   = res["infil"]
    sa_loss = res["sa_loss"]
    seep    = res["seep_loss"]
    p_rate  = p_rate_input

    q_mod_m3s = (r_out + q_sub) * A_BV_M2

    # 4) Bilan volumique
    V_mod_m3 = float(np.nansum(np.clip(q_mod_m3s, 0.0, None)) * dt)
    V_obs_m3 = float(np.sum(q_obs_for_vol) * dt)

    print("\n===== BILAN VOLUMES SUR LA PÉRIODE =====")
    print(f"Volume observé V_obs     = {V_obs_m3:.1f} m³")
    print(f"Volume modélisé V_mod    = {V_mod_m3:.1f} m³")
    if V_obs_m3 > 0:
        print(f"Rapport V_mod / V_obs    = {V_mod_m3 / V_obs_m3:.3f}")
    print("=========================================\n")

    # 5) Bilan de masse
    print_mass_balance(res["mass_balance"])

    # 6) Conversion en mm/pas et cumuls
    factor_mm_5min = dt * 1000.0

    P_mm_5      = p_rate * factor_mm_5min
    ET_mm_5     = sa_loss * 1000.0
    Infil_mm_5  = infil * factor_mm_5min
    Seep_mm_5   = seep * 1000.0
    Runoff_mm_5 = (r_out + q_sub) * factor_mm_5min

    P_cum_mm      = np.cumsum(P_mm_5)
    ET_cum_mm     = np.cumsum(ET_mm_5)
    Infil_cum_mm  = np.cumsum(Infil_mm_5)
    Seep_cum_mm   = np.cumsum(Seep_mm_5)
    Runoff_cum_mm = np.cumsum(Runoff_mm_5)

    # 7) Figures
    base_dir = Path(__file__).resolve().parent
    plots_dir = (base_dir.parent/ "03_Plots"/ "Avec Routage & Réservoir Laggé"/ event_name)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 7.0 Volumes cumulés obs / mod
    V_obs_cum = np.cumsum(q_obs_for_vol * dt)
    V_mod_cum = np.cumsum(np.clip(q_mod_m3s, 0.0, None) * dt)

    print("\n=== Check volumes cumulés ===")
    print(f"  V_obs_cum final = {V_obs_cum[-1]:.1f} m3")
    print(f"  V_mod_cum final = {V_mod_cum[-1]:.1f} m3")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_index, V_obs_cum, label="V_obs cumulé", linewidth=1.6)
    ax.plot(time_index, V_mod_cum, label="V_mod cumulé", linewidth=1.6, linestyle="--")
    ax.scatter(time_index[-1], V_obs_cum[-1], s=30, zorder=5)
    ax.scatter(time_index[-1], V_mod_cum[-1], s=30, zorder=5)
    ax.set_ylabel("Volume cumulé (m³)")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "cumuls_V_obs_V_mod.png", dpi=200)
    plt.close(fig)

    # 7.1 Hydrogramme
    fig, axQ = plt.subplots(figsize=(12, 4))
    axQ.plot(time_index, q_obs_m3s, label="Q_obs (m³/s)", linewidth=1.0, alpha=0.7)
    axQ.plot(time_index, q_mod_m3s, label="Q_mod total (m³/s)", linewidth=1.2)
    axQ.set_xlabel("Date")
    axQ.set_ylabel("Débit (m³/s)")
    axQ.grid(True, linewidth=0.4, alpha=0.6)

    axP = axQ.twinx()
    dt_days = dt / 86400.0
    axP.bar(
        time_index,
        P_mm_5,
        width=dt_days * 0.8,
        align="center",
        alpha=0.4,
        label="P (mm / pas)",
    )
    axP.set_ylabel("Pluie (mm / pas)")
    axP.invert_yaxis()
    maxP = np.nanmax(P_mm_5) if np.nanmax(P_mm_5) > 0 else 1.0
    axP.set_ylim(maxP * 1.05, 0.0)

    lines1, labels1 = axQ.get_legend_handles_labels()
    lines2, labels2 = axP.get_legend_handles_labels()
    axQ.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle("Q_obs et Q_mod avec pluie en haut")
    fig.tight_layout()
    fig.savefig(plots_dir / "Q_mod_vs_Q_obs_P_runoff_top.png", dpi=200)
    plt.close(fig)

    # 7.2 États des réservoirs
    fig2, axr = plt.subplots(figsize=(12, 4))
    axr.plot(time_index, h_a,   label="h_a (Ia)",          linewidth=1.0)
    axr.plot(time_index, h_s,   label="h_s (sol)",         linewidth=1.0)
    axr.plot(time_index, h_r,   label="h_r (surface)",     linewidth=1.0)
    axr.plot(time_index, h_sub, label="h_sub (souterrain)", linewidth=1.0)
    axr.set_xlabel("Date")
    axr.set_ylabel("Hauteur d'eau (m)")
    axr.grid(True, linewidth=0.4, alpha=0.6)
    axr.legend(loc="upper left")
    fig2.suptitle("États des réservoirs (Ia, sol, surface, souterrain)")
    fig2.tight_layout()
    fig2.savefig(plots_dir / "etats_reservoirs_runoff_baseflow.png", dpi=200)
    plt.close(fig2)

    # 7.3 Cumuls P / ETP / infiltration / seepage / ruissellement
    fig3, axc = plt.subplots(figsize=(12, 4))
    axc.plot(time_index, P_cum_mm,      label="P cumulée",               linewidth=1.3)
    axc.plot(time_index, ET_cum_mm,     label="ETP effective cumulée",   linestyle=":",  linewidth=1.1)
    axc.plot(time_index, Infil_cum_mm,  label="Infiltration cumulée",    linewidth=1.1)
    axc.plot(time_index, Seep_cum_mm,   label="Percolation profonde cumulée", linestyle="--", linewidth=1.1)
    axc.plot(time_index, Runoff_cum_mm, label="Ruissellement total cumulé",    linewidth=1.3)
    axc.set_xlabel("Date")
    axc.set_ylabel("Lame cumulée (mm)")
    axc.grid(True, linewidth=0.4, alpha=0.6)
    axc.legend(loc="upper left")
    fig3.suptitle("Cumuls P / ETP / infiltration / percolation / ruissellement (évènement)")
    fig3.tight_layout()
    fig3.savefig(plots_dir / "cumuls_P_ETP_infil_seep_runoff_event.png", dpi=200)
    plt.close(fig3)

    print(f"[OK] Figures sauvegardées dans : {plots_dir}")


if __name__ == "__main__":
    main()
