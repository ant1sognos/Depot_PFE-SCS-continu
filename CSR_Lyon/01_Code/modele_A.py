# -*- coding: utf-8 -*-
"""
SCS–HSM (CONTINU) — MODELE A "SANS ROUTAGE"
==========================================

Objectif :
- Modèle SCS-HSM avec 3 stocks : h_a (Ia), h_s (S), h_r (surface)
- Production du ruissellement généré r_gen = max(q - infil, 0)
- Pas de routage : Q_mod = r_gen * A  (A en m²)

Calibration :
- Ia et S FIXES (cohérent avec CSR)
- Calage uniquement sur k_infiltr (m/s) et k_seepage (s^-1)
- Minimisation d'une log-RMSE sur Q (en m³/s)

Entrées :
- CSV événement dans 02_Data/...
  Colonnes attendues :
    - date ou dateP
    - P_mm
    - Q (optionnel) : Q_LH, Q_LS, Q_m3s, Q_ls, Q_lh ...

Sorties :
- Figures dans 03_Plots/Modele_A_SansRoutage/<event_name>/
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize


# Optimisation affichage
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0


# ======================================================================
# CONFIG (à modifier)
# ======================================================================

DT_S = 300.0                 # pas de temps (s). Si dt dans le CSV, tu peux l'inférer (non fait ici)
A_BV_M2 = 94.0               # CSR parking typiquement ~94 m² (à adapter)
EVENT_CSV_REL = "all_events1/2024/event_2024_014.csv"   # relatif à 02_Data/

# Ia et S FIXES
I_A_FIXED = 0.002            # m
S_FIXED   = 0.13             # m

# ETP : pour de l'événementiel court, tu peux mettre 0 sans honte.
# Si tu veux SAFRAN, remplace read_etp_zero par une lecture SAFRAN.
USE_ETP = False

# Calibration
DO_CALIBRATION = True
N_STARTS = 30

# Bornes sur k_infiltr en mm/h (plus lisible), converties en m/s
KINF_MIN_MM_H = 0.5
KINF_MAX_MM_H = 5.0

# Bornes sur k_seepage en s^-1 (ordre de grandeur à adapter)
KSEEP_MIN = 1e-7
KSEEP_MAX = 1e-4


# ======================================================================
# CONVERSIONS / UTILITAIRES
# ======================================================================

def mm_per_step_to_mps(mm_per_step: np.ndarray, dt_s: float) -> np.ndarray:
    """mm/pas -> m/s"""
    return np.asarray(mm_per_step, dtype=float) * 1e-3 / float(dt_s)

def lh_to_m3s(q_lh: np.ndarray) -> np.ndarray:
    """L/h -> m³/s"""
    return np.asarray(q_lh, dtype=float) / 1000.0 / 3600.0

def ls_to_m3s(q_ls: np.ndarray) -> np.ndarray:
    """L/s -> m³/s"""
    return np.asarray(q_ls, dtype=float) / 1000.0

def safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(float)

def infer_dt_seconds(time_index: pd.DatetimeIndex) -> float:
    diffs = time_index.to_series().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        raise ValueError("Série trop courte pour inférer dt.")
    return float(np.median(diffs))


# ======================================================================
# LECTURE EVENEMENT (P + Q)
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

def read_event_csv(csv_rel: str, sep: str = ";", dt_fallback: float = DT_S):
    """
    Lit un événement et retourne :
    - time_index (DatetimeIndex)
    - P_mm (mm/pas)
    - p_rate (m/s)
    - q_obs_m3s (m³/s) ou None si absent
    - dt (s) : inféré si possible, sinon dt_fallback
    """
    csv_path = BASE_DIR / "02_Data" / Path(csv_rel)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path, sep=sep).copy()

    # Date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        time_index = pd.DatetimeIndex(df["date"])
    elif "dateP" in df.columns:
        df["dateP"] = pd.to_datetime(df["dateP"], errors="raise")
        time_index = pd.DatetimeIndex(df["dateP"])
    else:
        raise ValueError(f"Aucune colonne date/dateP dans {csv_path.name}")

    df = df.sort_values(time_index.name if time_index.name else df.columns[0]).reset_index(drop=True)
    time_index = pd.DatetimeIndex(pd.to_datetime(time_index)).sort_values()

    # dt
    try:
        dt = infer_dt_seconds(time_index)
    except Exception:
        dt = float(dt_fallback)

    # pluie
    if "P_mm" not in df.columns:
        raise ValueError(f"Colonne P_mm manquante dans {csv_path.name}")
    P_mm = safe_numeric(df["P_mm"])
    P_mm = np.clip(P_mm, 0.0, None)
    p_rate = mm_per_step_to_mps(P_mm, dt)

    # débit observé : gestion unités robuste selon colonnes
    q_obs_m3s = None
    q_candidates = list(df.columns)

    # Priorité explicite
    if "Q_m3s" in q_candidates:
        q_obs_m3s = safe_numeric(df["Q_m3s"])
    elif "Q_LH" in q_candidates:
        q_obs_m3s = lh_to_m3s(safe_numeric(df["Q_LH"]))
    elif "Q_lh" in q_candidates:
        q_obs_m3s = lh_to_m3s(safe_numeric(df["Q_lh"]))
    elif "Q_LS" in q_candidates:
        q_obs_m3s = ls_to_m3s(safe_numeric(df["Q_LS"]))
    elif "Q_ls" in q_candidates:
        q_obs_m3s = ls_to_m3s(safe_numeric(df["Q_ls"]))
    elif "Q" in q_candidates:
        # dernier recours : on ne peut pas deviner l'unité, donc on refuse silencieusement
        # et on force l'utilisateur à expliciter
        raise ValueError(
            "Colonne Q trouvée mais unité inconnue. Renomme en Q_LH, Q_LS ou Q_m3s."
        )

    if q_obs_m3s is not None:
        q_obs_m3s = np.clip(np.asarray(q_obs_m3s, dtype=float), 0.0, None)

    return time_index, P_mm, p_rate, q_obs_m3s, dt


def read_etp_zero(time_index: pd.DatetimeIndex) -> np.ndarray:
    """ETP nulle (m/s)"""
    return np.zeros(len(time_index), dtype=float)


# ======================================================================
# MODELE SCS–HSM (sans routage)
# ======================================================================

def run_scs_hsm(
    dt: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray,
    i_a: float,
    s: float,
    k_infiltr: float,   # m/s
    k_seepage: float,   # s^-1
    h_a_init: float = 0.0,
    h_s_init: float = 0.0,
    h_r_init: float = 0.0,
) -> dict:

    p_rate = np.nan_to_num(np.asarray(p_rate, dtype=float), nan=0.0)
    etp_rate = np.nan_to_num(np.asarray(etp_rate, dtype=float), nan=0.0)
    nt = len(p_rate)

    # Etats
    h_a = np.zeros(nt + 1, dtype=float)
    h_s = np.zeros(nt + 1, dtype=float)
    h_r = np.zeros(nt + 1, dtype=float)
    h_a[0], h_s[0], h_r[0] = float(h_a_init), float(h_s_init), float(h_r_init)

    # Flux
    q = np.zeros(nt, dtype=float)          # pluie nette (m/s)
    infil = np.zeros(nt, dtype=float)      # infiltration (m/s)
    r_gen = np.zeros(nt, dtype=float)      # ruissellement généré (m/s)
    sa_loss = np.zeros(nt, dtype=float)    # ETP sur Ia (m/pas)
    seep_loss = np.zeros(nt, dtype=float)  # percolation sol (m/pas)

    for n in range(nt):
        p = float(p_rate[n])
        etp = float(etp_rate[n])

        # 1) ET sur Ia
        etp_eff = min(etp * dt, h_a[n])
        sa_loss[n] = etp_eff
        h_a_after = h_a[n] - etp_eff

        # 2) Ia -> pluie nette q
        h_temp = h_a_after + p * dt
        if h_temp < i_a:
            q_n = 0.0
            h_a[n + 1] = h_temp
        else:
            q_n = (h_temp - i_a) / dt
            h_a[n + 1] = i_a
        q[n] = q_n

        # 3) infiltration potentielle HSM
        h_s_b = h_s[n]
        Xb = 1.0 - h_s_b / s if s > 0 else 1e-12
        if Xb <= 0.0:
            Xb = 1e-12

        Xe = 1.0 / (1.0 / Xb + k_infiltr * dt / s)
        h_s_end = (1.0 - Xe) * s
        infil_pot = (h_s_end - h_s_b) / dt

        # 4) limitation infiltration par eau dispo surface (q + stock h_r)
        water_avail = q_n + h_r[n] / dt
        water_avail = max(water_avail, 0.0)
        infil_n = min(max(infil_pot, 0.0), water_avail)
        infil[n] = infil_n

        # 5) update sol + seepage
        hs_temp = h_s_b + infil_n * dt
        if k_seepage > 0.0:
            h_s_after = hs_temp * math.exp(-k_seepage * dt)
            seep_loss[n] = hs_temp - h_s_after
        else:
            h_s_after = hs_temp
        h_s[n + 1] = h_s_after

        # 6) ruissellement généré
        r_gen_n = max(q_n - infil_n, 0.0)
        r_gen[n] = r_gen_n

        # 7) stock surface (pas de routage, juste stockage interne)
        h_r[n + 1] = max(h_r[n] + (q_n - infil_n) * dt, 0.0)

    # Bilan de masse (diagnostic)
    P_tot = float(np.sum(p_rate) * dt)
    ET_tot = float(np.sum(sa_loss))
    Seep_tot = float(np.sum(seep_loss))
    R_gen_tot = float(np.sum(r_gen) * dt)

    delta_storage = (h_a[-1] - h_a[0]) + (h_s[-1] - h_s[0]) + (h_r[-1] - h_r[0])
    closure = P_tot - (ET_tot + Seep_tot + delta_storage)

    mass_balance = dict(
        P_tot_m=P_tot,
        R_gen_tot_m=R_gen_tot,  # ruissellement généré (pas forcément sorti)
        ET_tot_m=ET_tot,
        Seep_tot_m=Seep_tot,
        Delta_storage_m=delta_storage,
        Closure_error_m=closure,
        Closure_error_mm=closure * 1000.0,
        Relative_error_pct=100.0 * closure / P_tot if P_tot > 0 else np.nan,
    )

    return dict(
        h_a=h_a, h_s=h_s, h_r=h_r,
        q=q, infil=infil, r_gen=r_gen,
        sa_loss=sa_loss, seep_loss=seep_loss,
        mass_balance=mass_balance,
    )


# ======================================================================
# OBJECTIF : LOG-RMSE SUR Q
# ======================================================================

def compute_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 1e9
    d = x[mask] - y[mask]
    return float(np.sqrt(np.mean(d * d)))

def objective_logrmse_theta_log10(theta_log10: np.ndarray, data: dict) -> float:
    """
    theta_log10 = [log10(k_infiltr), log10(k_seepage)]
    Objectif = RMSE( log(Q_mod+eps), log(Q_obs+eps) )
    """
    log10_ki, log10_ks = theta_log10
    (lo_ki, hi_ki), (lo_ks, hi_ks) = data["bounds_log10"]

    if not (lo_ki <= log10_ki <= hi_ki): return 1e9
    if not (lo_ks <= log10_ks <= hi_ks): return 1e9

    k_infiltr = 10.0 ** log10_ki
    k_seepage = 10.0 ** log10_ks
    if (k_infiltr <= 0.0) or (k_seepage < 0.0):
        return 1e9

    try:
        res = run_scs_hsm(
            dt=data["dt"],
            p_rate=data["p_rate"],
            etp_rate=data["etp_rate"],
            i_a=data["i_a"],
            s=data["s"],
            k_infiltr=k_infiltr,
            k_seepage=k_seepage,
        )
    except Exception:
        return 1e9

    # Q_mod (m³/s) : sans routage => r_gen * A
    q_mod_m3s = res["r_gen"] * data["A_BV_M2"]
    q_obs_m3s = data["q_obs_m3s"]

    # --- LOG-RMSE ---
    eps = max(1e-12, 1e-3 * float(np.nanmax(q_obs_m3s)) if np.nanmax(q_obs_m3s) > 0 else 1e-12)
    j = compute_rmse(np.log(q_obs_m3s + eps), np.log(q_mod_m3s + eps))
    if not np.isfinite(j):
        return 1e9
    return float(j)

    # Si tu veux RMSE simple à la place (sans log), remplace par :
    # return compute_rmse(q_obs_m3s, q_mod_m3s)


def sample_uniform(bounds_log10: list[tuple[float, float]]) -> np.ndarray:
    return np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds_log10], dtype=float)

def calibrate_multistart_powell(data: dict, bounds_log10: list[tuple[float, float]], n_starts: int = 20):
    best_x = None
    best_J = np.inf
    for i in range(n_starts):
        x0 = sample_uniform(bounds_log10)
        res = minimize(
            objective_logrmse_theta_log10,
            x0,
            args=(data,),
            method="Powell",
            bounds=bounds_log10,
            options={"maxiter": 250, "disp": False},
        )
        J = float(res.fun) if np.isfinite(res.fun) else 1e9
        print(f"Essai {i+1}/{n_starts} : J(logRMSE) = {J:.6e}")
        if J < best_J:
            best_J = J
            best_x = np.array(res.x, dtype=float)
    return best_x, best_J


# ======================================================================
# BILAN / PLOTS
# ======================================================================

def print_mass_balance(mb: dict):
    print("\n=== BILAN DE MASSE ===")
    print(f"P_tot                  = {mb['P_tot_m']*1000:.2f} mm")
    print(f"Ruissellement généré    = {mb['R_gen_tot_m']*1000:.2f} mm (diagnostic)")
    print(f"Seepage profond         = {mb['Seep_tot_m']*1000:.2f} mm")
    print(f"ETP sur Ia              = {mb['ET_tot_m']*1000:.2f} mm")
    print(f"ΔStock (Ia+sol+surf)    = {mb['Delta_storage_m']*1000:.4f} mm")
    print(f"Erreur fermeture         = {mb['Closure_error_mm']:.4f} mm ({mb['Relative_error_pct']:.4f} %)")

def main():
    # 1) lecture données
    time_index, P_mm, p_rate, q_obs_m3s, dt = read_event_csv(EVENT_CSV_REL, dt_fallback=DT_S)
    event_name = Path(EVENT_CSV_REL).stem

    if q_obs_m3s is None:
        raise RuntimeError("Pas de Q observé dans le CSV (ajoute Q_LH / Q_LS / Q_m3s).")

    # 2) ETP
    etp_rate = read_etp_zero(time_index) if (not USE_ETP) else read_etp_zero(time_index)

    # 3) bornes calibration
    ki_lo = (KINF_MIN_MM_H / 1000.0) / 3600.0
    ki_hi = (KINF_MAX_MM_H / 1000.0) / 3600.0

    bounds_log10 = [
        (math.log10(ki_lo), math.log10(ki_hi)),
        (math.log10(KSEEP_MIN), math.log10(KSEEP_MAX)),
    ]

    # 4) calibration
    if DO_CALIBRATION:
        data = dict(
            dt=dt,
            p_rate=p_rate,
            etp_rate=etp_rate,
            q_obs_m3s=np.asarray(q_obs_m3s, dtype=float),
            A_BV_M2=float(A_BV_M2),
            i_a=float(I_A_FIXED),
            s=float(S_FIXED),
            bounds_log10=bounds_log10,
        )
        print("\n=== Calibration Modèle A (sans routage) ===")
        print(f"Ia fixé = {I_A_FIXED:.6f} m ; S fixé = {S_FIXED:.6f} m")
        print(f"k_infiltr ∈ [{ki_lo:.3e}, {ki_hi:.3e}] m/s ({KINF_MIN_MM_H:.2f}–{KINF_MAX_MM_H:.2f} mm/h)")
        print(f"k_seepage ∈ [{KSEEP_MIN:.1e}, {KSEEP_MAX:.1e}] s^-1\n")

        theta_opt, Jopt = calibrate_multistart_powell(data, bounds_log10, n_starts=N_STARTS)
        log10_ki, log10_ks = theta_opt
        k_infiltr = 10.0 ** log10_ki
        k_seepage = 10.0 ** log10_ks

        print("\n--- Paramètres optimaux ---")
        print(f"k_infiltr = {k_infiltr:.3e} m/s ({k_infiltr*3600*1000:.3f} mm/h)")
        print(f"k_seepage = {k_seepage:.3e} s^-1")
        print(f"J_opt (logRMSE) = {Jopt:.6e}")
    else:
        k_infiltr = 1e-6
        k_seepage = 1e-5

    # 5) simulation finale
    res = run_scs_hsm(
        dt=dt,
        p_rate=p_rate,
        etp_rate=etp_rate,
        i_a=I_A_FIXED,
        s=S_FIXED,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
    )

    r_gen = res["r_gen"]
    q_mod_m3s = r_gen * A_BV_M2

    # bilans
    print_mass_balance(res["mass_balance"])
    V_obs = float(np.sum(q_obs_m3s) * dt)
    V_mod = float(np.sum(q_mod_m3s) * dt)
    print("\n===== BILAN VOLUMES =====")
    print(f"V_obs = {V_obs:.6f} m³")
    print(f"V_mod = {V_mod:.6f} m³")
    if V_obs > 0:
        print(f"Ratio V_mod/V_obs = {V_mod/V_obs:.3f}")

    # 6) plots
    out = BASE_DIR / "03_Plots" / "Modele_A_SansRoutage" / event_name
    out.mkdir(parents=True, exist_ok=True)

    dt_days = dt / 86400.0

    # Figure Q + pluie
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_index, q_obs_m3s, label="Q_obs (m³/s)", lw=1.0, alpha=0.7)
    ax.plot(time_index, q_mod_m3s, label="Q_mod (m³/s) = r_gen*A", lw=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Débit (m³/s)")
    ax.grid(alpha=0.6)

    ax2 = ax.twinx()
    ax2.bar(time_index, P_mm, width=dt_days * 0.9, alpha=0.3, label="Pluie (mm/pas)")
    ax2.invert_yaxis()
    ax2.set_ylabel("Pluie (mm/pas)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out / "Qmod_Qobs_P.png", dpi=200)
    plt.close(fig)

    # Etats
    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]

    fig2, axr = plt.subplots(figsize=(12, 4))
    axr.plot(time_index, h_a, label="h_a (Ia)", lw=1.0)
    axr.plot(time_index, h_s, label="h_s (sol)", lw=1.0)
    axr.plot(time_index, h_r, label="h_r (surface, non routé)", lw=1.0)
    axr.set_xlabel("Date")
    axr.set_ylabel("Hauteur d'eau (m)")
    axr.grid(True, linewidth=0.4, alpha=0.6)
    axr.legend(loc="upper left")
    fig2.tight_layout()
    fig2.savefig(out / "Etats_reservoirs.png", dpi=200)
    plt.close(fig2)

    # Cumuls en mm
    factor = dt * 1000.0
    ET_mm_step    = res["sa_loss"] * 1000.0
    Infil_mm_step = res["infil"] * factor
    Seep_mm_step  = res["seep_loss"] * 1000.0
    Rgen_mm_step  = res["r_gen"] * factor

    P_cum     = np.cumsum(P_mm)
    ET_cum    = np.cumsum(ET_mm_step)
    Infil_cum = np.cumsum(Infil_mm_step)
    Seep_cum  = np.cumsum(Seep_mm_step)
    Rgen_cum  = np.cumsum(Rgen_mm_step)

    fig3, axc = plt.subplots(figsize=(12, 4))
    axc.plot(time_index, P_cum,     label="P cumulée", lw=1.3)
    axc.plot(time_index, ET_cum,    label="ET Ia cumulée", ls=":", lw=1.2)
    axc.plot(time_index, Infil_cum, label="Infiltration cumulée", lw=1.1)
    axc.plot(time_index, Seep_cum,  label="Seepage cumulé", ls="--", lw=1.1)
    axc.plot(time_index, Rgen_cum,  label="Ruissellement généré cumulé", lw=1.3)
    axc.set_xlabel("Date")
    axc.set_ylabel("Lame cumulée (mm)")
    axc.grid(alpha=0.6)
    axc.legend()
    fig3.tight_layout()
    fig3.savefig(out / "Cumuls_mm.png", dpi=200)
    plt.close(fig3)

    print(f"\n[OK] Figures sauvegardées dans : {out}")


if __name__ == "__main__":
    main()
