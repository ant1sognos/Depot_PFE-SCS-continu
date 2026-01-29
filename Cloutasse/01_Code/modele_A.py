
"""
SCS–HSM CONTINU — version sans routage 

Objectif
- Simuler la production pluie–ruissellement à pas de temps constant (dt) avec la formulation SCS–HSM.

Structure des réservoirs
- h_a : réservoir d’abstraction initiale (Ia)
- h_s : réservoir de sol (capacité S) pilotant l’infiltration HSM
- h_r : stock de surface (diagnostic uniquement, non routé vers l’exutoire)

Principe hydraulique (sans propagation)
- Pluie brute p → gestion de Ia → pluie nette q
- Infiltration potentielle HSM contrôlée par (h_s, S) puis limitée par l’eau disponible en surface
- Percolation profonde depuis le sol : seepage ~ exp(-k_seepage·dt)
- Ruissellement généré : r_gen = max(q − infiltration, 0)
- Débit modélisé à l’exutoire (hypothèse instantanée) :
      Q_mod = r_gen * A_BV     (conversion effectuée dans main)

Données d’entrée
- CSV évènement (../02_Data/…) : dateP ; P_mm ; (option) Q_ls
- ETP SAFRAN journalière (../02_Data/ETP_SAFRAN_J.csv) projetée sur la grille temporelle

Calage (optionnel)
- Optimisation multistart + Powell sur :
      theta = [i_a, s, log10(k_infiltr), log10(k_seepage)]
- Critère : combinaison niveau (RMSE), forme (RMSE sur hydrogrammes normalisés) et volume (erreur relative)

Sorties
- Impression du bilan volumique + bilan de masse (fermeture P = ET + seepage + Δstocks)
- Figures sauvegardées dans ../03_Plots/Sans Routage/ :
    1) Q_mod vs Q_obs + pluie inversée
    2) États des réservoirs (h_a, h_s, h_r)
    3) Cumuls P / ETP / infiltration / percolation / ruissellement généré
"""


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import matplotlib as mpl

# Optimisation affichage
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0


# ======================================================================
# 1. MODELE SCS-HSM 
# ======================================================================

def run_scs_hsm(
    dt,
    p_rate,
    etp_rate,
    i_a,
    s,
    k_infiltr,
    k_seepage,
    h_a_init=0.0,
    h_s_init=0.0,
    h_r_init=0.0,
):

    p_rate = np.nan_to_num(p_rate.astype(float))
    etp_rate = np.nan_to_num(etp_rate.astype(float))
    nt = len(p_rate)

    # États
    h_a = np.zeros(nt + 1)
    h_s = np.zeros(nt + 1)
    h_r = np.zeros(nt + 1)

    h_a[0] = h_a_init
    h_s[0] = h_s_init
    h_r[0] = h_r_init

    # Flux
    q = np.zeros(nt)          # pluie nette (m/s)
    infil = np.zeros(nt)      # infiltration (m/s)
    r_gen = np.zeros(nt)      # ruissellement généré (m/s)
    sa_loss = np.zeros(nt)    # ETP effective (m/pas)
    seep_loss = np.zeros(nt)  # percolation (m/pas)

    for n in range(nt):

        p = p_rate[n]     # m/s
        etp = etp_rate[n] # m/s

        # 1) ETP sur Ia
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

        # 3) Infiltration potentielle HSM
        h_s_b = h_s[n]
        Xb = 1 - h_s_b / s
        if Xb <= 0:
            Xb = 1e-12

        Xe = 1.0 / (1.0 / Xb + k_infiltr * dt / s)
        h_s_end = (1 - Xe) * s
        infil_pot = (h_s_end - h_s_b) / dt

        # 4) infiltration limitée par eau dispo en surface
        water_avail = q_n + h_r[n] / dt
        if water_avail < 0:
            water_avail = 0.0

        infil_n = min(max(infil_pot, 0.0), water_avail)
        infil[n] = infil_n

        # 5) Mise à jour du sol + percolation (seepage)
        hs_temp = h_s_b + infil_n * dt
        if k_seepage > 0:
            h_s_after = hs_temp * math.exp(-k_seepage * dt)
            seep_loss[n] = hs_temp - h_s_after
        else:
            h_s_after = hs_temp
        h_s[n + 1] = h_s_after

        # 6) Ruissellement généré (immédiat)
        r_gen_n = max(q_n - infil_n, 0.0)
        r_gen[n] = r_gen_n

        # 7) Stock temporaire dans h_r (pas routé)
        h_r[n + 1] = h_r[n] + (q_n - infil_n) * dt
        if h_r[n + 1] < 0:
            h_r[n + 1] = 0.0

    # =========================================================
    # BILAN DE MASSE — 
    # ---------------------------------------------------------
    # P_tot = ET_tot + Seep_tot + Δ(h_a + h_s + h_r)
    # r_gen alimente h_r -> flux interne
    # =========================================================
    P_tot = np.sum(p_rate) * dt               # [m]
    ET_tot = np.sum(sa_loss)                  # [m]
    Seep_tot = np.sum(seep_loss)              # [m]

    # ruissellement généré (diagnostic)
    R_tot = np.sum(r_gen) * dt                # [m]

    delta_storage = (
        (h_a[-1] - h_a[0]) +
        (h_s[-1] - h_s[0]) +
        (h_r[-1] - h_r[0])
    )

    closure = P_tot - (ET_tot + Seep_tot + delta_storage)

    mass_balance = dict(
        P_tot_m=P_tot,
        R_tot_m=R_tot,              # lame de ruissellement générée (vers h_r)
        ET_tot_m=ET_tot,
        Seep_tot_m=Seep_tot,
        Delta_storage_m=delta_storage,
        Closure_error_m=closure,
        Closure_error_mm=closure * 1000.0,
        Relative_error_pct=100.0 * closure / P_tot if P_tot > 0 else np.nan,
    )


    return dict(
        h_a=h_a,
        h_s=h_s,
        h_r=h_r,
        q=q,
        infil=infil,
        r_gen=r_gen,
        sa_loss=sa_loss,
        seep_loss=seep_loss,
        mass_balance=mass_balance,
    )


# ======================================================================
# 2. LECTURE P + Q
# ======================================================================

def read_rain_series(csv_name, dt):
    base = Path(__file__).resolve().parent
    df = pd.read_csv(base.parent / "02_Data" / csv_name, sep=";")

    time = pd.to_datetime(df["dateP"])
    P_mm = df["P_mm"].astype(float).fillna(0).to_numpy()
    p_rate = P_mm * 1e-3 / dt
    
    q_obs = None
    if "Q_ls" in df.columns:
        q_obs = (
            pd.to_numeric(df["Q_ls"], errors="coerce")  # tout ce qui est non-numérique → NaN
              .fillna(0.0)                              # les NaN -> 0
              .to_numpy() / 1000.0                      # l/s -> m3/s
        )


    return time, P_mm, p_rate, q_obs


# ======================================================================
# 3. LECTURE ETP SAFRAN
# ======================================================================

def read_etp(etp_csv, time_index):
    base = Path(__file__).resolve().parent
    df = pd.read_csv(base.parent / "02_Data" / etp_csv, sep=";")

    df["DATE"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d")
    df["DATE"] = df["DATE"].dt.normalize()
    df["ETP"] = df["ETP"].astype(float).fillna(0.0)

    etp_dict = dict(zip(df["DATE"], df["ETP"]))

    time_index = pd.DatetimeIndex(time_index)   
    dates = time_index.normalize()
    mm_day = np.array([etp_dict.get(d, 0.0) for d in dates])

    etp_rate = mm_day * 1e-3 / (24*3600)
    return etp_rate


# ======================================================================
# 4. RMSE + OBJECTIF + CALIBRATION MULTISTART+POWELL
# ======================================================================

def compute_rmse(qo, qm):
    mask = np.isfinite(qo) & np.isfinite(qm)
    if mask.sum() == 0:
        return 1e6
    return np.sqrt(np.mean((qm[mask] - qo[mask]) ** 2))

def compute_composite_loss(
    q_obs,
    q_mod,
    p_rate,
    dt,
    A_BV,
    qref_percentile=85.0,
    qevent_percentile=85.0,
    alpha_events=30.0,   
    beta_runoff=20.0,    
    gamma_smooth=10.0,   
):
    q_obs = np.asarray(q_obs)
    q_mod = np.asarray(q_mod)

    mask = np.isfinite(q_obs) & np.isfinite(q_mod)
    q_obs = q_obs[mask]
    q_mod = q_mod[mask]

    # 1) RMSE global
    Q_ref = np.percentile(q_obs, qref_percentile)
    rmse_all = np.sqrt(np.mean((q_mod - q_obs)**2)) / Q_ref

    # 2) RMSE crues
    q_event = np.percentile(q_obs, qevent_percentile)
    event_mask = q_obs > q_event
    if event_mask.sum() > 0:
        rmse_evt = np.sqrt(np.mean((q_mod[event_mask] - q_obs[event_mask])**2)) / Q_ref
    else:
        rmse_evt = 0

    # 3) Volume runoff coefficient
    V_obs = np.sum(q_obs)*dt
    V_mod = np.sum(q_mod)*dt
    P_tot = np.sum(p_rate)*dt

    if V_obs>0 and P_tot>0:
        C_obs = V_obs/(P_tot*A_BV)
        C_mod = V_mod/(P_tot*A_BV)
        pen_runoff = (C_mod/C_obs - 1)**2
    else:
        pen_runoff = 1e3

    dq = np.diff(q_mod)
    # Option : ou diff seconde
    # dq2 = np.diff(q_mod,2)
    pen_smooth = np.mean(dq**2)

    # 5) combinaison
    J = (
        rmse_all
        + alpha_events * rmse_evt
        + beta_runoff * pen_runoff
        + gamma_smooth * pen_smooth     # nouveau
    )

    return float(J)


def objective(theta, data):
    """
    Fonction objectif COMPOSITE (même logique que le modèle C, mais sans routage ni q_sub).

    theta = [i_a, s, log10(k_infiltr), log10(k_seepage)]
    """
    i_a, s, logki, logks = theta

    # Gardes-fous physiques simples
    if i_a < 0.0 or i_a > 0.3 or s <= 0.0 or s > 1.0:
        return 1e6

    k_infiltr = 10.0 ** logki
    k_seepage = 10.0 ** logks

    # Simulation du modèle SCS-HSM (sans routage)
    res = run_scs_hsm(
        dt=data["dt"],
        p_rate=data["p_rate"],
        etp_rate=data["etp_rate"],
        i_a=i_a,
        s=s,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
    )

    # Ruissellement généré (m/s) -> Q_mod (m3/s)
    r_gen = res["r_gen"]
    q_mod = r_gen * data["A"]          # A = aire du BV en m²

    q_obs = np.asarray(data["q_obs"], dtype=float)   # m3/s
    dt     = data["dt"]

    # Masque des valeurs valides
    mask = np.isfinite(q_obs) & np.isfinite(q_mod)
    if mask.sum() < 5:
        return 1e6

    q_obs_m = q_obs[mask]
    q_mod_m = q_mod[mask]

    # --- 1) Terme "niveau" : RMSE normalisé par le pic observé ---
    q_peak_obs = np.max(q_obs_m)
    if q_peak_obs <= 0.0:
        return 1e6

    rmse_level = compute_rmse(q_obs_m, q_mod_m) / q_peak_obs  # adimensionnel

    # --- 2) Terme "forme" : RMSE sur hydrogrammes normalisés ---
    eps = 1e-9
    q_obs_norm = q_obs_m / (q_peak_obs + eps)

    q_peak_mod = np.max(q_mod_m)
    if q_peak_mod <= 0.0:
        return 1e6

    q_mod_norm = q_mod_m / (q_peak_mod + eps)

    rmse_shape = compute_rmse(q_obs_norm, q_mod_norm)  # déjà adimensionnel

    # --- 3) Terme "volume" : erreur relative sur le volume ---
    V_obs = float(np.sum(q_obs_m) * dt)
    V_mod = float(np.sum(q_mod_m) * dt)
    if V_obs <= 0.0:
        return 1e6

    rel_vol_err = abs(V_mod - V_obs) / V_obs  # adimensionnel

    # --- Combinaison (mêmes poids que dans C) ---
    w1, w2, w3 = 1.0, 2.0, 1.0
    J = w1 * rmse_level + w2 * rmse_shape + w3 * rel_vol_err

    if not np.isfinite(J):
        return 1e6

    return float(J)

def calibrate_multistart(data, bounds, nstart):

    best = None
    bestJ = np.inf

    for k in range(nstart):
        th0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(
            objective, th0, args=(data,),
            method="Powell", bounds=bounds,
            options={"maxiter": 200, "disp": False},
        )
        print(f"Essai {k+1}/{nstart} : J = {res.fun:.5f}")

        if res.fun < bestJ:
            best = res.x.copy()
            bestJ = res.fun

    return best, bestJ


# ======================================================================
# 5. AFFICHAGE BILAN MASSE
# ======================================================================

def print_mass_balance(mb):
    print("\n=== BILAN DE MASSE ===")
    print(f"P_tot          = {mb['P_tot_m']*1000:.1f} mm")
    print(f"Ruissellement généré (→ h_r) = {mb['R_tot_m']*1000:.1f} mm")
    print(f"Seepage profond= {mb['Seep_tot_m']*1000:.1f} mm")
    print(f"ETP effective  = {mb['ET_tot_m']*1000:.1f} mm")
    print(f"ΔStock (Ia+sol+surf) = {mb['Delta_storage_m']*1000:.3f} mm")
    print(f"Erreur fermeture = {mb['Closure_error_mm']:.3f} mm "
          f"({mb['Relative_error_pct']:.3f} %)")

# ======================================================================
# 6. MAIN
# ======================================================================

def main():

    nstart = 5
    
    dt = 300.0
    A_BV = 800000.0

    csv_event = "all_events/2020/event_2020_011.csv"
    csv_etp   = "ETP_SAFRAN_J.csv"

    # Données
    time, P_mm, p_rate, q_obs = read_rain_series(csv_event, dt)
    etp_rate = read_etp(csv_etp, time)

    if q_obs is None:
        raise RuntimeError("Pas de Q_obs dans le fichier événement.")

    # Calibration ?
    DO_CALIB = True

    if DO_CALIB:
        bounds = [
            (0.0, 0.003),   # i_a
            (0.001, 0.2),  # s
            (-8, -6),       # log10(k_infiltr)
            (-7, -5),       # log10(k_seepage)
        ]
        data = dict(
            dt=dt, p_rate=p_rate, etp_rate=etp_rate,
            q_obs=q_obs, A=A_BV
        )

        print("\n=== Lancement calibration ===")
        theta_opt, Jopt = calibrate_multistart(data, bounds, nstart)

        i_a, s, logki, logks = theta_opt
        k_infiltr = 10 ** logki
        k_seepage = 10 ** logks

        print("\n--- Paramètres optimaux ---")
        print(f"i_a      = {i_a:.6f}")
        print(f"s        = {s:.6f}")
        print(f"k_infiltr= {k_infiltr:.3e}")
        print(f"k_seepage= {k_seepage:.3e}")
        print(f"J_opt    = {Jopt:.5f}")

    else:
        i_a = 0.000007
        s = 0.022467
        k_infiltr = 1.179e-05
        k_seepage = 5.809e-05

    # Simulation
    res = run_scs_hsm(
        dt, p_rate, etp_rate,
        i_a, s, k_infiltr, k_seepage,
    )  
    # États pour les tracés (on enlève le dernier point pour coller à la longueur de "time")
    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]


    r_gen = res["r_gen"]
    q_mod = r_gen * A_BV

    # Bilans en mm
    factor = dt * 1000.0
    P_mm_step     = P_mm
    ET_mm_step    = res["sa_loss"] * 1000.0
    Infil_mm_step = res["infil"] * factor
    Seep_mm_step  = res["seep_loss"] * 1000.0
    R_mm_step     = r_gen * factor

    P_cum_mm     = np.cumsum(P_mm_step)
    ET_cum_mm    = np.cumsum(ET_mm_step)
    Infil_cum_mm = np.cumsum(Infil_mm_step)
    Seep_cum_mm  = np.cumsum(Seep_mm_step)
    R_cum_mm     = np.cumsum(R_mm_step)

    # Volumes
    V_obs = float(np.sum(q_obs) * dt)
    V_mod = float(np.sum(q_mod) * dt)

    print("\n===== BILAN VOLUMES =====")
    print(f"V_obs = {V_obs:.1f} m3")
    print(f"V_mod = {V_mod:.1f} m3")
    if V_obs > 0:
        print(f"Ratio V_mod/V_obs = {V_mod/V_obs:.3f}")

    print_mass_balance(res["mass_balance"])

    # ==================================================================
    # FIGURES — SAUVEGARDE DANS : 03_Plots/Sans-Routage
    # ==================================================================
    base = Path(__file__).resolve().parent
    out = base.parent / "03_Plots" / "Sans Routage" 
    out.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # FIGURE 1 : Q_mod vs Q_obs + pluie inversée en haut
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(time, q_obs, label="Q_obs (m3/s)", lw=1.0, alpha=0.7)
    ax.plot(time, q_mod, label="Q_mod (m3/s)", lw=1.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Débit (m3/s)")
    ax.grid(alpha=0.6)

    ax2 = ax.twinx()
    dt_days = dt / 86400.0

    ax2.bar(
        time, P_mm_step,
        width=dt_days * 0.9, alpha=0.3, color="blue",
        label="Pluie (mm/5min)"
    )
    ax2.invert_yaxis()
    ax2.set_ylabel("Pluie (mm/5min)")

    # légende combinée
    h1,l1 = ax.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out/"Qmod_Qobs_Pluie_SansRoutage.png", dpi=200)
    plt.close(fig)   
    
    # ============================================================
    # FIGURE 2 :  États des réservoirs
    # ============================================================
    fig2, axr = plt.subplots(figsize=(12, 4))

    axr.plot(time, h_a, label="h_a (Ia)",      color="grey",  linewidth=1.0)
    axr.plot(time, h_s, label="h_s (sol)",     color="green", linewidth=1.0)
    axr.plot(time, h_r, label="h_r (surface)", color="red",   linewidth=1.0)

    axr.set_xlabel("Date")
    axr.set_ylabel("Hauteur d'eau (m)")
    axr.grid(True, linewidth=0.4, alpha=0.6)
    axr.legend(loc="upper left")
    fig2.suptitle("États des réservoirs (Ia, sol, surface)")
    fig2.tight_layout()
    fig2.savefig(out / "Etats_reservoirs_SansRoutage.png", dpi=200)
    plt.close(fig2)

    # ============================================================
    # FIGURE 3 : Cumuls
    # ============================================================
    fig3, axc = plt.subplots(figsize=(12, 4))

    axc.plot(time, P_cum_mm,      label="P cumulée",               lw=1.3)
    axc.plot(time, ET_cum_mm,     label="ETP cumulée",             ls=":",  lw=1.2)
    axc.plot(time, Infil_cum_mm,  label="Infiltration cumulée",    lw=1.1)
    axc.plot(time, Seep_cum_mm,   label="Percolation cumulée",     ls="--", lw=1.1)
    axc.plot(time, R_cum_mm,      label="Ruissellement cumulé",    lw=1.3)

    axc.set_xlabel("Date")
    axc.set_ylabel("Lame cumulée (mm)")
    axc.grid(alpha=0.6)
    axc.legend()

    fig3.tight_layout()
    fig3.savefig(out / "Cumuls_SansRoutage.png", dpi=200)
    plt.close(fig3)


if __name__ == "__main__":
    main()
