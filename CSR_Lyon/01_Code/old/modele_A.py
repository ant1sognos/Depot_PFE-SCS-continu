# -*- coding: utf-8 -*-
"""
SCS CLASSIQUE (Ia, S) – version "import/export CSR" (évènementiel)

- Lit un fichier event CSV : date ; P_mm ; Q_inf_LH ; Q_ruiss_LH
- Convertit P_mm (mm/pas) -> p_rate (m/s)
- Simule SCS classique :
    * réservoir d'accumulation initiale ha (se remplit jusqu'à Ia)
    * pluie nette q dès que ha atteint Ia
    * partition q -> infiltration vers sol vs ruissellement via loi SCS
- Compare Q_ruiss_mod (L/h) à Q_ruiss_obs (L/h)
- Sauve :
    * figures (Qruiss + pluie, états, cumuls)
    * CSV de sortie avec toutes les séries utiles

⚠️ SCS classique = pas de routage => pas de décrue réaliste sans couplage.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["agg.path.chunksize"] = 10000


# =========================
# Conversions
# =========================
def lh_to_m3s(q_lh):  # L/h -> m3/s
    return np.asarray(q_lh, dtype=float) / 1000.0 / 3600.0

def m3s_to_lh(q_m3s):  # m3/s -> L/h
    return np.asarray(q_m3s, dtype=float) * 1000.0 * 3600.0

def mm_per_step_to_mps(mm_per_step, dt_s):  # mm/pas -> m/s
    return np.asarray(mm_per_step, dtype=float) * 1e-3 / float(dt_s)


# =========================
# Reader CSV évènement
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]

def read_parking_event_csv(csv_event_rel, sep=";"):
    csv_path = BASE_DIR / "02_Data" / Path(csv_event_rel)
    if not csv_path.exists():
        raise FileNotFoundError(f"Event CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path, sep=sep)

    required = ["date", "P_mm", "Q_inf_LH", "Q_ruiss_LH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes {missing} dans {csv_path.name}. Colonnes présentes={list(df.columns)}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if df["date"].isna().any():
        bad = df.loc[df["date"].isna(), "date"].head(5).tolist()
        raise ValueError(f"Dates illisibles dans {csv_path.name} (exemples={bad})")

    df = df.sort_values("date").reset_index(drop=True)
    time_index = pd.DatetimeIndex(df["date"])

    diffs = time_index.to_series().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        raise ValueError(f"Série trop courte pour inférer dt: {csv_path.name}")
    dt_sec = float(np.median(diffs))

    P_mm = pd.to_numeric(df["P_mm"], errors="coerce").fillna(0.0).to_numpy(float)
    Q_inf_LH = pd.to_numeric(df["Q_inf_LH"], errors="coerce").fillna(0.0).to_numpy(float)
    Q_ruiss_LH = pd.to_numeric(df["Q_ruiss_LH"], errors="coerce").fillna(0.0).to_numpy(float)

    P_mm = np.clip(P_mm, 0.0, None)
    Q_inf_LH = np.clip(Q_inf_LH, 0.0, None)
    Q_ruiss_LH = np.clip(Q_ruiss_LH, 0.0, None)

    return time_index, P_mm, Q_inf_LH, Q_ruiss_LH, dt_sec


# =========================
# SCS classique (forme "original")
# =========================
def run_scs_classic(
    dt: float,
    p_rate: np.ndarray,
    i_a: float = 2e-3,   # m
    s: float = 0.02,     # m
    h_a_init: float = 0.0,
    h_s_init: float = 0.0,
    h_r_init: float = 0.0,
) -> dict:
    """
    Implémentation fidèle à ton SCS_original :
    - ha se remplit, déborde à Ia => q (pluie nette) = (ha - Ia)/dt, ha=Ia
    - partition q entre infiltration vers sol et ruissellement via :
        X_begin = 1 - hs/S
        X_end   = 1/(1/X_begin + q*dt/S)
        hs_next = (1 - X_end)*S
        infil   = (hs_next - hs)/dt
        hr_next = hr + (q - infil)*dt
    """
    p_rate = np.nan_to_num(np.asarray(p_rate, dtype=float), nan=0.0)
    nt = len(p_rate)

    t = np.array([i * dt for i in range(nt + 1)], dtype=float)

    # États
    h_a = np.zeros(nt + 1, dtype=float)
    h_s = np.zeros(nt + 1, dtype=float)
    h_r = np.zeros(nt + 1, dtype=float)
    h_a[0], h_s[0], h_r[0] = float(h_a_init), float(h_s_init), float(h_r_init)

    # Flux
    q_net = np.zeros(nt, dtype=float)     # pluie nette (m/s)
    infil = np.zeros(nt, dtype=float)     # infiltration vers sol (m/s)
    r_rate = np.zeros(nt, dtype=float)    # ruissellement (m/s) = dhr/dt
    p_store = np.zeros(nt, dtype=float)

    for n in range(nt):
        p = float(p_rate[n])
        p_store[n] = p

        # 1) Accumulation initiale (Ia)
        h_a_temp = h_a[n] + p * dt
        if h_a_temp < i_a:
            q = 0.0
            h_a[n + 1] = h_a_temp
        else:
            q = (h_a_temp - i_a) / dt
            h_a[n + 1] = i_a
        q_net[n] = q

        # 2) Réservoir sol SCS (mise à jour fermée)
        hs0 = h_s[n]
        if s <= 0:
            raise ValueError("S doit être > 0.")
        X_begin = 1.0 - hs0 / s
        # robustesse numérique
        X_begin = max(X_begin, 1e-12)

        X_end = 1.0 / (1.0 / X_begin + (q * dt) / s)
        hs1 = (1.0 - X_end) * s
        h_s[n + 1] = hs1

        infil_n = (hs1 - hs0) / dt
        infil[n] = max(infil_n, 0.0)

        # 3) Ruissellement (cumul hr)
        r_n = max(q - infil[n], 0.0)
        r_rate[n] = r_n
        h_r[n + 1] = h_r[n] + r_n * dt

    # Bilans simples (pas d'ET, pas de seepage)
    P_tot = float(np.nansum(p_rate) * dt)
    R_tot = float(np.nansum(r_rate) * dt)
    I_tot = float(np.nansum(infil) * dt)
    delta_storage = (h_a[-1] - h_a[0]) + (h_s[-1] - h_s[0]) + (h_r[-1] - h_r[0])
    closure = P_tot - (R_tot + I_tot + delta_storage)

    mass_balance = {
        "P_tot_m": P_tot,
        "R_tot_m": R_tot,
        "I_tot_m": I_tot,
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
        "q_net": q_net,
        "infil": infil,
        "r_rate": r_rate,
        "mass_balance": mass_balance,
    }


def print_mass_balance_scs(mb: dict):
    print("\n=== Bilan de masse (SCS classique) ===")
    print(f"P_tot              = {mb['P_tot_m']*1000:.2f} mm")
    print(f"Infiltration (sol) = {mb['I_tot_m']*1000:.2f} mm")
    print(f"Ruissellement      = {mb['R_tot_m']*1000:.2f} mm")
    print(f"ΔStock (Ia+sol+run) = {mb['Delta_storage_m']*1000:.2f} mm")
    print(f"Erreur fermeture   = {mb['Closure_error_mm']:.6f} mm ({mb['Relative_error_%']:.6f} %)")


# =========================
# MAIN (même style CSR)
# =========================
def main():
    base_dir = Path(__file__).resolve().parent

    # ----- Choix évènement
    csv_event_rel = "all_events1/2024/event_2024_001.csv"
    event_name = Path(csv_event_rel).stem

    # ----- Paramètres bassin
    A_BV_M2 = 94.0  # m²

    # ----- Paramètres SCS
    I_A = 0.0002     # m  (met 0 si tu veux que tout parte direct en pluie nette)
    S   = 0.13      # m  (capacité du sol SCS)

    print("=== PARAMÈTRES SCS CLASSIQUE ===")
    print(f"A_BV_M2 = {A_BV_M2:.1f} m²")
    print(f"Ia      = {I_A:.6f} m")
    print(f"S       = {S:.6f} m")
    print("================================\n")

    # ----- Lecture données
    time_index, P_mm_event, qinf_obs_lh, qruiss_obs_lh, dt = read_parking_event_csv(csv_event_rel)
    p_rate_input = mm_per_step_to_mps(P_mm_event, dt)

    # ----- Simulation SCS classique
    res = run_scs_classic(
        dt=dt,
        p_rate=p_rate_input,
        i_a=I_A,
        s=S,
        h_a_init=0.0,
        h_s_init=0.0,
        h_r_init=0.0,
    )

    print_mass_balance_scs(res["mass_balance"])

    # ----- Débits modélisés (ruissellement) en L/h
    qruiss_obs_m3s = lh_to_m3s(qruiss_obs_lh)
    qruiss_mod_m3s = res["r_rate"] * A_BV_M2
    qruiss_mod_lh  = m3s_to_lh(qruiss_mod_m3s)

    # Infiltration modélisée (vers sol) en "débit" L/h (si tu veux le tracer)
    qinf_mod_m3s = res["infil"] * A_BV_M2
    qinf_mod_lh  = m3s_to_lh(qinf_mod_m3s)

    # ----- Cumuls (mm)
    factor_mm = dt * 1000.0
    P_mm = p_rate_input * factor_mm
    infil_mm  = res["infil"] * factor_mm
    runoff_mm = res["r_rate"] * factor_mm

    P_cum      = np.cumsum(P_mm)
    infil_cum  = np.cumsum(infil_mm)
    runoff_cum = np.cumsum(runoff_mm)

    # ----- États (alignés sur time_index)
    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]

    # ----- Dossier plots
    plots_dir = base_dir.parent / "03_Plots" / "Parking_CSR" / "SCS_CLASSIC" / event_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    dt_days = dt / 86400.0
    maxP = float(np.nanmax(P_mm)) if np.nanmax(P_mm) > 0 else 1.0

    # FIG 1: Q_ruiss + pluie
    fig, axQ = plt.subplots(figsize=(10, 4))
    axQ.plot(time_index, qruiss_obs_lh, label="Q_ruiss_obs (L/h)", linewidth=1.0, alpha=0.7)
    axQ.plot(time_index, qruiss_mod_lh, label="Q_ruiss_mod SCS (L/h)", linewidth=1.4)
    axQ.set_xlabel("Date")
    axQ.set_ylabel("Débit ruisselé (L/h)")
    axQ.grid(True, linewidth=0.4, alpha=0.6)

    axP = axQ.twinx()
    axP.bar(time_index, P_mm, width=dt_days * 0.8, align="center", alpha=0.35, label="P (mm/pas)")
    axP.set_ylabel("Pluie (mm/pas)")
    axP.invert_yaxis()
    axP.set_ylim(maxP * 1.05, 0.0)

    lines1, labels1 = axQ.get_legend_handles_labels()
    lines2, labels2 = axP.get_legend_handles_labels()
    axQ.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle("SCS classique: Q_ruiss_obs vs Q_ruiss_mod + Pluie")
    fig.tight_layout()
    fig.savefig(plots_dir / "Qruiss_obs_vs_mod_LH_P.png", dpi=200)
    plt.close(fig)

    # FIG 2: états
    fig2, axr = plt.subplots(figsize=(10, 4))
    axr.plot(time_index, h_a, label="h_a (Ia)", linewidth=1.0)
    axr.plot(time_index, h_s, label="h_s (sol)", linewidth=1.0)
    axr.plot(time_index, h_r, label="h_r (runoff cumul)", linewidth=1.0)
    axr.set_xlabel("Date")
    axr.set_ylabel("Hauteur / lame (m)")
    axr.grid(True, linewidth=0.4, alpha=0.6)
    axr.legend(loc="upper left")
    fig2.suptitle("SCS classique: états (Ia, sol, runoff cumulé)")
    fig2.tight_layout()
    fig2.savefig(plots_dir / "etats_reservoirs.png", dpi=200)
    plt.close(fig2)

    # FIG 3: cumuls (mm)
    fig3, axc = plt.subplots(figsize=(10, 4))
    axc.plot(time_index, P_cum,      label="P cumulée", linewidth=1.3)
    axc.plot(time_index, infil_cum,  label="Infiltration cumulée (sol)", linewidth=1.1)
    axc.plot(time_index, runoff_cum, label="Ruissellement cumulé", linewidth=1.3)
    axc.set_xlabel("Date")
    axc.set_ylabel("Lame cumulée (mm)")
    axc.grid(True, linewidth=0.4, alpha=0.6)
    axc.legend(loc="upper left")
    fig3.suptitle("SCS classique: cumuls P / infil / runoff")
    fig3.tight_layout()
    fig3.savefig(plots_dir / "cumuls_mm.png", dpi=200)
    plt.close(fig3)

    # FIG 4: (option) Q_inf observé vs infiltration modélisée
    fig4, axI = plt.subplots(figsize=(10, 4))
    axI.plot(time_index, qinf_obs_lh, label="Q_inf_obs (L/h)", linewidth=1.0, alpha=0.7)
    axI.plot(time_index, qinf_mod_lh, label="Infiltration SCS (L/h équiv.)", linewidth=1.4)
    axI.set_xlabel("Date")
    axI.set_ylabel("Débit (L/h)")
    axI.grid(True, linewidth=0.4, alpha=0.6)
    axI.legend(loc="upper right")
    fig4.suptitle("Comparaison indicative: Q_inf_obs vs infiltration SCS")
    fig4.tight_layout()
    fig4.savefig(plots_dir / "Qinf_obs_vs_infilSCS_LH.png", dpi=200)
    plt.close(fig4)

    # ----- Export CSV des séries simulées
    out_df = pd.DataFrame({
        "date": time_index,
        "P_mm": P_mm,
        "Q_ruiss_obs_LH": np.asarray(qruiss_obs_lh, float),
        "Q_ruiss_mod_LH": np.asarray(qruiss_mod_lh, float),
        "Q_inf_obs_LH": np.asarray(qinf_obs_lh, float),
        "Q_infilSCS_eq_LH": np.asarray(qinf_mod_lh, float),
        "h_a_m": h_a,
        "h_s_m": h_s,
        "h_r_m": h_r,
        "q_net_mps": res["q_net"],
        "infil_mps": res["infil"],
        "r_rate_mps": res["r_rate"],
    })
    out_csv = plots_dir / "scs_classic_timeseries.csv"
    out_df.to_csv(out_csv, index=False, sep=";")

    print(f"\n[OK] Figures sauvegardées dans : {plots_dir}")
    print(f"[OK] CSV sortie : {out_csv}")


if __name__ == "__main__":
    main()
