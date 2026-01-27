# -*- coding: utf-8 -*-
"""
--------------------------------------------
ANALYSE : lit la simulation du RUN (simulation.csv) + all_events1,
puis calcule des métriques évènementielles et produit des diagnostics.

Compatible RUN warmup (period = warmup/calib/valid) + période hydro 2022-10 -> 2023-09.

Robustesse :
- Ne filtre pas sur une année fixe
- Ne garde que les évènements dont [t_start, t_end] intersecte la période simulée
- Si aucun évènement valide -> export Excel vide + arrêt propre (pas de KeyError)

Dépendances : pandas, numpy, matplotlib, openpyxl
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
def get_config():
    """
    Centralise la config pour éviter les variables globales éparpillées.
    Modifie ici et seulement ici.
    """
    try:
        root = Path(__file__).resolve().parents[1]  # CSR_Lyon
    except NameError:
        root = Path.cwd()

    out_base = root / "03_Plots"
    split_name = "calib_H1_valid_H2"
    split_dir = out_base / split_name

    cfg = dict(
        ROOT=root,
        OUT_BASE=out_base,
        SPLIT_NAME=split_name,
        SPLIT_DIR=split_dir,
        SIM_CSV=split_dir / "simulation.csv",
        META_EVENTS=root / "02_Data" / "all_events1" / "hr_hp_events.csv",
        USE_ALL_EVENTS1=True,
        PREF_QINF_MOD_COLS=["seep_mod_LH", "infil_mod_LH", "qnet_mod_LH"],

        # Tracé évènements
        MAKE_EVENT_PNG=True,
        PLOT_MODE="topN",         # "all" | "topN" | "filter"
        TOP_N=25,
        TOP_METRIC="P_tot_mm",    # "P_tot_mm" | "Qmax_obs_LH"
        FILTER_EXPR="NSE_Qruiss < 0",

        # Filtres / affichage
        QMOD_EPS_LH=1e-6,
        RAIN_FRACTION=0.25,
        RAIN_ALPHA=0.30,
    )
    return cfg


# =========================================================
# Conversions / utilitaires
# =========================================================
def lh_to_m3s(q_lh):
    return np.asarray(q_lh, float) / 1000.0 / 3600.0


def safe_nanmax(x):
    x = np.asarray(x, float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def infer_dt_seconds(time_index: pd.DatetimeIndex) -> float:
    dt_s = float(pd.Series(time_index).diff().dropna().dt.total_seconds().median())
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise RuntimeError("Impossible d'inférer dt à partir de la colonne date.")
    return dt_s


# =========================================================
# Metrics (évènement)
# =========================================================
def rmse(obs, mod):
    obs = np.asarray(obs, float)
    mod = np.asarray(mod, float)
    m = np.isfinite(obs) & np.isfinite(mod)
    if m.sum() < 2:
        return np.nan
    d = mod[m] - obs[m]
    return float(np.sqrt(np.mean(d * d)))


def bias_mean(obs, mod):
    obs = np.asarray(obs, float)
    mod = np.asarray(mod, float)
    m = np.isfinite(obs) & np.isfinite(mod)
    if m.sum() < 2:
        return np.nan
    return float(np.mean(mod[m] - obs[m]))


def nse(obs, mod):
    obs = np.asarray(obs, float)
    mod = np.asarray(mod, float)
    m = np.isfinite(obs) & np.isfinite(mod)
    if m.sum() < 3:
        return np.nan
    o = obs[m]
    sse = np.sum((mod[m] - o) ** 2)
    sst = np.sum((o - np.mean(o)) ** 2)
    if sst <= 1e-30:
        return np.nan
    return float(1.0 - sse / sst)


def kge(obs, mod):
    obs = np.asarray(obs, float)
    mod = np.asarray(mod, float)
    m = np.isfinite(obs) & np.isfinite(mod)
    if m.sum() < 3:
        return np.nan
    o = obs[m]
    p = mod[m]
    if np.std(o) <= 1e-30 or np.std(p) <= 1e-30:
        return np.nan
    r = float(np.corrcoef(o, p)[0, 1])
    alpha = float(np.std(p) / np.std(o))
    beta = float(np.mean(p) / np.mean(o)) if abs(np.mean(o)) > 1e-30 else np.nan
    if not np.isfinite(beta):
        return np.nan
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def peak_timing_error_minutes(time_index, obs, mod):
    obs = np.asarray(obs, float)
    mod = np.asarray(mod, float)
    if len(obs) == 0 or len(mod) == 0:
        return np.nan
    if not (np.isfinite(obs).any() and np.isfinite(mod).any()):
        return np.nan
    i_obs = int(np.nanargmax(obs))
    i_mod = int(np.nanargmax(mod))
    t_obs = pd.Timestamp(time_index[i_obs])
    t_mod = pd.Timestamp(time_index[i_mod])
    return float((t_mod - t_obs).total_seconds() / 60.0)


# =========================================================
# Overlay pluie sur axe débit
# =========================================================
def overlay_rain_on_discharge_axis(axQ, tt, Pmm, dt_s,
                                  fraction=0.25, alpha=0.30,
                                  add_right_axis=True):
    P_clip = np.clip(np.asarray(Pmm, float), 0.0, None)
    if not np.isfinite(P_clip).any():
        return
    pmax = float(np.nanmax(P_clip))
    if pmax <= 0:
        return

    y0, y1 = axQ.get_ylim()
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
        return

    rain_h_max = (y1 - y0) * float(fraction)
    rain_h = rain_h_max * (P_clip / pmax)
    rain_bottom = y1 - rain_h
    bar_width_days = float(dt_s) / 86400.0

    axQ.bar(tt, rain_h, bottom=rain_bottom, width=bar_width_days,
            align="center", alpha=float(alpha), label="P (mm/pas)")

    if add_right_axis:
        axP = axQ.twinx()
        axP.set_ylim(axQ.get_ylim())
        mm_ticks = np.array([0.0, 0.5 * pmax, pmax], float)
        y_ticks = y1 - rain_h_max * (mm_ticks / pmax)
        axP.set_yticks(y_ticks)
        axP.set_yticklabels([f"{v:.2f}" for v in mm_ticks])
        axP.set_ylabel("Pluie (mm/pas)")
        axP.grid(False)


# =========================================================
# Lecture simulation
# =========================================================
def load_simulation(sim_csv: Path, pref_qinf_cols):
    if not sim_csv.exists():
        raise FileNotFoundError(f"Simulation introuvable: {sim_csv}")

    sim = pd.read_csv(sim_csv, sep=";")
    sim["date"] = pd.to_datetime(sim["date"], errors="coerce")
    sim = sim.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    t = pd.DatetimeIndex(sim["date"])

    required = ["P_mm", "Q_ruiss_obs_LH", "Q_ruiss_mod_LH", "Q_inf_obs_LH"]
    missing = [c for c in required if c not in sim.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {sim_csv.name}: {missing}\nCols={list(sim.columns)}")

    if "period" not in sim.columns:
        sim["period"] = "all"
    sim["period"] = sim["period"].astype(str).str.lower().replace({"nan": "all", "": "all"})

    dt_s = infer_dt_seconds(t)

    qinf_mod_col = None
    for c in pref_qinf_cols:
        if c in sim.columns and sim[c].notna().any():
            qinf_mod_col = c
            break

    t_sim0 = pd.Timestamp(t.min())
    t_sim1 = pd.Timestamp(t.max())

    return sim, t, dt_s, qinf_mod_col, t_sim0, t_sim1


# =========================================================
# Events: parsing + intersection période simulée
# =========================================================
def parse_year_evt(row):
    """
    Essaie d'extraire (year, evt_num) depuis les 2 premières colonnes de hr_hp_events.csv
    Cas supportés :
      - (year, "event_YYYY_NNN")
      - (year, "NNN")
      - ("YYYY", "NNN")
    """
    year_raw = row.iloc[0]
    evt_raw = row.iloc[1]

    try:
        year = int(str(year_raw).strip())
    except Exception:
        year = None

    evt_s = str(evt_raw).strip()

    # event_YYYY_NNN
    m = re.search(r"event_(\d{4})_(\d{1,4})", evt_s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # YYYY ... NNN
    m = re.search(r"(\d{4})\D+(\d{1,4})$", evt_s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # fallback: evt_num seul
    try:
        evt_num = int(evt_s)
    except Exception:
        evt_num = None

    return year, evt_num


def read_parking_event_csv(root: Path, year: int, evt_num: int, sep=";"):
    evt_name = f"event_{year}_{evt_num:03d}"
    csv_path = root / "02_Data" / "all_events1" / str(year) / f"{evt_name}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, sep=sep)
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if len(df) < 2:
        return None

    ti = pd.DatetimeIndex(df["date"])
    return dict(event_name=evt_name, t_start=pd.Timestamp(ti[0]), t_end=pd.Timestamp(ti[-1]))


def build_events_table(root: Path, meta_events: Path, t_sim0: pd.Timestamp, t_sim1: pd.Timestamp,
                       use_all_events1=True):
    if not use_all_events1:
        raise RuntimeError("USE_ALL_EVENTS1=False non géré ici (mets True).")
    if not meta_events.exists():
        raise FileNotFoundError(f"Méta events introuvable: {meta_events}")

    meta = pd.read_csv(meta_events, sep=";")
    if meta.shape[1] < 2:
        raise RuntimeError("hr_hp_events.csv doit avoir au moins 2 colonnes (année, id/num).")

    rows = []
    for _, row in meta.iterrows():
        year, evt_num = parse_year_evt(row)
        if year is None or evt_num is None:
            continue

        ev = read_parking_event_csv(root, year, evt_num)
        if ev is None:
            continue

        # Conserver uniquement ceux qui intersectent la période simulée
        if (ev["t_end"] < t_sim0) or (ev["t_start"] > t_sim1):
            continue

        rows.append(ev)

    evt = pd.DataFrame(rows)
    if len(evt) == 0:
        raise RuntimeError(
            "Aucun évènement ne tombe dans la période simulée.\n"
            "=> Vérifie all_events1 (dates) et la période de simulation."
        )

    evt = evt.sort_values("t_start").reset_index(drop=True)
    return evt


# =========================================================
# Métriques par évènement
# =========================================================
def compute_event_metrics(sim: pd.DataFrame,
                          t: pd.DatetimeIndex,
                          dt_s: float,
                          evt: pd.DataFrame,
                          qinf_mod_col: str | None,
                          qmod_eps_lh: float):
    rows = []

    for _, er in evt.iterrows():
        name = str(er["event_name"])
        t0 = pd.Timestamp(er["t_start"])
        t1 = pd.Timestamp(er["t_end"])

        m = (t >= t0) & (t <= t1)
        if m.sum() < 3:
            continue

        df = sim.loc[m].copy()
        tt = pd.DatetimeIndex(df["date"])

        # Période de l'évènement (warmup/calib/valid/mixed)
        period_vals = df["period"].astype(str).str.lower().replace({"nan": "all", "": "all"})
        uniqp = sorted(set([p for p in period_vals.unique() if p != "all"]))
        event_period = uniqp[0] if len(uniqp) == 1 else ("mixed" if len(uniqp) > 1 else "all")

        Pmm = df["P_mm"].to_numpy(float)
        qobs = df["Q_ruiss_obs_LH"].to_numpy(float)
        qmod = df["Q_ruiss_mod_LH"].to_numpy(float)
        iobs = df["Q_inf_obs_LH"].to_numpy(float)

        imod = None
        if qinf_mod_col is not None and qinf_mod_col in df.columns:
            imod = df[qinf_mod_col].to_numpy(float)

        # Skip évènement si modèle "mort"
        if not np.isfinite(qmod).any():
            continue
        if float(np.nanmax(qmod)) <= float(qmod_eps_lh):
            continue

        Vru_obs = float(np.sum(np.clip(lh_to_m3s(qobs), 0, None)) * dt_s)
        Vru_mod = float(np.sum(np.clip(lh_to_m3s(qmod), 0, None)) * dt_s)

        Vinf_obs = float(np.sum(np.clip(lh_to_m3s(iobs), 0, None)) * dt_s)
        Vinf_mod = np.nan
        if imod is not None and np.isfinite(imod).any():
            Vinf_mod = float(np.sum(np.clip(lh_to_m3s(imod), 0, None)) * dt_s)

        denom_obs = Vru_obs + Vinf_obs
        denom_mod = Vru_mod + (Vinf_mod if np.isfinite(Vinf_mod) else 0.0)
        Cr_obs = (Vru_obs / denom_obs) if denom_obs > 1e-12 else np.nan
        Cr_mod = (Vru_mod / denom_mod) if denom_mod > 1e-12 else np.nan

        rows.append(dict(
            event_name=name,
            period=event_period,
            t_start=str(t0),
            t_end=str(t1),
            n_steps=int(len(df)),
            duration_h=float(len(df) * dt_s / 3600.0),
            P_tot_mm=float(np.sum(np.clip(Pmm, 0, None))),
            Qmax_obs_LH=safe_nanmax(qobs),
            Qmax_mod_LH=safe_nanmax(qmod),
            Vruiss_obs_m3=Vru_obs,
            Vruiss_mod_m3=Vru_mod,
            Vruiss_ratio=(Vru_mod / Vru_obs) if Vru_obs > 1e-12 else np.nan,
            RMSE_Qruiss_LH=rmse(qobs, qmod),
            BIAS_Qruiss_LH=bias_mean(qobs, qmod),
            NSE_Qruiss=nse(qobs, qmod),
            KGE_Qruiss=kge(qobs, qmod),
            peak_timing_err_min=peak_timing_error_minutes(tt, qobs, qmod),
            Vinf_obs_m3=Vinf_obs,
            Vinf_mod_m3=Vinf_mod,
            Cr_obs=Cr_obs,
            Cr_mod=Cr_mod,
            Cr_bias=(Cr_mod - Cr_obs) if (np.isfinite(Cr_obs) and np.isfinite(Cr_mod)) else np.nan,
            qinf_mod_col=(qinf_mod_col if qinf_mod_col else ""),
        ))

    expected_cols = [
        "event_name", "period", "t_start", "t_end", "n_steps", "duration_h", "P_tot_mm",
        "Qmax_obs_LH", "Qmax_mod_LH", "Vruiss_obs_m3", "Vruiss_mod_m3", "Vruiss_ratio",
        "RMSE_Qruiss_LH", "BIAS_Qruiss_LH", "NSE_Qruiss", "KGE_Qruiss", "peak_timing_err_min",
        "Vinf_obs_m3", "Vinf_mod_m3", "Cr_obs", "Cr_mod", "Cr_bias", "qinf_mod_col"
    ]

    df_evt = pd.DataFrame(rows)
    if len(df_evt) == 0:
        df_evt = pd.DataFrame(columns=expected_cols)

    return df_evt


# =========================================================
# Sélection évènements à tracer
# =========================================================
def select_events_to_plot(df_evt: pd.DataFrame, plot_mode: str, top_metric: str,
                          top_n: int, filter_expr: str):
    if df_evt is None or len(df_evt) == 0:
        return df_evt

    if plot_mode == "all":
        return df_evt

    if plot_mode == "topN":
        if top_metric not in df_evt.columns:
            return df_evt
        d = df_evt.copy()
        d = d[np.isfinite(d[top_metric].to_numpy(float))]
        return d.sort_values(top_metric, ascending=False).head(int(top_n))

    if plot_mode == "filter":
        try:
            return df_evt.query(filter_expr).copy()
        except Exception:
            print(f"[WARN] FILTER_EXPR invalide: {filter_expr} -> fallback all")
            return df_evt

    return df_evt


# =========================================================
# Figures: évènement
# =========================================================
def ensure_event_dirs(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["all", "warmup", "calib", "valid", "mixed"]:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)
    return base_dir


def plot_event_figure(sim: pd.DataFrame, t: pd.DatetimeIndex, dt_s: float,
                      rr: pd.Series, qinf_mod_col: str | None,
                      out_events_dir: Path, rain_fraction: float, rain_alpha: float):
    name = rr["event_name"]
    t0 = pd.Timestamp(rr["t_start"])
    t1 = pd.Timestamp(rr["t_end"])
    event_period = rr["period"] if rr["period"] in ["warmup", "calib", "valid", "mixed"] else "all"

    m = (t >= t0) & (t <= t1)
    if m.sum() < 3:
        return

    df = sim.loc[m].copy()
    tt = pd.DatetimeIndex(df["date"])
    Pmm = df["P_mm"].to_numpy(float)

    qobs = df["Q_ruiss_obs_LH"].to_numpy(float)
    qmod = df["Q_ruiss_mod_LH"].to_numpy(float)
    iobs = df["Q_inf_obs_LH"].to_numpy(float)

    imod = None
    if qinf_mod_col is not None and qinf_mod_col in df.columns:
        imod = df[qinf_mod_col].to_numpy(float)

    fig = plt.figure(figsize=(12, 6))
    axQ = fig.add_subplot(2, 1, 1)
    axI = fig.add_subplot(2, 1, 2, sharex=axQ)

    # Qruiss
    axQ.plot(tt, qobs, label="Qruiss obs (L/h)")
    axQ.plot(tt, qmod, label="Qruiss mod (L/h)")
    axQ.set_ylabel("Qruiss (L/h)")
    axQ.grid(True, alpha=0.3)

    ymax_data = float(np.nanmax(np.r_[qobs, qmod])) if np.isfinite(np.r_[qobs, qmod]).any() else 1.0
    axQ.set_ylim(0.0, max(1e-9, ymax_data) * 1.05)

    overlay_rain_on_discharge_axis(
        axQ=axQ, tt=tt, Pmm=Pmm, dt_s=dt_s,
        fraction=rain_fraction, alpha=rain_alpha,
        add_right_axis=True
    )

    axQ.legend(loc="upper right")
    axQ.set_title(f"{name} | {event_period.upper()} | {t0} → {t1}")

    # Qinf
    axI.plot(tt, iobs, label="Qinf obs (L/h)")
    if imod is not None:
        axI.plot(tt, imod, label=f"{qinf_mod_col} (proxy) (L/h)")
    axI.set_ylabel("Qinf (L/h)")
    axI.grid(True, alpha=0.3)
    axI.legend(loc="upper right")

    # Petit cartouche métriques
    qmax_ratio = (rr["Qmax_mod_LH"] / rr["Qmax_obs_LH"]) if (
        np.isfinite(rr["Qmax_obs_LH"]) and rr["Qmax_obs_LH"] > 1e-12
    ) else np.nan

    txt = (
        f"RMSE={rr['RMSE_Qruiss_LH']:.1f} L/h | NSE={rr['NSE_Qruiss']:.2f} | KGE={rr['KGE_Qruiss']:.2f}\n"
        f"Vru ratio={rr['Vruiss_ratio']:.2f} | Qmax ratio={qmax_ratio:.2f}\n"
        f"Δt_pic={rr['peak_timing_err_min']:.0f} min"
    )
    fig.text(0.02, 0.01, txt)

    fig.tight_layout()

    f_all = out_events_dir / "all" / f"{name}.png"
    f_per = out_events_dir / event_period / f"{name}.png"
    fig.savefig(f_all, dpi=160)
    fig.savefig(f_per, dpi=160)
    plt.close(fig)


# =========================================================
# Diagnostics globaux
# =========================================================
def ensure_diag_dirs(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["all", "warmup", "calib", "valid"]:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)
    return base_dir


def scatter_obs_mod(x_obs, x_mod, xlabel, ylabel, title, out_png):
    x_obs = np.asarray(x_obs, float)
    x_mod = np.asarray(x_mod, float)
    m = np.isfinite(x_obs) & np.isfinite(x_mod)
    if m.sum() < 3:
        return
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_obs[m], x_mod[m])
    lo = min(np.nanmin(x_obs[m]), np.nanmin(x_mod[m]))
    hi = max(np.nanmax(x_obs[m]), np.nanmax(x_mod[m]))
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def hist(x, xlabel, title, out_png, bins=30):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def scatter_xy(x, y, xlabel, ylabel, title, out_png):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x[m], y[m])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def make_diag_for(df, tag, out_dir: Path):
    if df is None or len(df) < 3:
        return

    scatter_obs_mod(df["Vruiss_obs_m3"], df["Vruiss_mod_m3"],
                    "Vruiss obs (m³)", "Vruiss mod (m³)",
                    f"[{tag}] Volumes ruissellement : obs vs mod",
                    out_dir / f"{tag}_obs_mod_Vruiss.png")

    scatter_obs_mod(df["Qmax_obs_LH"], df["Qmax_mod_LH"],
                    "Qmax obs (L/h)", "Qmax mod (L/h)",
                    f"[{tag}] Pics ruissellement : obs vs mod",
                    out_dir / f"{tag}_obs_mod_Qmax.png")

    hist(df["Vruiss_ratio"], "Vruiss_mod / Vruiss_obs",
         f"[{tag}] Distribution ratio volume ruisselé",
         out_dir / f"{tag}_hist_ratio_Vruiss.png")

    hist(df["RMSE_Qruiss_LH"], "RMSE (L/h)",
         f"[{tag}] Distribution RMSE Qruiss (évènement)",
         out_dir / f"{tag}_hist_RMSE_Qruiss.png")

    hist(df["NSE_Qruiss"], "NSE (-∞ à 1)",
         f"[{tag}] Distribution NSE (évènement) sur Qruiss",
         out_dir / f"{tag}_hist_NSE_Qruiss.png")

    hist(df["KGE_Qruiss"], "KGE (-∞ à 1)",
         f"[{tag}] Distribution KGE (évènement) sur Qruiss",
         out_dir / f"{tag}_hist_KGE_Qruiss.png")

    hist(df["peak_timing_err_min"], "Δt pic (min) = mod - obs",
         f"[{tag}] Distribution erreur timing du pic",
         out_dir / f"{tag}_hist_peak_timing.png")

    scatter_xy(df["P_tot_mm"], df["Vruiss_ratio"],
               "Ptot (mm)", "Vruiss ratio",
               f"[{tag}] Biais de volume vs pluie totale",
               out_dir / f"{tag}_ratioVruiss_vs_Ptot.png")

    scatter_xy(df["Qmax_obs_LH"], df["Vruiss_ratio"],
               "Qmax obs (L/h)", "Vruiss ratio",
               f"[{tag}] Biais de volume vs intensité (Qmax)",
               out_dir / f"{tag}_ratioVruiss_vs_Qmaxobs.png")

    scatter_xy(df["duration_h"], df["NSE_Qruiss"],
               "Durée évènement (h)", "NSE",
               f"[{tag}] Performance (NSE) vs durée",
               out_dir / f"{tag}_NSE_vs_duration.png")


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = get_config()

    # 1) Lire simulation
    sim, t, dt_s, qinf_mod_col, t_sim0, t_sim1 = load_simulation(
        cfg["SIM_CSV"], cfg["PREF_QINF_MOD_COLS"]
    )

    print("============================================")
    print(f"Split analysé  : {cfg['SPLIT_NAME']}")
    print(f"Simulation     : {cfg['SIM_CSV']}")
    print(f"Période sim    : {t_sim0} -> {t_sim1}")
    print(f"dt_s           : {dt_s}")
    print(f"Proxy Qinf mod : {qinf_mod_col if qinf_mod_col else 'AUCUN'}")
    print("============================================\n")

    # 2) Construire table évènements (intersection période simulée)
    evt = build_events_table(
        root=cfg["ROOT"],
        meta_events=cfg["META_EVENTS"],
        t_sim0=t_sim0,
        t_sim1=t_sim1,
        use_all_events1=cfg["USE_ALL_EVENTS1"],
    )
    print(f"[OK] Evènements retenus (intersection période sim): {len(evt)}\n")

    # 3) Métriques évènementielles
    df_evt = compute_event_metrics(
        sim=sim,
        t=t,
        dt_s=dt_s,
        evt=evt,
        qinf_mod_col=qinf_mod_col,
        qmod_eps_lh=float(cfg["QMOD_EPS_LH"]),
    )

    out_xlsx = cfg["SPLIT_DIR"] / "metrics_events_plus.xlsx"
    df_evt.to_excel(out_xlsx, index=False)
    print(f"[OK] Métriques évènementielles -> {out_xlsx} (n={len(df_evt)})\n")

    if len(df_evt) == 0:
        print("[STOP] Aucun évènement exploitable après filtrage (Qmod trop faible / NaN / etc.).")
        raise SystemExit(0)

    # 4) Sélection events à tracer
    df_plot = select_events_to_plot(
        df_evt,
        plot_mode=str(cfg["PLOT_MODE"]),
        top_metric=str(cfg["TOP_METRIC"]),
        top_n=int(cfg["TOP_N"]),
        filter_expr=str(cfg["FILTER_EXPR"])
    )

    # 5) Figures évènementielles
    out_events_dir = ensure_event_dirs(cfg["SPLIT_DIR"] / "FIG_EVENTS")
    if bool(cfg["MAKE_EVENT_PNG"]):
        print(f"[INFO] Tracé évènements: mode={cfg['PLOT_MODE']} -> n={len(df_plot)}")
        for _, rr in df_plot.iterrows():
            plot_event_figure(
                sim=sim, t=t, dt_s=dt_s, rr=rr,
                qinf_mod_col=qinf_mod_col,
                out_events_dir=out_events_dir,
                rain_fraction=float(cfg["RAIN_FRACTION"]),
                rain_alpha=float(cfg["RAIN_ALPHA"]),
            )
        print(f"[OK] Figures évènements -> {out_events_dir}\n")
    else:
        print("[INFO] MAKE_EVENT_PNG=False : pas de figures évènement.\n")

    # 6) Diagnostics globaux
    out_diag_dir = ensure_diag_dirs(cfg["SPLIT_DIR"] / "FIG_DIAG")

    make_diag_for(df_evt, "all", out_diag_dir / "all")
    for tag in ["warmup", "calib", "valid"]:
        dsub = df_evt[df_evt["period"] == tag].copy()
        make_diag_for(dsub, tag, out_diag_dir / tag)

    print(f"[OK] Graphes diag -> {out_diag_dir}")


if __name__ == "__main__":
    main()
