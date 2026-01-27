# -*- coding: utf-8 -*-
"""
ANALYSE VITESSES D'INFILTRATION PAR EVENEMENT (CSR Lyon)
=======================================================

But:
- Partir des évènements déjà découpés (all_events1) + méta hr_hp_events.csv
- Construire v_inf(t) = Q_inf(t) / A en mm/h (champ temporel par évènement)
- Produire :
  (1) Excel synthèse (stats par évènement)
  (2) Plots globaux (distribution, relations avec hp/hi/durée)
  (3) Optionnel: figures v_inf(t) par évènement

Entrées:
- CSR_Lyon/02_Data/all_events1/hr_hp_events.csv
- CSR_Lyon/02_Data/all_events1/YYYY/event_YYYY_NNN.csv

Sorties:
- CSR_Lyon/03_Plots/Etude_hydrologique/INFILTRATION_VELOCITY/
    infiltration_velocity_events.xlsx
    *.png (synthèse)
    EVENTS/*.png (optionnel)

Notes importantes (sans blabla):
- v_inf(t) ici = flux moyen au pas de temps sur la surface A_SITE_M2, en mm/h.
- Si Q_inf_LH contient du bruit et des valeurs négatives: on clippe à 0 par défaut.
  Tu peux désactiver si tu veux diagnostiquer les artefacts capteur.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
def get_config():
    try:
        root = Path(__file__).resolve().parents[1]  # -> CSR_Lyon/
    except NameError:
        root = Path.cwd()

    cfg = dict(
        ROOT=root,
        A_SITE_M2=94.0,

        META_EVENTS=root / "02_Data" / "all_events1" / "hr_hp_events.csv",
        EVENTS_ROOT=root / "02_Data" / "all_events1",

        OUT_DIR=root / "03_Plots" / "Etude_hydrologique" / "INFILTRATION_VELOCITY",
        OUT_XLSX_NAME="infiltration_velocity_events.xlsx",

        # Nettoyage capteur
        CLIP_QINF_NEGATIVE=True,    # True = max(Qinf,0). False = garder le signe.
        QINF_ZERO_THR_LH=0.0,       # pour définir "temps actif" (0.0 ou un petit seuil)

        # Courbes par évènement (optionnel)
        MAKE_EVENT_PNG=True,
        EVENT_PLOT_MODE="topN",     # "all" | "topN"
        TOP_N=30,
        TOP_BY="v_inf_p95_mm_h",    # colonne de tri pour topN

        # Plots synthèse
        MAKE_SUMMARY_PNG=True,
        DPI=170,
    )
    return cfg


# =========================================================
# Utils
# =========================================================
def infer_dt_seconds(time_index: pd.DatetimeIndex) -> float:
    dt_s = float(pd.Series(time_index).diff().dropna().dt.total_seconds().median())
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise RuntimeError("Impossible d'inférer dt (dates non régulières / invalides).")
    return dt_s


def parse_year_evt(row):
    """
    Supporte:
      - year, event_id = (2023, "event_2023_012")
      - year, evt_num  = (2023, 12)
      - ("2023", "12")
    """
    year_raw = row.iloc[0]
    evt_raw = row.iloc[1]

    try:
        year = int(str(year_raw).strip())
    except Exception:
        year = None

    evt_s = str(evt_raw).strip()
    m = re.search(r"event_(\d{4})_(\d{1,4})", evt_s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # "YYYY ... NNN"
    m = re.search(r"(\d{4})\D+(\d{1,4})$", evt_s)
    if m:
        return int(m.group(1)), int(m.group(2))

    try:
        evt_num = int(evt_s)
    except Exception:
        evt_num = None

    return year, evt_num


def read_event_csv(events_root: Path, year: int, evt_num: int, sep=";"):
    name = f"event_{year}_{evt_num:03d}"
    csv_path = events_root / str(year) / f"{name}.csv"
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path, sep=sep)
    if "date" not in df.columns:
        return None, None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        return None, None

    # Colonnes attendues (ton découpage)
    for c in ["P_mm", "Q_inf_LH", "Q_ruiss_LH"]:
        if c not in df.columns:
            return None, None

    return name, df


def qlh_to_vmmh(q_lh, area_m2):
    """
    Convertit Q (L/h) -> v (mm/h) sur la surface area_m2:
      1 L = 0.001 m3
      mm/h = (m3/h) / A * 1000
          = (Q_lh*0.001)/A * 1000
          = Q_lh / A
    Donc: v(mm/h) = Q(L/h) / A(m2)  (simple et exact)
    """
    return np.asarray(q_lh, float) / float(area_m2)


def safe_percentile(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, p))


def safe_max(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if len(x) else np.nan


def safe_mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


# =========================================================
# Analyse par évènement
# =========================================================
def compute_velocity_metrics(df_evt: pd.DataFrame, A_site_m2: float,
                             clip_qinf_negative: bool, qinf_zero_thr_lh: float):
    t = pd.DatetimeIndex(df_evt["date"])
    dt_s = infer_dt_seconds(t)

    P = pd.to_numeric(df_evt["P_mm"], errors="coerce").fillna(0.0).to_numpy(float)
    Qinf = pd.to_numeric(df_evt["Q_inf_LH"], errors="coerce").fillna(0.0).to_numpy(float)
    Qru  = pd.to_numeric(df_evt["Q_ruiss_LH"], errors="coerce").fillna(0.0).to_numpy(float)

    if clip_qinf_negative:
        Qinf = np.clip(Qinf, 0.0, None)

    v_inf = qlh_to_vmmh(Qinf, A_site_m2)  # mm/h, à chaque pas

    duration_h = float(len(df_evt) * dt_s / 3600.0)
    duration_min = float(len(df_evt) * dt_s / 60.0)

    hp_mm = float(np.sum(np.clip(P, 0.0, None)))

    # volumes -> lames (comme ton code)
    # hi_mm (mm) = integral(Qinf)/A ; ici en discret:
    # Qinf (L/h) -> m3/s: /1000/3600 ; *dt -> m3 ; /A *1000 -> mm
    Qinf_m3s = (Qinf / 1000.0) / 3600.0
    Vinf_m3 = float(np.sum(Qinf_m3s) * dt_s)
    hi_mm = 1000.0 * Vinf_m3 / A_site_m2

    Qru_m3s = (Qru / 1000.0) / 3600.0
    Vru_m3 = float(np.sum(np.clip(Qru_m3s, 0, None)) * dt_s)
    hr_mm = 1000.0 * Vru_m3 / A_site_m2

    # "activité" infiltration (temps où Qinf dépasse un seuil)
    thr = float(qinf_zero_thr_lh)
    active = (Qinf > thr) & np.isfinite(Qinf)
    frac_active = float(np.mean(active)) if len(active) else np.nan
    active_h = float(np.sum(active) * dt_s / 3600.0)

    # timing du pic v_inf
    if np.isfinite(v_inf).any():
        i_peak = int(np.nanargmax(v_inf))
        t_peak = pd.Timestamp(t[i_peak])
        t0 = pd.Timestamp(t[0])
        ttp_min = float((t_peak - t0).total_seconds() / 60.0)
    else:
        ttp_min = np.nan

    # stats robustes
    v_p50 = safe_percentile(v_inf, 50)
    v_p75 = safe_percentile(v_inf, 75)
    v_p90 = safe_percentile(v_inf, 90)
    v_p95 = safe_percentile(v_inf, 95)
    v_max = safe_max(v_inf)
    v_mean = safe_mean(v_inf)

    # moyenne conditionnelle sur périodes actives
    v_mean_active = safe_mean(v_inf[active]) if np.any(active) else np.nan

    return dict(
        dt_sec=dt_s,
        t_start=str(pd.Timestamp(t.min())),
        t_end=str(pd.Timestamp(t.max())),
        duration_min=duration_min,
        duration_h=duration_h,

        hp_mm=hp_mm,
        hi_mm=hi_mm,
        hr_mm=hr_mm,

        Qmax_inf_LH=float(np.nanmax(Qinf)) if np.isfinite(Qinf).any() else np.nan,
        Qmax_ruiss_LH=float(np.nanmax(Qru)) if np.isfinite(Qru).any() else np.nan,

        v_inf_mean_mm_h=v_mean,
        v_inf_mean_active_mm_h=v_mean_active,
        v_inf_p50_mm_h=v_p50,
        v_inf_p75_mm_h=v_p75,
        v_inf_p90_mm_h=v_p90,
        v_inf_p95_mm_h=v_p95,
        v_inf_max_mm_h=v_max,
        t_to_peak_min=ttp_min,
        frac_active=frac_active,
        active_h=active_h,
    ), v_inf


def plot_event_velocity(df_evt, v_inf, out_png: Path, title: str, dpi: int):
    t = pd.DatetimeIndex(df_evt["date"])
    P = pd.to_numeric(df_evt["P_mm"], errors="coerce").fillna(0.0).to_numpy(float)
    Qinf = pd.to_numeric(df_evt["Q_inf_LH"], errors="coerce").fillna(0.0).to_numpy(float)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    # haut: Qinf et pluie (axe secondaire)
    ax1.plot(t, Qinf, label="Q_inf (L/h)")
    ax1.set_ylabel("Q_inf (L/h)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax1b = ax1.twinx()
    # bar width en jours approx: on prend median dt
    dt_s = infer_dt_seconds(t)
    w = float(dt_s) / 86400.0
    ax1b.bar(t, P, width=w, alpha=0.25)
    ax1b.set_ylabel("P (mm/pas)")
    ax1b.grid(False)

    # bas: v_inf
    ax2.plot(t, v_inf, label="v_inf = Q_inf/A (mm/h)")
    ax2.set_ylabel("v_inf (mm/h)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    ax2.set_xlabel("Date")

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


def hist_plot(x, xlabel, title, out_png: Path, dpi: int, bins=30):
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
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


def scatter_plot(x, y, xlabel, ylabel, title, out_png: Path, dpi: int):
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
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = get_config()
    out_dir = Path(cfg["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(cfg["META_EVENTS"])
    if not meta_path.exists():
        raise FileNotFoundError(f"Méta events introuvable: {meta_path}")

    meta = pd.read_csv(meta_path, sep=";")
    if meta.shape[1] < 2:
        raise RuntimeError("hr_hp_events.csv doit avoir au moins 2 colonnes (année, id/num).")

    rows = []
    all_v_samples = []  # pour histogramme global
    event_series = []   # pour feuille Excel "time series" optionnelle

    events_root = Path(cfg["EVENTS_ROOT"])
    A_site = float(cfg["A_SITE_M2"])

    n_ok = 0
    n_skip = 0

    for _, r in meta.iterrows():
        year, evt_num = parse_year_evt(r)
        if year is None or evt_num is None:
            n_skip += 1
            continue

        name, df_evt = read_event_csv(events_root, year, evt_num, sep=";")
        if df_evt is None:
            n_skip += 1
            continue

        met, v_inf = compute_velocity_metrics(
            df_evt=df_evt,
            A_site_m2=A_site,
            clip_qinf_negative=bool(cfg["CLIP_QINF_NEGATIVE"]),
            qinf_zero_thr_lh=float(cfg["QINF_ZERO_THR_LH"]),
        )

        met.update(dict(
            year=int(year),
            evt_num=int(evt_num),
            event_name=str(name),
            file_csv=str(events_root / str(year) / f"{name}.csv"),
        ))
        rows.append(met)

        # garde des échantillons pour histo global
        vv = np.asarray(v_inf, float)
        vv = vv[np.isfinite(vv)]
        if len(vv):
            all_v_samples.append(vv)

        # série pour feuille Excel (sparse)
        df_s = pd.DataFrame({
            "event_name": name,
            "date": pd.to_datetime(df_evt["date"]),
            "P_mm": pd.to_numeric(df_evt["P_mm"], errors="coerce"),
            "Q_inf_LH": pd.to_numeric(df_evt["Q_inf_LH"], errors="coerce"),
            "v_inf_mm_h": v_inf,
        })
        event_series.append(df_s)

        n_ok += 1

    df_out = pd.DataFrame(rows).sort_values(["year", "evt_num"]).reset_index(drop=True)
    print(f"[OK] Evènements analysés: {n_ok} | skip: {n_skip}")

    # Excel
    out_xlsx = out_dir / cfg["OUT_XLSX_NAME"]
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df_out.to_excel(xw, sheet_name="event_metrics", index=False)

        # feuille time series (grosse) -> tu peux commenter si trop lourd
        if len(event_series):
            ts = pd.concat(event_series, ignore_index=True)
            ts.to_excel(xw, sheet_name="v_inf_timeseries", index=False)

    print(f"[OK] Excel -> {out_xlsx}")

    # Plots synthèse
    if bool(cfg["MAKE_SUMMARY_PNG"]) and len(df_out) > 0:
        v_global = np.concatenate(all_v_samples) if len(all_v_samples) else np.array([], float)

        hist_plot(
            v_global,
            xlabel="v_inf (mm/h) [tous pas, tous évènements]",
            title="Distribution globale du champ v_inf(t)",
            out_png=out_dir / "HIST_v_inf_all_samples.png",
            dpi=int(cfg["DPI"]),
            bins=40
        )

        # Distribution des stats par évènement
        hist_plot(df_out["v_inf_mean_mm_h"], "v_inf mean (mm/h)", "Distribution v_inf_mean (par évènement)",
                  out_dir / "HIST_v_inf_mean_per_event.png", int(cfg["DPI"]), bins=30)
        hist_plot(df_out["v_inf_p95_mm_h"], "v_inf p95 (mm/h)", "Distribution v_inf_p95 (par évènement)",
                  out_dir / "HIST_v_inf_p95_per_event.png", int(cfg["DPI"]), bins=30)
        hist_plot(df_out["v_inf_max_mm_h"], "v_inf max (mm/h)", "Distribution v_inf_max (par évènement)",
                  out_dir / "HIST_v_inf_max_per_event.png", int(cfg["DPI"]), bins=30)

        # Relations utiles
        scatter_plot(df_out["hp_mm"], df_out["v_inf_p95_mm_h"],
                     "hp (mm)", "v_inf p95 (mm/h)",
                     "Intensité infiltration (p95) vs pluie totale",
                     out_dir / "SCAT_vp95_vs_hp.png", int(cfg["DPI"]))
        scatter_plot(df_out["hi_mm"], df_out["v_inf_p95_mm_h"],
                     "hi (mm)", "v_inf p95 (mm/h)",
                     "Intensité infiltration (p95) vs lame infiltrée",
                     out_dir / "SCAT_vp95_vs_hi.png", int(cfg["DPI"]))
        scatter_plot(df_out["duration_h"], df_out["v_inf_mean_mm_h"],
                     "Durée (h)", "v_inf mean (mm/h)",
                     "v_inf_mean vs durée (signature drainage / plateau)",
                     out_dir / "SCAT_vmean_vs_duration.png", int(cfg["DPI"]))
        scatter_plot(df_out["t_to_peak_min"], df_out["v_inf_max_mm_h"],
                     "Temps au pic (min)", "v_inf max (mm/h)",
                     "Réactivité infiltration: pic vs temps au pic",
                     out_dir / "SCAT_vmax_vs_ttp.png", int(cfg["DPI"]))

        print(f"[OK] Plots synthèse -> {out_dir}")

    # Courbes par évènement
    if bool(cfg["MAKE_EVENT_PNG"]) and len(df_out) > 0:
        out_ev_dir = out_dir / "EVENTS"
        out_ev_dir.mkdir(parents=True, exist_ok=True)

        df_plot = df_out.copy()
        mode = str(cfg["EVENT_PLOT_MODE"]).lower()
        if mode == "topn":
            key = str(cfg["TOP_BY"])
            if key in df_plot.columns:
                df_plot = df_plot.sort_values(key, ascending=False).head(int(cfg["TOP_N"]))
            else:
                df_plot = df_plot.head(int(cfg["TOP_N"]))

        for _, rr in df_plot.iterrows():
            # reload event pour plot (simple, robuste)
            m = re.search(r"event_(\d{4})_(\d{1,4})", str(rr["event_name"]))
            if not m:
                continue
            y = int(m.group(1))
            n = int(m.group(2))
            name, df_evt = read_event_csv(events_root, y, n, sep=";")
            if df_evt is None:
                continue

            _, v_inf = compute_velocity_metrics(
                df_evt=df_evt,
                A_site_m2=A_site,
                clip_qinf_negative=bool(cfg["CLIP_QINF_NEGATIVE"]),
                qinf_zero_thr_lh=float(cfg["QINF_ZERO_THR_LH"]),
            )

            title = (
                f"{name} | v_mean={rr['v_inf_mean_mm_h']:.2f} mm/h | "
                f"v_p95={rr['v_inf_p95_mm_h']:.2f} | v_max={rr['v_inf_max_mm_h']:.2f} | "
                f"hp={rr['hp_mm']:.2f} mm | hi={rr['hi_mm']:.2f} mm | dur={rr['duration_h']:.2f} h"
            )
            plot_event_velocity(
                df_evt=df_evt,
                v_inf=v_inf,
                out_png=out_ev_dir / f"{name}_v_inf.png",
                title=title,
                dpi=int(cfg["DPI"])
            )

        print(f"[OK] Courbes v_inf(t) -> {out_ev_dir}")

    print("[DONE]")


if __name__ == "__main__":
    main()
