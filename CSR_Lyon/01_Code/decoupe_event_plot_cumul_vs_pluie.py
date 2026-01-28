# -*- coding: utf-8 -*-
"""
Découpe d'évènements pluie–ruissellement pour le parking CSR (OTHU)

Objectif : ne garder que des évènements "propres" pour modélisation :
- évènement commence avec Q_ruiss ~ 0 ET Q_inf ~ 0 (et pluie ~0) sur une durée minimale
- évènement finit quand Q_inf revient à ~0 (et Q_ruiss ~0) durablement
- rejeter les évènements qui sont manifestement la fin d'un évènement précédent

ENTRÉE :
  CSR_Lyon/02_Data/Donnees_serie_complete_2022-2024_corrigee_AS.csv (sep=';')
  Colonnes attendues :
      Date ; Hauteur_de_pluie_mm ; Q_inf_LH ; Q_ruiss_LH

SORTIES :
  CSR_Lyon/02_Data/all_events1/YYYY/event_YYYY_NNN.csv
      date ; P_mm ; Q_inf_LH ; Q_ruiss_LH
  CSR_Lyon/03_Plots/Etude_hydrologique/YYYY/event_YYYY_NNN.png
  CSR_Lyon/02_Data/all_events1/hr_hp_events.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================================
# PARAMÈTRES
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # -> CSR_Lyon/

DATA_FILE = BASE_DIR / "02_Data" / "Donnees_serie_complete_2022-2024_corrigee_AS.csv"
OUT_EVENTS_DIR = BASE_DIR / "02_Data" / "all_events1"
OUT_PLOTS_DIR = BASE_DIR / "03_Plots" / "Etude_hydrologique"

A_SITE_M2 = 94.0

# --- Seuils pluie / débits
P_THR_MM = 0.0               # pluie considérée nulle (mets 0.05 ou 0.1 si bruit)
Q_THR_RUISS_LH = 2.0         # "wet noyau" si Qruiss dépasse ça

# --- Conditions ZERO pour start/end propre (à régler selon le bruit capteur)
Q0_RUISS_LH = 0.8            # "débit nul" ruissellement
Q0_INF_LH   = 0.8            # "débit nul" infiltration (drain)

START_ZERO_MIN = 60.0        # avant le noyau, exige START_ZERO_MIN minutes de (P~0,Qruiss~0,Qinf~0)
END_ZERO_MIN   = 60.0        # pour finir l'event, exige END_ZERO_MIN minutes de (P~0,Qruiss~0,Qinf~0)

# --- Détection du noyau (comme avant)
END_DRY_MINUTES = 60.0       # fin noyau : pluie nulle + Qruiss faible durablement
Q_END_RUISS_LH = 1.0

DRY_MIN_HOURS = 1.0          # séparation d'évènements

# --- Filtres soft
MIN_CORE_DURATION_MIN = 6.0
MIN_HP_MM = 0.05
MIN_QMAX_RUISS_LH = 3.0

# --- Plots
MAKE_PLOTS = True
PLOT_DPI = 180

# Optionnel : afficher un peu après la fin export (PLOT SEULEMENT)
PLOT_POST_PAD_MIN = 120.0    # 0 si tu ne veux pas

# ======================================================================
# OUTILS
# ======================================================================

def parse_datetime_series(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("Colonne manquante: 'Date'")
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(5)
        raise ValueError(f"Dates illisibles. Exemples:\n{bad}")
    return df

def infer_dt_seconds(dates: pd.Series) -> float:
    dt = dates.diff().dt.total_seconds().dropna()
    if len(dt) == 0:
        raise ValueError("Impossible d'inférer dt (série trop courte).")
    return float(np.median(dt))

def build_is_wet(P_mm: np.ndarray, Q_ruiss_LH: np.ndarray) -> np.ndarray:
    return (P_mm > P_THR_MM) | (Q_ruiss_LH > Q_THR_RUISS_LH)

def is_end_of_core(P_mm: np.ndarray, Q_ruiss_LH: np.ndarray, i: int, dt_sec: float) -> bool:
    """
    Fin core = END_DRY_MINUTES consécutives avec :
      P <= P_THR_MM ET Qruiss <= Q_END_RUISS_LH
    """
    n = len(P_mm)
    need = max(int(np.ceil(END_DRY_MINUTES * 60.0 / dt_sec)), 1)
    if i + need >= n:
        return True
    for j in range(i, i + need):
        if P_mm[j] > P_THR_MM:
            return False
        if Q_ruiss_LH[j] > Q_END_RUISS_LH:
            return False
    return True

def find_core_events(P_mm: np.ndarray, Q_ruiss_LH: np.ndarray, dt_sec: float):
    """
    Début core = premier wet après DRY_MIN_HOURS secs
    Fin core = is_end_of_core
    """
    n = len(P_mm)
    if n == 0:
        return []

    is_wet = build_is_wet(P_mm, Q_ruiss_LH)
    dry_need = max(int(np.ceil(DRY_MIN_HOURS * 3600.0 / dt_sec)), 1)

    events = []
    i = 0

    # skip wet au tout début
    while i < n and is_wet[i]:
        i += 1

    while i < n:
        dry_count = 0
        while i < n and not is_wet[i]:
            dry_count += 1
            i += 1
        if i >= n:
            break

        # si pas assez sec, on laisse quand même (le recadrage "start propre" filtrera)
        i0 = i

        i1 = None
        while i < n:
            if is_end_of_core(P_mm, Q_ruiss_LH, i, dt_sec):
                i1 = i - 1
                skip = max(int(np.ceil(END_DRY_MINUTES * 60.0 / dt_sec)), 1)
                i = min(i + skip, n)
                break
            i += 1

        if i1 is None:
            i1 = n - 1

        if i0 <= i1:
            events.append((i0, i1))

    return events

def _need_samples(minutes: float, dt_sec: float) -> int:
    return max(int(np.ceil(minutes * 60.0 / dt_sec)), 1)

def find_clean_start(i0_core: int, P: np.ndarray, Qr: np.ndarray, Qi: np.ndarray, dt_sec: float):
    """
    On exige START_ZERO_MIN minutes juste avant i0_core avec P~0 & Qr~0 & Qi~0.
    Si ok : on retourne i0_evt = i0_core - start_need (=> l'event commence "à zéro")
    Sinon : None
    """
    need = _need_samples(START_ZERO_MIN, dt_sec)
    i0 = i0_core - need
    if i0 < 0:
        return None

    cond = (P[i0:i0_core] <= P_THR_MM) & (Qr[i0:i0_core] <= Q0_RUISS_LH) & (Qi[i0:i0_core] <= Q0_INF_LH)
    if np.all(cond):
        return i0
    return None

def find_clean_end(i1_core: int, P: np.ndarray, Qr: np.ndarray, Qi: np.ndarray, dt_sec: float):
    """
    On cherche la première fenêtre END_ZERO_MIN minutes après i1_core
    où P~0 & Qr~0 & Qi~0. L'event se termine à la fin de cette fenêtre.
    Si non trouvé : None
    """
    n = len(P)
    need = _need_samples(END_ZERO_MIN, dt_sec)

    start = i1_core + 1
    if start >= n:
        return None

    # scan fenêtre glissante
    for k in range(start, n - need + 1):
        window_ok = (P[k:k+need] <= P_THR_MM) & (Qr[k:k+need] <= Q0_RUISS_LH) & (Qi[k:k+need] <= Q0_INF_LH)
        if np.all(window_ok):
            return k + need - 1  # fin de la fenêtre zéro => event finit "à zéro"
    return None

def compute_event_metrics(df_part: pd.DataFrame, dt_sec: float, area_m2: float):
    hp_mm = float(df_part["P_mm"].sum())

    Qr_LH = df_part["Q_ruiss_LH"].to_numpy(dtype=float)
    Qi_LH = df_part["Q_inf_LH"].to_numpy(dtype=float)

    Qr_m3s = (Qr_LH / 3600.0) / 1000.0
    Qi_m3s = (Qi_LH / 3600.0) / 1000.0

    Vr_m3 = float(np.sum(Qr_m3s) * dt_sec)
    Vi_m3 = float(np.sum(Qi_m3s) * dt_sec)

    hr_mm = 1000.0 * Vr_m3 / area_m2 if area_m2 > 0 else np.nan
    hi_mm = 1000.0 * Vi_m3 / area_m2 if area_m2 > 0 else np.nan

    return dict(
        hp_mm=hp_mm,
        hr_mm=hr_mm,
        hi_mm=hi_mm,
        Vruiss_m3=Vr_m3,
        Vinf_m3=Vi_m3,
        Qmax_ruiss_LH=float(np.max(Qr_LH)) if len(Qr_LH) else 0.0,
        Qmax_inf_LH=float(np.max(Qi_LH)) if len(Qi_LH) else 0.0,
        dur_min=float(len(df_part) * dt_sec / 60.0),
    )

def plot_event(df_evt: pd.DataFrame, out_png: Path, title: str, dt_sec: float, area_m2: float):
    t = df_evt["date"]
    P = df_evt["P_mm"].to_numpy(dtype=float)
    Qr_LH = df_evt["Q_ruiss_LH"].to_numpy(dtype=float)
    Qi_LH = df_evt["Q_inf_LH"].to_numpy(dtype=float)

    P_cum_mm = np.cumsum(P)

    Qr_m3s = (Qr_LH / 3600.0) / 1000.0
    Qi_m3s = (Qi_LH / 3600.0) / 1000.0

    Vr_m3 = np.cumsum(Qr_m3s) * dt_sec
    Vi_m3 = np.cumsum(Qi_m3s) * dt_sec

    Hr_cum_mm = 1000.0 * Vr_m3 / area_m2
    Hi_cum_mm = 1000.0 * Vi_m3 / area_m2

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax2 = ax1.twinx()
    ax2.bar(t, P, width=0.002, alpha=0.3)
    ax2.set_ylabel("P (mm / pas)")

    ax1.plot(t, Qr_LH, label="Q_ruiss (L/h)")
    ax1.plot(t, Qi_LH, label="Q_inf (L/h)")
    ax1.set_ylabel("Débit (L/h)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax3.plot(t, P_cum_mm, label="Pluie cumulée (mm)")
    ax3.plot(t, Hr_cum_mm, label="Ruissellement cumulé (mm)")
    ax3.plot(t, Hi_cum_mm, label="Infiltration cumulée (mm)")
    ax3.set_ylabel("Cumul (mm)")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    plt.suptitle(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)
    
def plot_event_volume_scatter(meta: pd.DataFrame, out_png: Path, use_mm=True):
    """
    Scatter inter-évènements:
      - soit hr(mm) vs hp(mm) (use_mm=True)
      - soit Vruiss(m3) vs Vpluie(m3) (use_mm=False)
    + ligne y=x (ruissellement = pluie), juste pour se repérer.
    """

    if meta is None or len(meta) == 0:
        return

    if use_mm:
        x = meta["hp_mm"].to_numpy(dtype=float)
        y = meta["hr_mm"].to_numpy(dtype=float)
        xlabel = "hp (mm) pluie évènement"
        ylabel = "hr (mm) ruissellement évènement"
        title = "Nuage hr=f(hp) (mm)"
    else:
        # Volume pluie = hp_mm * area_m2 / 1000
        A = float(A_SITE_M2)
        Vp_m3 = meta["hp_mm"].to_numpy(dtype=float) * A / 1000.0
        x = Vp_m3
        y = meta["Vruiss_m3"].to_numpy(dtype=float)
        xlabel = "Vpluie (m³)"
        ylabel = "Vruiss (m³)"
        title = "Nuage Vruiss=f(Vpluie) (m³)"

    # bornes pour la droite y=x
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    xmax = float(np.max(x[finite]))
    ymax = float(np.max(y[finite]))
    m = max(xmax, ymax)
    if m <= 0:
        m = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.scatter(x, y, s=18, alpha=0.8)

    ax.plot([0, m], [0, m], linewidth=1.5, alpha=0.7)  # y=x

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)
    
def plot_hr_hp_with_event_ids(meta: pd.DataFrame, out_png: Path,
                              x_max: float = None, y_max: float = None,
                              max_list: int = 140,
                              label_only_in_window: bool = False):
    """
    Scatter hr(mm) vs hp(mm) avec:
      - un numéro sur chaque point
      - une colonne texte à droite: numéro -> event_id (+ hp/hr)
    Option:
      - x_max/y_max: applique un zoom (xlim/ylim)
      - label_only_in_window: si True, ne numérote/liste QUE les points dans la fenêtre.
    """

    if meta is None or len(meta) == 0:
        return

    dfp = meta.copy()
    dfp["hp_mm"] = pd.to_numeric(dfp["hp_mm"], errors="coerce")
    dfp["hr_mm"] = pd.to_numeric(dfp["hr_mm"], errors="coerce")
    dfp = dfp[np.isfinite(dfp["hp_mm"]) & np.isfinite(dfp["hr_mm"])].reset_index(drop=True)
    if len(dfp) == 0:
        return

    # Filtre fenêtre si demandé
    if x_max is not None or y_max is not None:
        mask = np.ones(len(dfp), dtype=bool)
        if x_max is not None:
            mask &= (dfp["hp_mm"].to_numpy() <= float(x_max))
        if y_max is not None:
            mask &= (dfp["hr_mm"].to_numpy() <= float(y_max))
        if label_only_in_window:
            dfp = dfp[mask].reset_index(drop=True)

    # Figure: 2 colonnes (scatter + liste)
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.8])
    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])

    x = dfp["hp_mm"].to_numpy(dtype=float)
    y = dfp["hr_mm"].to_numpy(dtype=float)

    ax.scatter(x, y, s=18, alpha=0.85)

    # Droite y=x pour repère (sur la fenêtre affichée)
    xmax = float(np.max(x)) if x_max is None else float(x_max)
    ymax = float(np.max(y)) if y_max is None else float(y_max)
    m = max(xmax, ymax, 1.0)
    ax.plot([0, m], [0, m], linewidth=1.2, alpha=0.6)

    if x_max is not None:
        ax.set_xlim(0, float(x_max))
    if y_max is not None:
        ax.set_ylim(0, float(y_max))

    ax.set_xlabel("hp (mm) pluie évènement")
    ax.set_ylabel("hr (mm) ruissellement évènement")
    title = "Nuage hr=f(hp) avec identifiants"
    if x_max is not None or y_max is not None:
        title += f" (zoom: hp≤{x_max if x_max is not None else '∞'} ; hr≤{y_max if y_max is not None else '∞'})"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # Numérotation + liste
    ax_txt.axis("off")

    n = len(dfp)
    if n > max_list:
        # On garde les points les plus "importants" visuellement (par exemple gros hr puis gros hp)
        dfp = dfp.sort_values(["hr_mm", "hp_mm"], ascending=False).head(max_list).reset_index(drop=True)
        x = dfp["hp_mm"].to_numpy(dtype=float)
        y = dfp["hr_mm"].to_numpy(dtype=float)
        n = len(dfp)

    # numéros sur les points
    for k in range(n):
        ax.text(x[k], y[k], str(k + 1),
                fontsize=8, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", alpha=0.25, linewidth=0.0))

    # colonne texte à droite
    lines = []
    for k in range(n):
        eid = str(dfp.loc[k, "event_id"])
        hp = float(dfp.loc[k, "hp_mm"])
        hr = float(dfp.loc[k, "hr_mm"])
        lines.append(f"{k+1:>3}  {eid}  (hp={hp:.2f}, hr={hr:.2f})")

    ax_txt.text(0.0, 1.0, "\n".join(lines),
                va="top", ha="left", fontsize=9, family="monospace")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)
   


# ======================================================================
# MAIN
# ======================================================================
def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, sep=";")

    required = ["Date", "Hauteur_de_pluie_mm", "Q_inf_LH", "Q_ruiss_LH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}\nColonnes présentes: {list(df.columns)}")

    df = parse_datetime_series(df)
    df["P_mm"] = pd.to_numeric(df["Hauteur_de_pluie_mm"], errors="coerce").fillna(0.0)
    df["Q_inf_LH"] = pd.to_numeric(df["Q_inf_LH"], errors="coerce").fillna(0.0)
    df["Q_ruiss_LH"] = pd.to_numeric(df["Q_ruiss_LH"], errors="coerce").fillna(0.0)
    df = df.sort_values("date").reset_index(drop=True)

    dt_sec = infer_dt_seconds(df["date"])
    print(f"[INFO] dt estimé = {dt_sec:.1f} s")

    P  = df["P_mm"].to_numpy(dtype=float)
    Qr = df["Q_ruiss_LH"].to_numpy(dtype=float)
    Qi = df["Q_inf_LH"].to_numpy(dtype=float)

    core_events = find_core_events(P, Qr, dt_sec)
    print(f"[INFO] Nombre d'évènements noyau détectés : {len(core_events)}")

    OUT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    event_counters = {}
    skips = {"soft_filter": 0, "no_clean_start": 0, "no_clean_end": 0, "written": 0}

    for (i0_core, i1_core) in core_events:
        df_core = df.iloc[i0_core:i1_core+1][["date", "P_mm", "Q_inf_LH", "Q_ruiss_LH"]].copy()

        dur_core_min = len(df_core) * dt_sec / 60.0
        hp_core = float(df_core["P_mm"].sum())
        qmax_core = float(df_core["Q_ruiss_LH"].max()) if len(df_core) else 0.0

        # soft filters
        if dur_core_min < MIN_CORE_DURATION_MIN or hp_core < MIN_HP_MM or qmax_core < MIN_QMAX_RUISS_LH:
            skips["soft_filter"] += 1
            continue

        # start/end propres
        i0_evt = find_clean_start(i0_core, P, Qr, Qi, dt_sec)
        if i0_evt is None:
            skips["no_clean_start"] += 1
            continue

        i1_evt = find_clean_end(i1_core, P, Qr, Qi, dt_sec)
        if i1_evt is None:
            skips["no_clean_end"] += 1
            continue

        df_evt = df.iloc[i0_evt:i1_evt+1][["date", "P_mm", "Q_inf_LH", "Q_ruiss_LH"]].copy()

        met_evt = compute_event_metrics(df_evt, dt_sec, A_SITE_M2)
        yr = int(df_evt["date"].iloc[0].year)

        event_counters.setdefault(yr, 0)
        event_counters[yr] += 1
        eid = f"event_{yr}_{event_counters[yr]:03d}"

        out_csv = OUT_EVENTS_DIR / f"{yr}" / f"{eid}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_evt.to_csv(out_csv, sep=";", index=False)

        # plot : event + optionnel post-pad uniquement visuel
        if MAKE_PLOTS:
            i1_plot = i1_evt
            if PLOT_POST_PAD_MIN > 0:
                add = int(np.ceil((PLOT_POST_PAD_MIN * 60.0) / dt_sec))
                i1_plot = min(i1_evt + add, len(df) - 1)
            df_plot = df.iloc[i0_evt:i1_plot+1][["date", "P_mm", "Q_inf_LH", "Q_ruiss_LH"]].copy()

            out_png = OUT_PLOTS_DIR / f"{yr}" / f"{eid}.png"
            title = (
                f"{eid} | hp={met_evt['hp_mm']:.2f} mm | hr={met_evt['hr_mm']:.2f} mm | hi={met_evt['hi_mm']:.2f} mm | "
                f"Qmax={met_evt['Qmax_ruiss_LH']:.2f} L/h | dur={met_evt['dur_min']:.0f} min"
            )
            plot_event(df_plot, out_png, title, dt_sec=dt_sec, area_m2=A_SITE_M2)

        meta_rows.append(dict(
            year=yr,
            event_id=eid,
            dt_sec=dt_sec,
            i0_core=i0_core, i1_core=i1_core,
            i0_evt=i0_evt, i1_evt=i1_evt,
            file_csv=str(out_csv),

            hp_mm=met_evt["hp_mm"],
            hr_mm=met_evt["hr_mm"],
            hi_mm=met_evt["hi_mm"],
            Vruiss_m3=met_evt["Vruiss_m3"],
            Vinf_m3=met_evt["Vinf_m3"],
            Qmax_ruiss_LH=met_evt["Qmax_ruiss_LH"],
            Qmax_inf_LH=met_evt["Qmax_inf_LH"],
            dur_evt_min=met_evt["dur_min"],
        ))

        skips["written"] += 1

    meta = pd.DataFrame(meta_rows)
    out_meta = OUT_EVENTS_DIR / "hr_hp_events.csv"
    meta.to_csv(out_meta, sep=";", index=False)

    # --- PLOTS DE SYNTHESE inter-évènements
    if MAKE_PLOTS and len(meta):
        out_scatter_mm = OUT_PLOTS_DIR / "SUMMARY_hr_hp_mm.png"
        plot_event_volume_scatter(meta, out_scatter_mm, use_mm=True)

        out_scatter_m3 = OUT_PLOTS_DIR / "SUMMARY_Vruiss_Vpluie_m3.png"
        plot_event_volume_scatter(meta, out_scatter_m3, use_mm=False)

        # --- NOUVEAU: scatter avec IDs lisibles à droite
        out_ids = OUT_PLOTS_DIR / "SUMMARY_hr_hp_mm_WITH_IDS.png"
        plot_hr_hp_with_event_ids(meta, out_ids, max_list=160)

        # --- NOUVEAU: scatter zoomé sur hp<=60 mm et hr<=35 mm (on n’étiquette que ceux dans la fenêtre)
        out_zoom = OUT_PLOTS_DIR / "SUMMARY_hr_hp_mm_ZOOM_60_35_WITH_IDS.png"
        plot_hr_hp_with_event_ids(
            meta, out_zoom,
            x_max=60.0, y_max=35.0,
            max_list=250,
            label_only_in_window=True
        )

        print(f"[OK] Scatter mm : {out_scatter_mm}")
        print(f"[OK] Scatter m3 : {out_scatter_m3}")
        print(f"[OK] Scatter IDs : {out_ids}")
        print(f"[OK] Scatter ZOOM : {out_zoom}")

    print(f"[OK] Evènements exportés (propres) : {len(meta)}")
    print(f"[OK] Méta-fichier : {out_meta}")
    print(f"[INFO] Skips: {skips}")

    if len(meta):
        print(meta[["year", "event_id", "hp_mm", "hr_mm", "hi_mm", "Qmax_ruiss_LH", "dur_evt_min"]].head(15))


if __name__ == "__main__":
    main()
