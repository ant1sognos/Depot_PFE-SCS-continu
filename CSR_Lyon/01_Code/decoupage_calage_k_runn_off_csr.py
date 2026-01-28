# -*- coding: utf-8 -*-
"""
one_k_clean_from_events.py
--------------------------
Parcourt tous les CSV évènementiels (2022-2024) et détecte des décrues "propres"
(SEC + retour à ~0 stable) puis fit un seul k (s^-1) via ln(Q)=a+bt.

Entrées attendues (dans chaque event CSV) :
  date ; P_mm ; Q_inf_LH ; Q_ruiss_LH

Dossiers parcourus (selon ta demande) :
  02_Data/Etude_hydrologique/2022
  02_Data/Etude_hydrologique/2023
  02_Data/Etude_hydrologique/2024

Sorties :
  03_Plots/Identification constantes vidange CSR/ONE_K_clean_FROM_EVENTS/Recessions_QA/*.png
  03_Plots/Identification constantes vidange CSR/ONE_K_clean_FROM_EVENTS/recessions_onek_from_events.xlsx
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0


# ======================================================================
# PARAMÈTRES
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # -> CSR_Lyon/

# Dossier racine où sont tes events (selon ton message)
EVENTS_ROOT = BASE_DIR / "02_Data" / "all_events1"
YEARS = [2022, 2023, 2024]

# Sorties
OUT_DIR = BASE_DIR / "03_Plots" / "Identification constantes runoff CSR"
QA_DIR = OUT_DIR / "Recessions_QA"

# Critères stricts (à ajuster)
RAIN_THRESH_MM = 0.0            # strictement sec
ALLOW_SMALL_BUMPS = True
BUMP_TOL_FRAC = 0.02            # tolérance relative des petites remontées de Q

# Seuils débits (en m3/s) pour la détection
Q_START_MIN_M3S = 1e-6          # pic minimal pour lancer une décrue
Q_ZERO_THRESH_M3S = 2e-7        # considéré ~0 en dessous
MIN_LEN_PTS_POS = 10            # points Q>0 min pour fitter
N_ZERO_END = 5                  # points consécutifs ~0 en fin de segment

# Fit
DROP_FIRST_N = 1                # retire 1 point après le pic (souvent perturbé)

# Graphiques
PLOT_DPI = 200


# ======================================================================
# LECTURE EVENT CSV
# ======================================================================

def read_event_csv(csv_path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, float]:
    """
    Retourne:
      time_index (DatetimeIndex),
      P_mm (mm/pas),
      Q_m3s (m3/s) basé sur Q_ruiss_LH,
      dt_sec (float)
    """
    df = pd.read_csv(csv_path, sep=";")

    # Supporte deux formats au cas où: (date,P_mm,...) ou (Date,Hauteur_de_pluie_mm,...)
    if "date" in df.columns:
        t = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        P = pd.to_numeric(df.get("P_mm", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        Q_LH = pd.to_numeric(df.get("Q_ruiss_LH", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        # fallback “série complète-like”
        t = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        P = pd.to_numeric(df["Hauteur_de_pluie_mm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        Q_LH = pd.to_numeric(df["Q_ruiss_LH"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if t.isna().any():
        raise ValueError(f"Dates illisibles dans {csv_path.name}")

    time_index = pd.DatetimeIndex(t)

    # L/h -> m3/s
    Q_m3s = Q_LH / 1000.0 / 3600.0

    # dt médian (robuste)
    if len(time_index) >= 3:
        dts = np.diff(time_index.values).astype("timedelta64[s]").astype(float)
        dt_sec = float(np.median(dts))
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            dt_sec = 120.0
    else:
        dt_sec = 120.0

    return time_index, P, Q_m3s, dt_sec


# ======================================================================
# DÉTECTION DÉCRUES PROPRES
# ======================================================================

def detect_clean_recessions_to_zero(
    time_index: pd.DatetimeIndex,
    P_mm: np.ndarray,
    Q_m3s: np.ndarray,
    dt_sec: float,
    rain_thresh_mm: float = RAIN_THRESH_MM,
    q_start_min_m3s: float = Q_START_MIN_M3S,
    q_zero_thresh_m3s: float = Q_ZERO_THRESH_M3S,
    min_len_pts_pos: int = MIN_LEN_PTS_POS,
    n_zero_end: int = N_ZERO_END,
    allow_small_bumps: bool = ALLOW_SMALL_BUMPS,
    bump_tol_frac: float = BUMP_TOL_FRAC,
) -> list[dict]:

    n = len(Q_m3s)
    if n < 5:
        return []

    is_dry = P_mm <= rain_thresh_mm
    recs: list[dict] = []
    i = 1  # évite bords

    def is_local_max(idx: int) -> bool:
        return (Q_m3s[idx] >= Q_m3s[idx - 1]) and (Q_m3s[idx] >= Q_m3s[idx + 1])

    while i < n - 1:
        if is_dry[i] and (Q_m3s[i] >= q_start_min_m3s) and is_local_max(i):
            start = i
            j = start + 1
            last_Q = Q_m3s[start]
            hit_zero_run = 0
            pos_count = 1

            while j < n and is_dry[j]:
                qj = Q_m3s[j]

                if qj > q_zero_thresh_m3s:
                    pos_count += 1
                    hit_zero_run = 0
                else:
                    hit_zero_run += 1

                # contrôle bumps
                if not allow_small_bumps:
                    if (qj > last_Q) and (qj > q_zero_thresh_m3s):
                        break
                else:
                    if (last_Q > q_zero_thresh_m3s) and (qj > q_zero_thresh_m3s):
                        if qj > last_Q * (1.0 + bump_tol_frac):
                            break

                last_Q = qj

                # fin à ~0 stable
                if hit_zero_run >= n_zero_end:
                    end = j
                    seg_P = P_mm[start:end + 1]
                    if np.any(seg_P > rain_thresh_mm):
                        break

                    if pos_count >= min_len_pts_pos:
                        seg_time = time_index[start:end + 1]
                        seg_Q = Q_m3s[start:end + 1]
                        name = f"rec_{seg_time[0].strftime('%Y%m%d_%H%M%S')}"

                        recs.append(
                            dict(
                                name=name,
                                start_idx=int(start),
                                end_idx=int(end),
                                time=seg_time,
                                P_mm=seg_P,
                                Q_m3s=seg_Q,
                                dt_sec=float(dt_sec),
                            )
                        )

                    i = end + 1
                    break

                j += 1

            # si pas de fin trouvée, avancer
            if j >= n or (i == start):
                i = start + 1
            continue

        i += 1

    return recs


def prepare_positive_recession_part(
    seg_Q: np.ndarray,
    seg_P: np.ndarray,
    q_zero_thresh_m3s: float,
    drop_first_n: int = DROP_FIRST_N,
    min_points: int = MIN_LEN_PTS_POS,
) -> np.ndarray | None:
    Q = np.asarray(seg_Q, float)
    P = np.asarray(seg_P, float)

    if np.any(P > 0.0):
        return None

    idx_pos = np.where(Q > q_zero_thresh_m3s)[0]
    if idx_pos.size == 0:
        return None

    Qpos = Q[idx_pos[0] : idx_pos[-1] + 1]

    if drop_first_n > 0 and len(Qpos) > drop_first_n:
        Qpos = Qpos[drop_first_n:]

    Qpos = Qpos[Qpos > q_zero_thresh_m3s]

    if len(Qpos) < min_points:
        return None

    return Qpos


def fit_single_segment(t_sec: np.ndarray, lnQ: np.ndarray) -> dict:
    b, a = np.polyfit(t_sec, lnQ, 1)
    pred = a + b * t_sec
    resid = lnQ - pred

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((lnQ - lnQ.mean()) ** 2))
    R2 = 1.0 - rss / tss if tss > 0 else np.nan
    k = -b

    return dict(slope=float(b), intercept=float(a), k=float(k), RSS=rss, R2=float(R2))


# ======================================================================
# UTILITAIRES
# ======================================================================

def list_event_csvs() -> list[Path]:
    csvs: list[Path] = []
    for y in YEARS:
        d = EVENTS_ROOT / str(y)
        if d.exists():
            csvs.extend(sorted(d.glob("*.csv")))
    return csvs


# ======================================================================
# MAIN
# ======================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    csv_list = list_event_csvs()
    print(f"[INFO] CSV évènements trouvés : {len(csv_list)}")
    if not csv_list:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {EVENTS_ROOT}/{{2022,2023,2024}}")

    rows = []

    for csv_path in csv_list:
        event_id = csv_path.stem  # ex: event_2022_001
        year = csv_path.parent.name

        try:
            time_index, P_mm, Q_m3s, dt_sec = read_event_csv(csv_path)
        except Exception as e:
            print(f"[WARN] skip {csv_path.name} (lecture): {e}")
            continue

        # détection décrues propres dans CET évènement
        recs = detect_clean_recessions_to_zero(
            time_index=time_index,
            P_mm=P_mm,
            Q_m3s=Q_m3s,
            dt_sec=dt_sec,
        )

        if not recs:
            continue

        for rec in recs:
            name = rec["name"]
            seg_time = rec["time"]
            seg_P = rec["P_mm"]
            seg_Q = rec["Q_m3s"]

            Qpos = prepare_positive_recession_part(seg_Q, seg_P, q_zero_thresh_m3s=Q_ZERO_THRESH_M3S)
            if Qpos is None:
                continue

            npos = len(Qpos)
            t_sec = np.arange(npos, dtype=float) * dt_sec
            t_h = t_sec / 3600.0
            lnQ = np.log(Qpos)

            # décroissance moyenne
            if (lnQ[0] - lnQ[-1]) <= 0:
                continue

            fit = fit_single_segment(t_sec, lnQ)
            if fit["k"] <= 0:
                continue

            Qmax_ls = float(np.max(Qpos) * 1000.0)  # m3/s -> L/s
            Qend_ls = float(np.min(Qpos) * 1000.0)

            rows.append(
                dict(
                    year=int(year) if str(year).isdigit() else year,
                    event_id=event_id,
                    rec_name=f"{event_id}__{name}",
                    start_time=seg_time[0],
                    end_time=seg_time[-1],
                    n_points_pos=npos,
                    dt_sec=dt_sec,
                    Q_max_Ls=Qmax_ls,
                    Q_end_pos_Ls=Qend_ls,
                    k_s_1=fit["k"],
                    R2=fit["R2"],
                    RSS=fit["RSS"],
                    rain_thresh_mm=RAIN_THRESH_MM,
                    q_zero_thresh_m3s=Q_ZERO_THRESH_M3S,
                    n_zero_end=N_ZERO_END,
                    bumps_allowed=ALLOW_SMALL_BUMPS,
                    bump_tol_frac=BUMP_TOL_FRAC,
                    source_csv=str(csv_path),
                )
            )

            # QA plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(t_h, lnQ, s=10, label="ln(Q) obs (Q>0)")

            pred = fit["intercept"] + fit["slope"] * t_sec
            ax.plot(
                t_h, pred, linestyle="--",
                label=f"fit: k={fit['k']:.2e} s⁻¹, R²={fit['R2']:.3f}"
            )

            ax.set_xlabel("Temps depuis début décrue (h)")
            ax.set_ylabel("ln(Q) (ln(m³/s))")
            ax.set_title(f"{event_id} | {name} | sec + fin~0")

            ax.grid(True, alpha=0.4)
            ax.legend(loc="best")

            fig.tight_layout()
            fig.savefig(QA_DIR / f"{event_id}__{name}_lnQ_onek_QA.png", dpi=PLOT_DPI)
            plt.close(fig)

    if not rows:
        print("[WARN] Aucun résultat exploitable avec ces critères stricts.")
        print("Pistes: augmenter POST_PAD_MIN dans ton export d'events, ou assouplir Q_ZERO_THRESH / N_ZERO_END / Q_START_MIN.")
        return

    df_out = pd.DataFrame(rows).sort_values(["year", "start_time"])
    out_xlsx = OUT_DIR / "recessions_onek_from_events.xlsx"
    df_out.to_excel(out_xlsx, index=False)

    print(f"[OK] Export Excel : {out_xlsx}")
    print(f"[OK] QA : {QA_DIR}")
    print("\n=== SYNTHÈSE k (ONE_K, décrues propres, par events) ===")
    print(f"N = {len(df_out)}")
    print(f"médian = {df_out['k_s_1'].median():.3e} s^-1")
    print(f"moyen  = {df_out['k_s_1'].mean():.3e} s^-1")
    print(f"min/max= {df_out['k_s_1'].min():.3e} / {df_out['k_s_1'].max():.3e} s^-1")


if __name__ == "__main__":
    main()
