# -*- coding: utf-8 -*-
"""
recessions_simple_regression_CSR_events_v2.py
---------------------------------------------
Parcourt les events exportés par hr_vs_hp.py :
  date ; P_mm ; Q_inf_LH ; Q_ruiss_LH

Détecte des décrues "propres" (sec + ln(Q) ~ linéaire) et fit k (s^-1) sur Q cible.

Sorties :
  03_Plots/Identification constantes vidange CSR/RECESSIONS_SIMPLE_EVENTS/QA/*.png
  03_Plots/Identification constantes vidange CSR/RECESSIONS_SIMPLE_EVENTS/recessions_simple_events.xlsx
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PARAMS
# =========================
# Cible
Q_TARGET_COL = "Q_inf_LH"   # ou "Q_ruiss_LH"

# Racine events (CHEZ TOI: 02_Data/all_events1/YYYY/event_*.csv)
EVENTS_ROOT = "02_Data/all_events1"
YEARS = [2022, 2023, 2024]

# Alias colonnes acceptés
DATE_ALIASES = ["date", "Date", "timestamp", "time"]
RAIN_ALIASES = ["P_mm", "Hauteur_de_pluie_mm", "P", "rain_mm"]

# Critères "sec" + qualité
rain_thr_mm = 0.1     # "sec" (mm / pas)
q_min_LH = 5.0         # ignore le bruit bas (L/h)
min_pts = 15           # longueur mini de régression
r2_min = 0.9          # qualité mini ln(Q)~linéaire

drop_first = 1         # enlève 1 point juste après le pic
max_hours = 12         # longueur max décrue

# Filtre robustesse event (optionnel)
FILTER_ROBUST_EVENTS = True
robust_min_duration_min = 10
robust_require_rain_after_peak_dry = False
robust_tail_zero_pts = 3
robust_q_zero_LH = 4

MAX_QA_PLOTS = 2000


# =========================
# Helpers colonnes
# =========================
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = list(df.columns)
    for c in candidates:
        if c in df.columns:
            return c
    low_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    raise ValueError(f"Aucune des colonnes {candidates} trouvée. Colonnes dispo: {cols}")


def pick_q_col(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    low_map = {c.lower(): c for c in df.columns}
    if target.lower() in low_map:
        return low_map[target.lower()]
    raise ValueError(f"Colonne Q cible '{target}' introuvable. Colonnes dispo: {list(df.columns)}")


# =========================
# IO events
# =========================
def iter_event_files(events_root: Path, years: list[int]):
    for y in years:
        d = events_root / str(y)
        if not d.exists():
            continue
        for p in sorted(d.glob("event_*.csv")):
            if p.is_file():
                yield p
def read_event(csv_path: Path):
    df = pd.read_csv(csv_path, sep=";", na_values=["NA", "NaN", "", -9999, -9999.0])

    date_col = pick_col(df, DATE_ALIASES)
    rain_col = pick_col(df, RAIN_ALIASES)
    q_col = pick_q_col(df, Q_TARGET_COL)

    # --- dates (utilise bien date_col, pas df["date"])
    t = pd.to_datetime(df[date_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if t.isna().any():
        t = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    if t.isna().any():
        bad = df[t.isna()].head(3)
        raise ValueError(f"Dates illisibles dans {csv_path.name}. Exemples:\n{bad}")

    P = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    Q = pd.to_numeric(df[q_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # --- dt médian robuste (t est une Series -> pas de to_series())
    if len(t) >= 3:
        dt = t.diff().dt.total_seconds().dropna().to_numpy()
        dt_sec = float(np.median(dt)) if len(dt) else 120.0
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            dt_sec = 120.0
    else:
        dt_sec = 120.0

    return t, P, Q, dt_sec, date_col, rain_col, q_col

# =========================
# Détection simple décrues
# =========================
def is_local_max(Q, i):
    return (Q[i] >= Q[i - 1]) and (Q[i] >= Q[i + 1])


def extract_recession_from_peak(t, P, Q, dt_sec, i_peak):
    n = len(Q)
    i0 = i_peak
    i = i_peak
    max_pts = int((max_hours * 3600) / dt_sec)

    while i + 1 < n and (i - i0) < max_pts:
        if P[i + 1] > rain_thr_mm:
            break
        if Q[i + 1] < q_min_LH:
            break
        i += 1

    i1 = i
    if (i1 - i0 + 1) < (min_pts + drop_first):
        return None

    tt = t[i0 : i1 + 1]
    PP = P[i0 : i1 + 1]
    QQ = Q[i0 : i1 + 1]

    if np.any(PP > rain_thr_mm):
        return None

    QQ2 = QQ[drop_first:]
    if QQ2.size < min_pts:
        return None
    if np.any(QQ2 <= 0):
        return None

    lnQ = np.log(QQ2)
    if (lnQ[0] - lnQ[-1]) <= 0:
        return None

    t_sec = np.arange(QQ2.size, dtype=float) * dt_sec
    b, a = np.polyfit(t_sec, lnQ, 1)
    pred = a + b * t_sec
    resid = lnQ - pred

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((lnQ - lnQ.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    k = -b  # 1/s

    if not np.isfinite(r2) or r2 < r2_min or k <= 0:
        return None

    info = {
        "start_idx": int(i0),
        "end_idx": int(i1),
        "start_time": tt.iloc[0],
        "end_time": tt.iloc[-1],
        "n_fit": int(QQ2.size),
        "k_s_1": float(k),              # <- clé python-safe
        "R2": float(r2),
        "Qmax_LH": float(np.max(QQ)),
        "Qend_LH": float(np.min(QQ2)),
        "ln_drop": float(lnQ[0] - lnQ[-1]),
        "a": float(a),
        "b": float(b),
    }
    return info, (t_sec / 3600.0, lnQ, pred)


# =========================
# Filtre robustesse event
# =========================
def is_robust_event(t, P, Q, dt_sec):
    dur_min = (len(t) - 1) * dt_sec / 60.0
    if dur_min < robust_min_duration_min:
        return False, "duration"

    if np.nanmax(Q) < q_min_LH:
        return False, "too_low"

    i_peak = int(np.nanargmax(Q))
    if i_peak >= len(Q) - 2:
        return False, "peak_at_end"

    if robust_require_rain_after_peak_dry:
        if np.any(P[i_peak + 1 :] > rain_thr_mm):
            return False, "rain_after_peak"

    if len(Q) >= robust_tail_zero_pts:
        tail = Q[-robust_tail_zero_pts:]
        if not np.all(tail <= robust_q_zero_LH):
            return False, "no_zero_end"
    else:
        return False, "too_short_for_tail"

    return True, "ok"


# =========================
# MAIN
# =========================
def main():
    base_dir = Path(__file__).resolve().parent
    events_root = base_dir.parent / Path(EVENTS_ROOT)

    if not events_root.exists():
        raise FileNotFoundError(f"Racine évènements introuvable : {events_root}")

    out_dir = base_dir.parent / "03_Plots" / "Identification constantes vidange CSR" / "k_seepage"
    qa_dir = out_dir / "QA"
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    event_files = list(iter_event_files(events_root, YEARS))
    print(f"[INFO] Events trouvés : {len(event_files)} dans {events_root}")
    print(f"[INFO] Cible Q = '{Q_TARGET_COL}' | filtre robustesse = {FILTER_ROBUST_EVENTS}")

    rows = []
    qa_count = 0
    skipped = {"read_error": 0, "not_robust": 0, "no_segment": 0}

    for csv_path in event_files:
        try:
            t, P, Q_LH, dt_sec, date_col, rain_col, q_col = read_event(csv_path)
        except Exception as e:
            skipped["read_error"] += 1
            print(f"[WARN] Lecture impossible {csv_path.name}: {e}")
            continue

        if FILTER_ROBUST_EVENTS:
            ok, _reason = is_robust_event(t, P, Q_LH, dt_sec)
            if not ok:
                skipped["not_robust"] += 1
                continue

        dry = P <= rain_thr_mm
        active = Q_LH >= q_min_LH

        found_any = False
        i = 1
        n = len(Q_LH)

        while i < n - 1:
            if dry[i] and active[i] and is_local_max(Q_LH, i):
                res = extract_recession_from_peak(t, P, Q_LH, dt_sec, i)
                if res is not None:
                    info, (th, lnQ, pred) = res

                    event_id = csv_path.stem
                    name = f"{event_id}__rec_{info['start_time'].strftime('%Y%m%d_%H%M')}"
                    info["name"] = name
                    info["event_file"] = str(csv_path)
                    info["event_id"] = event_id
                    info["Q_col_used"] = q_col
                    info["P_col_used"] = rain_col
                    info["date_col_used"] = date_col
                    info["dt_sec"] = float(dt_sec)

                    rows.append(info)
                    found_any = True

                    if qa_count < MAX_QA_PLOTS:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.scatter(th, lnQ, s=10, label="ln(Q) obs")
                        ax.plot(
                            th, pred, linestyle="--",
                            label=f"k={info['k_s_1']:.2e} s⁻¹ | R²={info['R2']:.3f}"
                        )
                        ax.set_xlabel("Temps depuis pic (h)")
                        ax.set_ylabel("ln(Q) (ln(L/h))")
                        ax.set_title(name)
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc="best", fontsize=8)
                        fig.tight_layout()
                        fig.savefig(qa_dir / f"{name}_QA.png", dpi=200)
                        plt.close(fig)
                        qa_count += 1

                    i = info["end_idx"] + 1
                    continue

            i += 1

        if not found_any:
            skipped["no_segment"] += 1

    if not rows:
        print("[WARN] Aucun segment retenu.")
        print("  -> Assouplis r2_min / q_min_LH / rain_thr_mm, ou mets FILTER_ROBUST_EVENTS=False.")
        print(f"[INFO] Skips: {skipped}")
        return

    df = pd.DataFrame(rows).sort_values(["event_id", "start_time"])
    out_xlsx = out_dir / "recessions_simple_events.xlsx"
    df.to_excel(out_xlsx, index=False)

    print(f"[OK] N segments = {len(df)}")
    print(f"[OK] Excel : {out_xlsx}")
    print(f"[OK] QA : {qa_dir} (plots={qa_count})")
    print(f"[INFO] Skips: {skipped}")
    print("\nRésumé k (s^-1):")
    print(df["k_s_1"].describe(percentiles=[0.1, 0.5, 0.9]))


if __name__ == "__main__":
    main()
