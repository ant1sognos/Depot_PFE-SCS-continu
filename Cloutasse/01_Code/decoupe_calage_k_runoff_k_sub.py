# -*- coding: utf-8 -*-
"""
Identification des constantes de vidange sur décrues - Bassin du Cloutasse

Détecte automatiquement les segments de décrue dans une série pluie-débit continue
et estime les constantes de vidange k (s⁻¹) via régression linéaire sur ln(Q) = a + b*t.
Teste systématiquement un modèle à 2 segments (rupture de pente) lorsque pertinent.

Entrées :
    - 02_Data/PQ_BV_Cloutasse_interp.csv (dateP, P_mm, Q_ls)

Sorties :
    - CSV complets : recessions_all_{MODE}.csv, recessions_two_slopes_{MODE}.csv
    - Graphiques Top 10 : Top10_QA/rank{XX}_{name}.png
    - Config utilisée : config_used_{MODE}.json

Modes disponibles :
    - strict    : Critères rigides, R² élevé, peu de segments retenus (qualité maximale)
    - loose     : Critères assouplis, acceptable pour bassins naturels avec bruit
    - loose_two : Optimisé détection 2 pentes, tolérant sur BIC, lissage actif
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Optimisation rendu matplotlib pour grandes séries
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0


# =========================================================
# CONFIGURATION MODES
# =========================================================
MODE = "loose_two"  
N_TOP_PLOTS = 10

CONFIGS: Dict[str, Dict] = {
    # Mode strict : qualité maximale, peu de segments
    # Usage : bassins bien instrumentés, décrues propres, besoin de robustesse
    "strict": dict(
        rain_thresh_detect_mm=0.2,    # Seuil pluie détection (mm)
        min_len_pts_detect=8,          # Points minimum pour détecter segment
        q_min_m3s=0.001,               # Débit minimum (m³/s)
        rain_keep_mm=0.0,              # Pluie max tolérée dans décrue (mm)
        min_points_clean=10,           # Points minimum après nettoyage
        drop_first_n=1,                # Supprimer N premiers points post-pic
        min_seg_points=8,              # Points min par segment (2-pentes)
        delta_BIC_min=2.0,             # BIC doit s'améliorer d'au moins 2
        allow_BIC_worse_up_to=0.0,     # BIC ne peut pas empirer
        min_delta_R2=0.00,             # Gain R² minimum
        min_R2_two=0.80,               # R² global min (2-pentes)
        min_R2_seg=0.80,               # R² min par segment
        min_delta_k_rel=0.30,          # Contraste k relatif min (30%)
        enforce_order=True,            # k1 >= k2 obligatoire
        enforce_ratio=True,            # k1/k2 >= min_ratio_k
        min_ratio_k=1.30,              # Ratio k minimum
        smooth_lnQ=False,              # Pas de lissage
        smooth_window=1,
        drop_nonmonotone=False,        # Garder points non monotones
        monotone_tol_rel=0.02,
        remove_baseflow=False,         # Pas de retrait débit de base
        baseflow_quantile=0.10,
        baseflow_min_pos_q=0.05,
    ),
    
    # Mode loose : compromis qualité/quantité
    # Usage : bassins naturels standard, bruit modéré
    "loose": dict(
        rain_thresh_detect_mm=0.2,
        min_len_pts_detect=6,
        q_min_m3s=0.001,
        rain_keep_mm=0.15,
        min_points_clean=8,
        drop_first_n=1,
        min_seg_points=4,
        delta_BIC_min=0.0,
        allow_BIC_worse_up_to=2.0,     # Tolère BIC légèrement pire
        min_delta_R2=0.02,
        min_R2_two=0.55,
        min_R2_seg=0.55,
        min_delta_k_rel=0.18,
        enforce_order=True,
        enforce_ratio=False,
        min_ratio_k=1.0,
        smooth_lnQ=True,               # Lissage actif
        smooth_window=3,
        drop_nonmonotone=False,
        monotone_tol_rel=0.03,
        remove_baseflow=False,
        baseflow_quantile=0.10,
        baseflow_min_pos_q=0.05,
    ),
    
    # Mode loose_two : maximise détection 2 pentes
    # Usage : recherche signatures multi-échelles, bassins à mémoire
    "loose_two": dict(
        rain_thresh_detect_mm=0.2,
        min_len_pts_detect=6,
        q_min_m3s=0.0007,              # Débit min très bas
        rain_keep_mm=0.20,
        min_points_clean=10,
        drop_first_n=1,
        min_seg_points=5,
        delta_BIC_min=-2.0,            # BIC peut empirer de 2
        allow_BIC_worse_up_to=6.0,
        min_delta_R2=0.00,
        min_R2_two=0.35,               # Seuil R² bas
        min_R2_seg=0.50,
        min_delta_k_rel=0.12,          # Faible contraste k accepté
        enforce_order=False,           # Pas d'ordre imposé
        enforce_ratio=False,
        min_ratio_k=1.0,
        smooth_lnQ=True,
        smooth_window=5,               # Lissage fort
        drop_nonmonotone=True,         # Supprimer remontées
        monotone_tol_rel=0.03,
        remove_baseflow=True,          # Retrait débit de base actif
        baseflow_quantile=0.10,
        baseflow_min_pos_q=0.05,
    ),
}

if MODE not in CONFIGS:
    raise ValueError(f"MODE invalide: {MODE}. Disponibles: {list(CONFIGS.keys())}")

CFG = CONFIGS[MODE]


# =========================================================
# LECTURE DONNÉES
# =========================================================
def read_full_series() -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, float]:
    """Charge série pluie-débit Cloutasse depuis CSV."""
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir.parent / "02_Data" / "PQ_BV_Cloutasse_interp.csv"

    df = pd.read_csv(csv_path, sep=";", na_values=["NA", "NaN", "", -9999, -9999.0])
    
    required_cols = ["dateP", "P_mm", "Q_ls"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path.name}: {missing}")

    t = pd.DatetimeIndex(pd.to_datetime(df["dateP"]))
    P = df["P_mm"].astype(float).fillna(0.0).to_numpy()
    Q = df["Q_ls"].astype(float).to_numpy() / 1000.0  # L/s → m³/s

    dt = (t[1] - t[0]).total_seconds() if len(t) >= 2 else 300.0
    return t, P, Q, float(dt)


# =========================================================
# DÉTECTION SEGMENTS CANDIDATS
# =========================================================
def detect_recessions(time_index, P, Q, *, rain_thresh_mm, min_len_pts, q_min_m3s) -> List[dict]:
    """
    Identifie segments de décrue candidats.
    
    Logique : période sèche + débit suffisant → découpe bloc → isole décrue post-pic
    """
    mask = (P <= rain_thresh_mm) & (Q >= q_min_m3s)
    n = len(Q)
    out: List[dict] = []
    i = 0
    
    while i < n:
        if not mask[i]:
            i += 1
            continue
            
        # Prolonge bloc tant que critères satisfaits
        j = i + 1
        while j < n and mask[j]:
            j += 1
        start_block, end_block = i, j - 1

        # Vérifie longueur suffisante
        if end_block - start_block + 1 >= min_len_pts:
            # Isole décrue depuis pic local
            idx_max = start_block + int(np.argmax(Q[start_block:end_block + 1]))
            start, end = idx_max, end_block
            
            if end - start + 1 >= min_len_pts:
                seg_time = time_index[start:end + 1]
                out.append({
                    "name": f"rec_{seg_time[0].strftime('%Y%m%d_%H%M')}",
                    "time": seg_time,
                    "P_mm": P[start:end + 1],
                    "Q_m3s": Q[start:end + 1],
                })
        i = j
        
    return out


def clean_segment(time_seg, Q, P, *, rain_keep_mm, min_points, drop_first_n, 
                  drop_nonmonotone, monotone_tol_rel):
    """
    Filtre segment : pluie résiduelle, premiers points, monotonie.
    Retourne (t_sec, Q_clean) ou None si insuffisant.
    """
    Q = np.asarray(Q, float)
    P = np.asarray(P, float)
    
    if len(Q) < min_points:
        return None

    # Filtre pluie + débit positif
    mask = (P <= float(rain_keep_mm)) & (Q > 0.0)
    if drop_first_n > 0:
        mask[:drop_first_n] = False
        
    if int(np.sum(mask)) < min_points:
        return None

    # Conversion temps relatif
    t0 = time_seg[mask][0]
    t_sec = (time_seg[mask] - t0).total_seconds().astype(float)
    Qc = Q[mask]
    
    # Vérifie pas de temps strictement croissant
    if len(t_sec) >= 2 and np.any(np.diff(t_sec) <= 0):
        return None

    # Option : supprime remontées non physiques
    if drop_nonmonotone:
        tol = float(monotone_tol_rel)
        keep = np.ones(len(Qc), dtype=bool)
        for k in range(1, len(Qc)):
            if Qc[k] > Qc[k - 1] * (1.0 + tol):
                keep[k] = False
        if int(np.sum(keep)) < min_points:
            return None
        t_sec, Qc = t_sec[keep], Qc[keep]

    return t_sec, Qc


# =========================================================
# AJUSTEMENTS LINÉAIRES
# =========================================================
def smooth_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Lissage moyenne mobile simple."""
    x = np.asarray(x, float)
    if window is None or window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def fit_single(t_sec, lnQ) -> dict:
    """Régression linéaire simple : ln(Q) = a + b*t, k = -b."""
    b, a = np.polyfit(t_sec, lnQ, 1)
    pred = a + b * t_sec
    resid = lnQ - pred
    rss = float(np.sum(resid**2))
    tss = float(np.sum((lnQ - lnQ.mean())**2))
    R2 = 1.0 - rss / tss if tss > 0 else np.nan
    
    return {
        "slope": float(b), 
        "intercept": float(a), 
        "k": float(-b), 
        "RSS": rss, 
        "R2": float(R2)
    }


def bic_from_rss(rss: float, n: int, p: int) -> float:
    """Critère d'information bayésien : BIC = n*ln(RSS/n) + p*ln(n)."""
    n_ = float(n)
    return float(n_ * np.log(rss / n_) + p * np.log(n_))


def fit_two(t_sec, lnQ, *, min_seg_points: int) -> Optional[dict]:
    """
    Ajustement 2 segments avec point de rupture automatique.
    Recherche exhaustive du meilleur breakpoint minimisant RSS total.
    """
    n = len(t_sec)
    if n < 2 * min_seg_points + 1:
        return None

    best = None
    rss_best = np.inf
    tss = float(np.sum((lnQ - lnQ.mean())**2))
    p_two = 4  # 4 paramètres : (a1, b1, a2, b2)

    for k_break in range(min_seg_points, n - min_seg_points):
        t_b = float(t_sec[k_break])

        # Segment 1 : [0, k_break]
        t1, lnQ1 = t_sec[:k_break + 1], lnQ[:k_break + 1]
        b1, a1 = np.polyfit(t1, lnQ1, 1)
        if b1 >= 0:  # Pente doit être négative (décrue)
            continue
        lnQ_b = a1 + b1 * t_b

        # Segment 2 : [k_break+1, end]
        t2, lnQ2 = t_sec[k_break + 1:], lnQ[k_break + 1:]
        if len(t2) < min_seg_points:
            continue
        b2, _ = np.polyfit(t2, lnQ2, 1)
        if b2 >= 0:
            continue
        a2 = lnQ_b - b2 * t_b  # Continuité imposée

        # Calcul erreurs
        pred1 = a1 + b1 * t1
        pred2 = a2 + b2 * t2
        rss1 = float(np.sum((lnQ1 - pred1)**2))
        rss2 = float(np.sum((lnQ2 - pred2)**2))
        rss_tot = rss1 + rss2

        if rss_tot < rss_best:
            rss_best = rss_tot
            R2_two = 1.0 - rss_tot / tss if tss > 0 else np.nan
            bic_two = float(n) * np.log(rss_tot / float(n)) + p_two * np.log(float(n))

            tss1 = float(np.sum((lnQ1 - lnQ1.mean())**2))
            tss2 = float(np.sum((lnQ2 - lnQ2.mean())**2))
            R2_1 = 1.0 - rss1 / tss1 if tss1 > 0 else np.nan
            R2_2 = 1.0 - rss2 / tss2 if tss2 > 0 else np.nan

            best = {
                "k_break_idx": int(k_break),
                "t_break_sec": float(t_b),
                "slope1": float(b1), "intercept1": float(a1), 
                "k1": float(-b1), "R2_1": float(R2_1),
                "slope2": float(b2), "intercept2": float(a2), 
                "k2": float(-b2), "R2_2": float(R2_2),
                "R2_two": float(R2_two), 
                "BIC_two": float(bic_two),
            }
            
    return best


def compute_lnQ(t_sec, Q_clean, cfg) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calcule ln(Q) avec option retrait débit de base.
    Retourne (t_used, lnQ, Qb_used).
    """
    Qb = np.nan
    t_used, Q_used = t_sec, Q_clean

    if bool(cfg.get("remove_baseflow", False)):
        q_b = float(np.quantile(Q_clean, float(cfg.get("baseflow_quantile", 0.10))))
        Q_eff = Q_clean - q_b
        Qb = q_b

        if np.any(Q_eff > 0):
            pos = Q_eff[Q_eff > 0]
            q_min_pos = float(np.quantile(pos, float(cfg.get("baseflow_min_pos_q", 0.05))))
            ok = Q_eff > q_min_pos
            if int(np.sum(ok)) >= int(cfg.get("min_points_clean", 8)):
                t_used, Q_used = t_sec[ok], Q_eff[ok]

    return t_used, np.log(Q_used), float(Qb)


def accept_two(single: dict, two: dict, n: int, cfg: Dict) -> Tuple[bool, dict]:
    """
    Critères acceptation modèle 2 segments vs simple.
    
    Vérifie : k positifs, R² suffisants, contraste k, ordre k, critères BIC/R².
    Retourne (use_two: bool, diagnostics: dict).
    """
    BIC_single = bic_from_rss(float(single["RSS"]), n, p=2)

    k1, k2 = float(two["k1"]), float(two["k2"])
    R2_1, R2_2 = float(two["R2_1"]), float(two["R2_2"])
    R2_two, BIC_two = float(two["R2_two"]), float(two["BIC_two"])

    delta_BIC = BIC_two - BIC_single
    delta_R2 = R2_two - float(single["R2"])
    delta_k_rel = (abs(k1 - k2) / max(k1, k2)) if (k1 > 0 and k2 > 0) else np.nan

    # Conditions cumulatives
    cond_positive = (k1 > 0) and (k2 > 0)
    cond_R2_seg = (R2_1 >= float(cfg["min_R2_seg"])) and (R2_2 >= float(cfg["min_R2_seg"]))
    cond_R2_global = (R2_two >= float(cfg["min_R2_two"]))
    cond_R2_floor = cond_R2_seg and cond_R2_global

    cond_contrast = True
    if not np.isnan(delta_k_rel):
        cond_contrast = delta_k_rel >= float(cfg["min_delta_k_rel"])

    cond_order = True
    if bool(cfg.get("enforce_order", False)):
        cond_order = k1 >= k2

    cond_ratio = True
    if bool(cfg.get("enforce_ratio", False)) and (k1 > 0 and k2 > 0):
        cond_ratio = (k1 / k2) >= float(cfg.get("min_ratio_k", 1.0))

    # Critères BIC/R² : au moins un doit être satisfait
    cond_BIC_good = BIC_two < (BIC_single - float(cfg["delta_BIC_min"]))
    cond_BIC_ok = BIC_two <= (BIC_single + float(cfg["allow_BIC_worse_up_to"]))
    cond_R2_gain = delta_R2 >= float(cfg["min_delta_R2"])
    cond_choice = cond_BIC_good or cond_BIC_ok or cond_R2_gain

    use_two = (cond_positive and cond_R2_floor and cond_contrast and 
               cond_order and cond_ratio and cond_choice)

    diag = {
        "BIC_single": float(BIC_single),
        "BIC_two": float(BIC_two),
        "delta_BIC": float(delta_BIC),
        "R2_single": float(single["R2"]),
        "R2_two": float(R2_two),
        "R2_1": float(R2_1),
        "R2_2": float(R2_2),
        "delta_R2": float(delta_R2),
        "k1": float(k1),
        "k2": float(k2),
        "delta_k_rel": float(delta_k_rel) if delta_k_rel == delta_k_rel else np.nan,
        "t_break_sec": float(two["t_break_sec"]),
        "accepted_two": bool(use_two),
    }
    return use_two, diag


# =========================================================
# VISUALISATION
# =========================================================
def plot_recession(t_sec, lnQ, single, two, name, model_type, out_path, cfg):
    """Génère graphique décrue : données + ajustements simple/double."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.scatter(t_sec / 3600.0, lnQ, s=15, alpha=0.6, label="ln(Q)", color='navy')
    ax.plot(t_sec / 3600.0, single["intercept"] + single["slope"] * t_sec, 
            linestyle="--", linewidth=2, color='orange',
            label=f"Simple: k={single['k']:.2e} s^-1 (R2={single['R2']:.3f})")
    
    if model_type == "two" and two is not None:
        kb = two["k_break_idx"]
        t1, t2 = t_sec[:kb + 1], t_sec[kb + 1:]
        ax.plot(t1 / 3600.0, two["intercept1"] + two["slope1"] * t1, 
                linewidth=2, color='red',
                label=f"Seg1: k1={two['k1']:.2e} s^-1 (R2={two['R2_1']:.3f})")
        ax.plot(t2 / 3600.0, two["intercept2"] + two["slope2"] * t2, 
                linewidth=2, color='darkgreen',
                label=f"Seg2: k2={two['k2']:.2e} s^-1 (R2={two['R2_2']:.3f})")
        ax.axvline(two["t_break_sec"] / 3600.0, linestyle=":", 
                   color='black', alpha=0.7, label="Rupture")
    
    ax.set_xlabel("Temps (h)", fontsize=11)
    ylabel = "ln(Q - Qb)" if bool(cfg.get("remove_baseflow", False)) else "ln(Q)"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{name} | Modele: {model_type} | MODE={MODE}", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# TRAITEMENT PRINCIPAL
# =========================================================
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir.parent / "03_Plots" / "Identification constantes vidange" 
    top10_dir = out_dir / "Top10_QA"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    top10_dir.mkdir(parents=True, exist_ok=True)

    # Trace config utilisée
    (out_dir / f"config_used_{MODE}.json").write_text(
        json.dumps(CFG, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Chargement données
    t, P, Q, dt_nom = read_full_series()

    # Détection segments candidats
    recessions = detect_recessions(
        t, P, Q,
        rain_thresh_mm=float(CFG["rain_thresh_detect_mm"]),
        min_len_pts=int(CFG["min_len_pts_detect"]),
        q_min_m3s=float(CFG["q_min_m3s"]),
    )

    print(f"\n{'='*70}")
    print(f"ANALYSE CONSTANTES VIDANGE - MODE: {MODE}")
    print(f"{'='*70}")

    rows_all: List[dict] = []
    rows_two_slopes: List[dict] = []
    plot_data: List[dict] = []

    for rec in recessions:
        name = rec["name"]
        seg_time, seg_P, seg_Q = rec["time"], rec["P_mm"], rec["Q_m3s"]

        # Nettoyage segment
        cleaned = clean_segment(
            seg_time, seg_Q, seg_P,
            rain_keep_mm=float(CFG["rain_keep_mm"]),
            min_points=int(CFG["min_points_clean"]),
            drop_first_n=int(CFG["drop_first_n"]),
            drop_nonmonotone=bool(CFG.get("drop_nonmonotone", False)),
            monotone_tol_rel=float(CFG.get("monotone_tol_rel", 0.03)),
        )
        if cleaned is None:
            continue
        
        t_sec_raw, Q_clean_raw = cleaned

        # Calcul ln(Q)
        t_sec, lnQ, Qb_used = compute_lnQ(t_sec_raw, Q_clean_raw, CFG)
        if bool(CFG["smooth_lnQ"]):
            lnQ = smooth_moving_average(lnQ, int(CFG["smooth_window"]))

        if float(lnQ[0] - lnQ[-1]) <= 0:
            continue

        # Ajustement simple
        single = fit_single(t_sec, lnQ)
        if float(single["k"]) <= 0:
            continue

        # Initialisation résultats
        best_model = "single"
        k_unique, k_fast, k_slow = float(single["k"]), np.nan, np.nan
        two = None

        diag = {
            "R2_single": float(single["R2"]),
            "BIC_single": float(bic_from_rss(float(single["RSS"]), len(t_sec), 2)),
            "R2_two": np.nan, "BIC_two": np.nan, "delta_BIC": np.nan,
            "R2_1": np.nan, "R2_2": np.nan, "delta_R2": np.nan,
            "delta_k_rel": np.nan, "t_break_sec": np.nan,
        }

        # Tentative ajustement 2 segments
        two = fit_two(t_sec, lnQ, min_seg_points=int(CFG["min_seg_points"]))
        
        if two is not None:
            use_two, diag2 = accept_two(single, two, len(t_sec), CFG)

            if use_two:
                best_model = "two"
                k_fast = float(max(two["k1"], two["k2"]))
                k_slow = float(min(two["k1"], two["k2"]))
                k_unique = np.nan
                
                diag.update({
                    "R2_two": float(diag2["R2_two"]),
                    "BIC_two": float(diag2["BIC_two"]),
                    "delta_BIC": float(diag2["delta_BIC"]),
                    "R2_1": float(diag2["R2_1"]),
                    "R2_2": float(diag2["R2_2"]),
                    "delta_R2": float(diag2["delta_R2"]),
                    "delta_k_rel": float(diag2["delta_k_rel"]) if diag2["delta_k_rel"] == diag2["delta_k_rel"] else np.nan,
                    "t_break_sec": float(diag2["t_break_sec"]),
                })

                # Stockage dédié décrues 2 pentes
                rows_two_slopes.append({
                    "name": name,
                    "n_points": int(len(t_sec)),
                    "duration_h": float(t_sec[-1] / 3600.0),
                    "Q_max_ls": float(np.max(Q_clean_raw) * 1000.0),
                    "k1_s-1": float(two["k1"]),
                    "k2_s-1": float(two["k2"]),
                    "t_break_h": float(two["t_break_sec"] / 3600.0),
                    "R2_1": float(two["R2_1"]),
                    "R2_2": float(two["R2_2"]),
                    "R2_two": float(diag2["R2_two"]),
                    "delta_BIC": float(diag2["delta_BIC"]),
                })

        # Stockage tous résultats
        rows_all.append({
            "name": name,
            "n_points": int(len(t_sec)),
            "dt_sec_nominal": float(dt_nom),
            "duration_h": float(t_sec[-1] / 3600.0) if len(t_sec) >= 2 else 0.0,
            "Q_max_ls": float(np.max(Q_clean_raw) * 1000.0),
            "Qb_m3s_used": float(Qb_used) if Qb_used == Qb_used else np.nan,
            "model": best_model,
            "k_unique_s-1": k_unique,
            "k_fast_s-1": k_fast,
            "k_slow_s-1": k_slow,
            "t_break_h": float(diag["t_break_sec"] / 3600.0) if diag["t_break_sec"] == diag["t_break_sec"] else np.nan,
            "R2_single": float(diag["R2_single"]),
            "R2_two": diag["R2_two"],
            "R2_1": diag["R2_1"],
            "R2_2": diag["R2_2"],
            "BIC_single": float(diag["BIC_single"]),
            "BIC_two": diag["BIC_two"],
            "delta_BIC": diag["delta_BIC"],
        })

        # Stockage pour sélection Top 10
        quality_score = float(diag2["R2_two"]) if best_model == "two" else float(single["R2"])
        plot_data.append({
            "name": name,
            "quality": quality_score,
            "t_sec": t_sec,
            "lnQ": lnQ,
            "single": single,
            "two": two,
            "model_type": best_model,
        })

    # =========================================================
    # SAUVEGARDE RÉSULTATS
    # =========================================================
    if not rows_all:
        print("[WARN] Aucun segment exploitable.")
        return

    df_all = pd.DataFrame(rows_all)
    df_two_slopes = pd.DataFrame(rows_two_slopes) if rows_two_slopes else pd.DataFrame()

    # CSV complet
    out_all = out_dir / f"recessions_all_{MODE}.csv"
    df_all.to_csv(out_all, index=False, sep=';')

    # CSV décrues 2 pentes
    if len(df_two_slopes) > 0:
        out_two = out_dir / f"recessions_two_slopes_{MODE}.csv"
        df_two_slopes.to_csv(out_two, index=False, sep=';')

    # =========================================================
    # AFFICHAGE DÉCRUES 2 PENTES
    # =========================================================
    n_two = len(df_two_slopes)
    if n_two > 0:
        print(f"\nDECRUES A 2 PENTES DETECTEES: {n_two}")
        print(f"{'-'*70}")
        for idx, row in df_two_slopes.iterrows():
            print(f"  [{idx+1}] {row['name']}")
            print(f"      k1 = {row['k1_s-1']:.2e} s^-1  |  k2 = {row['k2_s-1']:.2e} s^-1")
            print(f"      Rupture: t = {row['t_break_h']:.2f} h")
            print(f"      R2 global = {row['R2_two']:.3f}  |  delta_BIC = {row['delta_BIC']:.1f}")
            print()
    else:
        print("\n  Aucune decrue a 2 pentes retenue.")

    # =========================================================
    # GRAPHIQUES TOP 10
    # =========================================================
    print(f"\nGeneration plots Top {N_TOP_PLOTS}...")
    
    plot_data_sorted = sorted(plot_data, key=lambda x: x["quality"], reverse=True)
    top_n = plot_data_sorted[:N_TOP_PLOTS]

    for rank, item in enumerate(top_n, start=1):
        out_path = top10_dir / f"rank{rank:02d}_{item['name']}.png"
        plot_recession(
            t_sec=item["t_sec"],
            lnQ=item["lnQ"],
            single=item["single"],
            two=item["two"],
            name=item["name"],
            model_type=item["model_type"],
            out_path=out_path,
            cfg=CFG
        )

    # =========================================================
    # RÉSUMÉ
    # =========================================================
    print(f"\n{'='*70}")
    print(f"TRAITEMENT TERMINE")
    print(f"{'='*70}")
    print(f"  Segments analyses:        {len(df_all)}")
    print(f"  Decrues 1 pente:          {len(df_all) - n_two}")
    print(f"  Decrues 2 pentes:         {n_two}")
    print(f"\nFichiers generes:")
    print(f"  - {out_all.name}")
    if n_two > 0:
        print(f"  - {out_two.name}")
    print(f"  - Top {N_TOP_PLOTS} plots: {top10_dir.name}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()