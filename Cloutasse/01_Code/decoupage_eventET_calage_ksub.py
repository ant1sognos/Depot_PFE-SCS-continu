# -*- coding: utf-8 -*-

"""

Ce script identifie automatiquement des segments de décrue sur la chronique P–Q du BV de la Cloutasse, puis estime
la(les) constante(s) de vidange k (s⁻¹) en ajustant une loi exponentielle sur le débit observé :
    Q(t) ≈ Q0 * exp(-k t)  <=>  ln(Q(t)) = a + b t, avec k = -b.
Un modèle à 2 segments (double pente sur ln(Q)) peut être retenu pour détecter des transitions de régime
(décrue rapide puis lente) lorsque les critères de qualité et de cohérence sont satisfaits.

Entrées :
- 02_Data/PQ_BV_Cloutasse_interp.csv (sep=';')
  Colonnes attendues :
    - dateP  : timestamp
    - P_mm   : pluie (mm)
    - Q_ls   : débit (L/s)

Sorties :
Dans 03_Plots/Identification constantes vidange/filet_{MODE}/ :
- recessions_and_k_PRO_{MODE}.xlsx
    Tableau global des décrues retenues avec k (1 segment) ou (k_fast, k_slow) (2 segments),
    indicateurs d'ajustement (R², BIC, etc.), durée, Qmax, éventuel Qb (si retrait baseflow).
- two_segment_candidates_{MODE}.xlsx
    Détail de tous les fits 2 segments tentés (acceptés ou rejetés) + diagnostics de rejet.
- Recessions_QA/*.png
    QA par décrue : ln(Q) (après nettoyage) + fit 1 segment + fit 2 segments si accepté + rupture.

Principe de traitement :
1) Lecture et mise en forme
   - Conversion Q : L/s -> m³/s.
   - Construction d'un index temporel, estimation d'un pas nominal (diagnostic).

2) Détection "filet" des décrues candidates (détection large)
   - Détection de blocs continus où :
       P <= rain_thresh_detect_mm  ET  Q >= q_min_m3s
   - Le début de décrue est défini au maximum local de Q dans le bloc sec,
     la fin au dernier point du bloc.

3) Nettoyage intra-segment (rendre ln(Q) exploitable)
   - Suppression des points avec pluie résiduelle > rain_keep_mm.
   - Suppression des drop_first_n premiers points (transitoire post-pic).
   - Suppression des Q <= 0.
   - Option anti-rebond : retrait des points où Q remonte au-delà d'une tolérance relative.

4) Préparation du signal pour le fit
   - ln(Q) (par défaut).
   - Option "baseflow removal" : ln(Q - Qb) avec Qb défini comme un quantile bas de Q (baseflow_quantile),
     et filtrage des Qeff trop proches de 0 (baseflow_min_pos_q).

5) Ajustements
   - Modèle 1 segment : régression linéaire ln(Q) = a + b t => k = -b.
   - Modèle 2 segments : recherche d'une rupture, fit de deux droites continues en ln(Q), pentes négatives.

6) Décision 1 vs 2 segments (acceptation du 2 segments)
   - Critères "physique" : k1>0 et k2>0 (pentes négatives).
   - Critère robuste principal : R² par segment >= min_R2_seg (+ plancher global optionnel).
   - Cohérence : contraste minimal entre k1 et k2 (min_delta_k_rel) + contraintes optionnelles d'ordre/ratio.
   - Parcimonie : BIC meilleur OU BIC pas "trop pire" OU gain R² suffisant (selon MODE).


Réglages importants (à modifier en priorité)
--------------------------------------------
- MODE (en haut du fichier) : "strict" | "loose" | "loose_two"
  * strict     : segments rares mais "propres" (BIC strict, critères élevés)
  * loose      : plus de segments, lissage lnQ, tolérance pluie résiduelle
  * loose_two  : favorise la détection de doubles décrues, anti-rebond + retrait baseflow (recommandé Cloutasse)

- Paramètres de détection (impact : combien de décrues candidates)
  * rain_thresh_detect_mm : seuil pluie max pour considérer "sec" (détection)
  * min_len_pts_detect    : longueur minimale d'un bloc sec
  * q_min_m3s             : seuil débit minimal pour éviter les pseudo-décrues à très bas débit

- Paramètres de nettoyage (impact : qualité des fits)
  * rain_keep_mm          : pluie résiduelle tolérée à l'intérieur du segment
  * drop_first_n          : nombre de points supprimés après le pic
  * drop_nonmonotone + monotone_tol_rel : anti-rebond (recommandé sur données bruitées)
  * min_points_clean      : nb minimal de points après nettoyage

- Paramètres du 2 segments (impact : acceptation des doubles pentes)
  * min_seg_points        : nb minimal de points par segment
  * min_R2_seg            : R² minimal par segment (critère principal)
  * delta_BIC_min / allow_BIC_worse_up_to : tolérance BIC vs 1 segment
  * min_delta_k_rel       : contraste minimal entre k1 et k2
  * enforce_order / enforce_ratio : impose éventuellement k_fast>=k_slow et/ou un ratio minimal

- Baseflow removal (impact : stabilité sur bassin naturel)
  * remove_baseflow       : True/False
  * baseflow_quantile     : quantile bas utilisé pour Qb
  * baseflow_min_pos_q    : quantile de sécurité pour exclure les Qeff trop petits


Interprétation des sorties :
- k_unique_s-1 : constante de vidange unique (1 régime) issue de la pente de ln(Q).
- k_fast_s-1 / k_slow_s-1 : constantes issues d'un fit 2 segments accepté (valeurs classées max/min).
- t_break_h : instant de rupture (heures) depuis le début de la décrue retenue.
- R2_* et BIC_* : diagnostics de qualité et parcimonie (R² seg-wise à privilégier sur données bruitées).
- Qb_m3s_used : débit de base retiré si remove_baseflow=True (sinon NaN).

Exécution :
Depuis l'environnement projet :
- Placer le fichier dans 01_Code/ (ou équivalent) en conservant l'arborescence attendue.
- Vérifier que 02_Data/PQ_BV_Cloutasse_interp.csv existe et respecte les noms de colonnes.
- Lancer :
    python <nom_du_script>.py
Les résultats sont générés dans 03_Plots/Identification constantes vidange/filet_{MODE}/.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0


# =========================================================
# CONFIG
# =========================================================
MODE = "loose_two"  # "strict" | "loose" | "loose_two"

CONFIGS: Dict[str, Dict] = {
    "strict": dict(
        # Détection brute d’un bloc "sec & Q>seuil"
        rain_thresh_detect_mm=0.2,
        min_len_pts_detect=8,
        q_min_m3s=0.001,

        # Nettoyage points au sein du segment
        rain_keep_mm=0.0,
        min_points_clean=10,
        drop_first_n=1,

        # Fit 2 segments
        min_seg_points=8,
        delta_BIC_min=2.0,
        allow_BIC_worse_up_to=0.0,
        min_delta_R2=0.00,   # pas trop utile en strict si BIC strict
        min_R2_two=0.80,     # indicatif (R² global)
        min_R2_seg=0.80,     # R² par segment
        min_delta_k_rel=0.30,
        enforce_order=True,
        enforce_ratio=True,
        min_ratio_k=1.30,

        # Lissage / robustesse
        smooth_lnQ=False,
        smooth_window=1,

        # Anti-rebond
        drop_nonmonotone=False,
        monotone_tol_rel=0.02,

        # Baseflow removal
        remove_baseflow=False,
        baseflow_quantile=0.10,
        baseflow_min_pos_q=0.05,  # quantile de sécurité pour exclure Qeff trop petits
    ),

    "loose": dict(
        rain_thresh_detect_mm=0.2,
        min_len_pts_detect=6,
        q_min_m3s=0.001,

        rain_keep_mm=0.15,
        min_points_clean=8,
        drop_first_n=1,

        min_seg_points=4,
        delta_BIC_min=0.0,
        allow_BIC_worse_up_to=2.0,
        min_delta_R2=0.02,
        min_R2_two=0.55,     # indicatif
        min_R2_seg=0.55,
        min_delta_k_rel=0.18,
        enforce_order=True,
        enforce_ratio=False,
        min_ratio_k=1.0,

        smooth_lnQ=True,
        smooth_window=3,

        drop_nonmonotone=False,
        monotone_tol_rel=0.03,

        remove_baseflow=False,
        baseflow_quantile=0.10,
        baseflow_min_pos_q=0.05,
    ),

    # ⭐ Recommandé pour "beaucoup plus de doubles décrues" (mais sans partir en sucette)
    "loose_two": dict(
        rain_thresh_detect_mm=0.2,
        min_len_pts_detect=6,
        q_min_m3s=0.0007,      # un peu plus bas => + de segments candidats

        rain_keep_mm=0.20,     # autorise pluie résiduelle faible
        min_points_clean=10,   # + de points => fits plus stables
        drop_first_n=1,

        min_seg_points=5,      # 4 est souvent instable ; 5-6 = bon compromis
        delta_BIC_min=-2.0,    # autorise 2-seg même si BIC pas meilleur
        allow_BIC_worse_up_to=6.0,
        min_delta_R2=0.00,     # ne pas exiger gain R² global
        min_R2_two=0.35,       # indicatif global
        min_R2_seg=0.50,       # segment-wise (critère principal)
        min_delta_k_rel=0.12,
        enforce_order=False,   # au Cloutasse, l'ordre fast>=slow n'est pas toujours net
        enforce_ratio=False,
        min_ratio_k=1.0,

        smooth_lnQ=True,
        smooth_window=5,

        drop_nonmonotone=True,   # ⭐ anti-rebond
        monotone_tol_rel=0.03,

        remove_baseflow=True,    # ⭐ très rentable au Cloutasse
        baseflow_quantile=0.10,
        baseflow_min_pos_q=0.05,
    ),
}

if MODE not in CONFIGS:
    raise ValueError(f"MODE invalide: {MODE}. Choisir parmi {list(CONFIGS.keys())}.")
CFG = CONFIGS[MODE]


# =========================================================
# I/O
# =========================================================
def read_full_series() -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, float]:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "02_Data"
    csv_path = data_dir / "PQ_BV_Cloutasse_interp.csv"

    df = pd.read_csv(
        csv_path,
        sep=";",
        na_values=["NA", "NaN", "", -9999, -9999.0],
    )
    for col in ("dateP", "P_mm", "Q_ls"):
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' manquante dans {csv_path.name}")

    time_index = pd.DatetimeIndex(pd.to_datetime(df["dateP"]))
    P_mm = df["P_mm"].astype(float).fillna(0.0).to_numpy()
    Q_ls = df["Q_ls"].astype(float).to_numpy()
    Q_m3s = Q_ls / 1000.0

    if len(time_index) >= 2:
        dt_seconds = (time_index[1] - time_index[0]).total_seconds()
    else:
        dt_seconds = 300.0

    return time_index, P_mm, Q_m3s, float(dt_seconds)


# =========================================================
# DETECTION
# =========================================================
def detect_recessions_from_series(
    time_index: pd.DatetimeIndex,
    P_mm: np.ndarray,
    Q_m3s: np.ndarray,
    dt_sec_nominal: float,
    rain_thresh_mm: float,
    min_len_pts: int,
    q_min_m3s: float,
) -> List[dict]:
    """
    Détecte des blocs continus où (P <= seuil) et (Q >= seuil),
    puis définit la décrue comme [pic local -> fin du bloc].
    """
    n = len(Q_m3s)
    is_dry = P_mm <= rain_thresh_mm
    is_above_q = Q_m3s >= q_min_m3s
    mask = is_dry & is_above_q

    recessions: List[dict] = []
    i = 0
    while i < n:
        if mask[i]:
            start_block = i
            j = i + 1
            while j < n and mask[j]:
                j += 1
            end_block = j - 1

            if end_block - start_block + 1 >= min_len_pts:
                idx_max_local = start_block + int(np.argmax(Q_m3s[start_block : end_block + 1]))
                start = idx_max_local
                end = end_block

                if end - start + 1 >= min_len_pts:
                    seg_time = time_index[start : end + 1]
                    seg_P = P_mm[start : end + 1]
                    seg_Q = Q_m3s[start : end + 1]
                    name = f"rec_{seg_time[0].strftime('%Y%m%d_%H%M')}"
                    recessions.append(
                        dict(
                            name=name,
                            start_idx=int(start),
                            end_idx=int(end),
                            time=seg_time,
                            P_mm=seg_P,
                            Q_m3s=seg_Q,
                            dt_sec_nominal=float(dt_sec_nominal),
                        )
                    )
            i = j
        else:
            i += 1

    return recessions


# =========================================================
# CLEANING
# =========================================================
def clean_recession_segment_with_time(
    time_seg: pd.DatetimeIndex,
    Q_m3s: np.ndarray,
    P_mm: np.ndarray,
    *,
    rain_keep_mm: float,
    min_points: int,
    drop_first_n: int,
    drop_nonmonotone: bool,
    monotone_tol_rel: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Nettoyage:
    - retire points avec pluie résiduelle > rain_keep_mm
    - retire les drop_first_n premiers points (transitoire post-pic)
    - retire Q<=0
    - optionnel: retire points qui "rebondissent" trop (anti-rebond)
    Retourne t_sec (temps réel depuis premier point retenu), Qc, Pc.
    """
    Q = np.asarray(Q_m3s, float)
    P = np.asarray(P_mm, float)
    if len(Q) < min_points:
        return None

    mask = np.ones(len(Q), dtype=bool)
    mask &= (P <= float(rain_keep_mm))
    if drop_first_n > 0:
        mask[:drop_first_n] = False
    mask &= (Q > 0.0)

    if int(np.sum(mask)) < min_points:
        return None

    t0 = time_seg[mask][0]
    t_sec = (time_seg[mask] - t0).total_seconds().astype(float)
    Qc = Q[mask]
    Pc = P[mask]

    # sécurité monotonie du temps
    if len(t_sec) >= 2 and np.any(np.diff(t_sec) <= 0):
        return None

    # option : quasi-monotone (anti-rebond)
    if drop_nonmonotone:
        tol = float(monotone_tol_rel)
        keep = np.ones(len(Qc), dtype=bool)
        for i in range(1, len(Qc)):
            if Qc[i] > Qc[i - 1] * (1.0 + tol):
                keep[i] = False
        if int(np.sum(keep)) < min_points:
            return None
        t_sec = t_sec[keep]
        Qc = Qc[keep]
        Pc = Pc[keep]

    return t_sec, Qc, Pc


# =========================================================
# FIT HELPERS
# =========================================================
def smooth_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, float)
    if window is None or window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def fit_single_segment(t_sec: np.ndarray, lnQ: np.ndarray) -> dict:
    b, a = np.polyfit(t_sec, lnQ, 1)
    pred = a + b * t_sec
    resid = lnQ - pred

    rss = float(np.sum(resid**2))
    tss = float(np.sum((lnQ - lnQ.mean()) ** 2))
    R2 = 1.0 - rss / tss if tss > 0 else np.nan
    k = -b

    return {"slope": float(b), "intercept": float(a), "k": float(k), "RSS": rss, "TSS": tss, "R2": float(R2)}


def bic_from_rss(rss: float, n: int, p: int) -> float:
    n_points = float(n)
    return float(n_points * np.log(rss / n_points) + p * np.log(n_points))


def fit_two_segments(
    t_sec: np.ndarray,
    lnQ: np.ndarray,
    *,
    min_seg_points: int,
    enforce_negative_slopes: bool = True,
) -> Optional[dict]:
    """
    Fit 2 segments continus en ln(Q) avec une rupture.
    On explore les k_break possibles, on minimise RSS_total.
    Option: imposer pentes négatives (physique décrue).
    """
    n = len(t_sec)
    if n < 2 * min_seg_points + 1:
        return None

    best = None
    rss_best = np.inf

    tss = float(np.sum((lnQ - lnQ.mean()) ** 2))
    p_two = 4
    n_points = float(n)

    for k_break in range(min_seg_points, n - min_seg_points):
        t_b = float(t_sec[k_break])

        t1 = t_sec[: k_break + 1]
        lnQ1 = lnQ[: k_break + 1]
        b1, a1 = np.polyfit(t1, lnQ1, 1)

        if enforce_negative_slopes and b1 >= 0:
            continue

        lnQ_b = a1 + b1 * t_b

        t2 = t_sec[k_break + 1 :]
        lnQ2 = lnQ[k_break + 1 :]
        if len(t2) < min_seg_points:
            continue

        b2, _ = np.polyfit(t2, lnQ2, 1)

        if enforce_negative_slopes and b2 >= 0:
            continue

        a2 = lnQ_b - b2 * t_b

        pred1 = a1 + b1 * t1
        pred2 = a2 + b2 * t2
        rss1 = float(np.sum((lnQ1 - pred1) ** 2))
        rss2 = float(np.sum((lnQ2 - pred2) ** 2))
        rss_tot = rss1 + rss2

        if rss_tot < rss_best:
            rss_best = rss_tot
            k1 = -b1
            k2 = -b2

            # R² global (indicatif)
            R2_two = 1.0 - rss_tot / tss if tss > 0 else np.nan
            bic_two = n_points * np.log(rss_tot / n_points) + p_two * np.log(n_points)

            # R² par segment (critère robuste)
            tss1 = float(np.sum((lnQ1 - lnQ1.mean()) ** 2))
            tss2 = float(np.sum((lnQ2 - lnQ2.mean()) ** 2))
            R2_1 = 1.0 - rss1 / tss1 if tss1 > 0 else np.nan
            R2_2 = 1.0 - rss2 / tss2 if tss2 > 0 else np.nan

            best = {
                "k_break_idx": int(k_break),
                "t_break_sec": float(t_b),
                "slope1": float(b1),
                "intercept1": float(a1),
                "RSS1": rss1,
                "k1": float(k1),
                "R2_1": float(R2_1),
                "slope2": float(b2),
                "intercept2": float(a2),
                "RSS2": rss2,
                "k2": float(k2),
                "R2_2": float(R2_2),
                "RSS_total": rss_tot,
                "TSS": tss,
                "R2_two": float(R2_two),
                "BIC_two": float(bic_two),
            }

    return best


def evaluate_two_segment_acceptance(single: dict, two: dict, n: int, cfg: Dict) -> Tuple[bool, dict]:
    """
    Retourne (use_two, diagnostics).
    Philosophie :
    - physiquement : k1>0 & k2>0
    - qualité : R² par segment >= min_R2_seg (prioritaire)
    - cohérence : contraste minimal entre k1 et k2 (min_delta_k_rel)
    - choix modèle : BIC meilleur OU (R² gain) OU BIC pas "trop pire"
    """
    rss_single = float(single["RSS"])
    R2_single = float(single["R2"])
    BIC_single = bic_from_rss(rss_single, n, p=2)

    BIC_two = float(two["BIC_two"])
    R2_two = float(two["R2_two"])
    k1 = float(two["k1"])
    k2 = float(two["k2"])
    R2_1 = float(two.get("R2_1", np.nan))
    R2_2 = float(two.get("R2_2", np.nan))

    ratio_k = (k1 / k2) if (k1 > 0 and k2 > 0) else np.nan
    delta_BIC = BIC_two - BIC_single
    delta_R2 = R2_two - R2_single
    delta_k_rel = (abs(k1 - k2) / max(k1, k2)) if (k1 > 0 and k2 > 0) else np.nan

    cond_positive = (k1 > 0.0) and (k2 > 0.0)

    # ⭐ critère robuste : R² par segment
    min_R2_seg = float(cfg.get("min_R2_seg", cfg.get("min_R2_two", 0.6)))
    cond_R2_seg = (R2_1 >= min_R2_seg) and (R2_2 >= min_R2_seg)

    # (optionnel) garder aussi un plancher global indicatif
    min_R2_two = float(cfg.get("min_R2_two", 0.0))
    cond_R2_global = (R2_two >= min_R2_two)

    cond_R2_floor = cond_R2_seg and cond_R2_global

    cond_order = True
    if bool(cfg.get("enforce_order", False)):
        cond_order = (k1 >= k2)

    cond_ratio = True
    if bool(cfg.get("enforce_ratio", False)):
        cond_ratio = (ratio_k >= float(cfg.get("min_ratio_k", 1.0)))

    cond_contrast = True
    if not np.isnan(delta_k_rel):
        cond_contrast = (delta_k_rel >= float(cfg.get("min_delta_k_rel", 0.0)))

    cond_BIC_good = (BIC_two < (BIC_single - float(cfg.get("delta_BIC_min", 0.0))))
    cond_R2_gain = (delta_R2 >= float(cfg.get("min_delta_R2", 0.0)))
    cond_BIC_ok = (BIC_two <= (BIC_single + float(cfg.get("allow_BIC_worse_up_to", 0.0))))

    cond_model_choice = (cond_BIC_good or cond_R2_gain or cond_BIC_ok)

    use_two = cond_positive and cond_R2_floor and cond_order and cond_ratio and cond_contrast and cond_model_choice

    diag = {
        "BIC_single": float(BIC_single),
        "BIC_two": float(BIC_two),
        "delta_BIC": float(delta_BIC),
        "R2_single": float(R2_single),
        "R2_two": float(R2_two),
        "R2_1": float(R2_1),
        "R2_2": float(R2_2),
        "delta_R2": float(delta_R2),
        "k1": float(k1),
        "k2": float(k2),
        "ratio_k": float(ratio_k) if ratio_k == ratio_k else np.nan,
        "delta_k_rel": float(delta_k_rel) if delta_k_rel == delta_k_rel else np.nan,

        "cond_positive": bool(cond_positive),
        "cond_R2_floor": bool(cond_R2_floor),
        "cond_R2_seg": bool(cond_R2_seg),
        "cond_R2_global": bool(cond_R2_global),
        "cond_order": bool(cond_order),
        "cond_ratio": bool(cond_ratio),
        "cond_contrast": bool(cond_contrast),
        "cond_model_choice": bool(cond_model_choice),

        "cond_BIC_good": bool(cond_BIC_good),
        "cond_R2_gain": bool(cond_R2_gain),
        "cond_BIC_ok": bool(cond_BIC_ok),

        "t_break_sec": float(two["t_break_sec"]),
        "k_break_idx": int(two["k_break_idx"]),
    }
    return use_two, diag


def compute_lnQ_for_fit(
    t_sec: np.ndarray,
    Q_clean: np.ndarray,
    cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Retourne (t_sec_used, lnQ_used, Qb_used).
    Si remove_baseflow=True : ln(Q - Qb) avec Qb quantile bas.
    """
    Qb_used = np.nan
    t_used = t_sec
    Q_used = Q_clean

    if bool(cfg.get("remove_baseflow", False)):
        q_b = float(np.quantile(Q_clean, float(cfg.get("baseflow_quantile", 0.10))))
        Q_eff = Q_clean - q_b
        Qb_used = q_b

        # garder des valeurs strictement positives et pas trop proches de 0
        if not np.any(Q_eff > 0):
            return t_sec, np.log(Q_clean), Qb_used  # fallback
        pos = Q_eff[Q_eff > 0]
        q_min_pos = float(np.quantile(pos, float(cfg.get("baseflow_min_pos_q", 0.05))))
        ok = Q_eff > q_min_pos
        if int(np.sum(ok)) < int(cfg.get("min_points_clean", 8)):
            return t_sec, np.log(Q_clean), Qb_used  # fallback

        t_used = t_sec[ok]
        Q_used = Q_eff[ok]

    lnQ = np.log(Q_used)
    return t_used, lnQ, float(Qb_used)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir.parent / "03_Plots" / "Identification constantes vidange" / f"filet_{MODE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out_dir / "Recessions_QA"
    qa_dir.mkdir(parents=True, exist_ok=True)

    time_index, P_mm, Q_m3s, dt_sec_nominal = read_full_series()

    recessions = detect_recessions_from_series(
        time_index, P_mm, Q_m3s, dt_sec_nominal,
        rain_thresh_mm=float(CFG["rain_thresh_detect_mm"]),
        min_len_pts=int(CFG["min_len_pts_detect"]),
        q_min_m3s=float(CFG["q_min_m3s"]),
    )

    print(f"[INFO] MODE={MODE}")
    print(f"[INFO] Nombre de segments de décrue détectés (filet large) : {len(recessions)}")

    rows: List[dict] = []
    rows_two_candidates: List[dict] = []
    rej_counts = {
        "positive": 0,
        "R2_floor": 0,
        "order": 0,
        "ratio": 0,
        "contrast": 0,
        "model_choice": 0,
    }

    n_candidate_two = 0
    n_kept_two = 0

    for rec in recessions:
        name = rec["name"]
        seg_time = rec["time"]
        seg_P = rec["P_mm"]
        seg_Q = rec["Q_m3s"]

        cleaned = clean_recession_segment_with_time(
            seg_time, seg_Q, seg_P,
            rain_keep_mm=float(CFG["rain_keep_mm"]),
            min_points=int(CFG["min_points_clean"]),
            drop_first_n=int(CFG["drop_first_n"]),
            drop_nonmonotone=bool(CFG.get("drop_nonmonotone", False)),
            monotone_tol_rel=float(CFG.get("monotone_tol_rel", 0.03)),
        )
        if cleaned is None:
            continue

        t_sec_raw, Q_clean_raw, _ = cleaned
        t_h_raw = t_sec_raw / 3600.0

        # lnQ (avec option baseflow removal)
        t_sec, lnQ, Qb_used = compute_lnQ_for_fit(t_sec_raw, Q_clean_raw, CFG)

        # lissage optionnel (sur lnQ)
        if bool(CFG["smooth_lnQ"]):
            lnQ = smooth_moving_average(lnQ, int(CFG["smooth_window"]))

        # décrue exploitable si décroissance nette
        delta_lnQ = float(lnQ[0] - lnQ[-1])
        if delta_lnQ <= 0:
            continue

        # Fit 1 segment
        single = fit_single_segment(t_sec, lnQ)
        if float(single["k"]) <= 0:
            continue

        # Fit 2 segments
        two = fit_two_segments(
            t_sec, lnQ,
            min_seg_points=int(CFG["min_seg_points"]),
            enforce_negative_slopes=True,
        )

        best_model = "single"
        k_unique = float(single["k"])
        k_fast = np.nan
        k_slow = np.nan

        diag = {
            "BIC_single": bic_from_rss(float(single["RSS"]), len(t_sec), 2),
            "BIC_two": np.nan,
            "delta_BIC": np.nan,
            "R2_two": np.nan,
            "R2_1": np.nan,
            "R2_2": np.nan,
            "delta_R2": np.nan,
            "ratio_k": np.nan,
            "delta_k_rel": np.nan,
            "t_break_sec": np.nan,
        }

        if two is not None:
            n_candidate_two += 1
            use_two, full_diag = evaluate_two_segment_acceptance(single, two, len(t_sec), CFG)

            if not use_two:
                if not full_diag["cond_positive"]:
                    rej_counts["positive"] += 1
                if not full_diag["cond_R2_floor"]:
                    rej_counts["R2_floor"] += 1
                if not full_diag["cond_order"]:
                    rej_counts["order"] += 1
                if not full_diag["cond_ratio"]:
                    rej_counts["ratio"] += 1
                if not full_diag["cond_contrast"]:
                    rej_counts["contrast"] += 1
                if not full_diag["cond_model_choice"]:
                    rej_counts["model_choice"] += 1

            # export candidats (même rejetés)
            rows_two_candidates.append({
                "name": name,
                "n_points": int(len(t_sec)),
                "duration_h": float(t_sec[-1] / 3600.0) if len(t_sec) >= 2 else 0.0,
                "Q_max_ls": float(np.max(Q_clean_raw) * 1000.0),
                "Qb_m3s_used": float(Qb_used) if Qb_used == Qb_used else np.nan,
                "t_break_h": float(full_diag["t_break_sec"] / 3600.0),
                "k1_s-1": float(full_diag["k1"]),
                "k2_s-1": float(full_diag["k2"]),
                "k_fast_if_order_s-1": float(max(full_diag["k1"], full_diag["k2"])),
                "k_slow_if_order_s-1": float(min(full_diag["k1"], full_diag["k2"])),
                "R2_single": float(full_diag["R2_single"]),
                "R2_two": float(full_diag["R2_two"]),
                "R2_1": float(full_diag["R2_1"]),
                "R2_2": float(full_diag["R2_2"]),
                "delta_R2": float(full_diag["delta_R2"]),
                "BIC_single": float(full_diag["BIC_single"]),
                "BIC_two": float(full_diag["BIC_two"]),
                "delta_BIC": float(full_diag["delta_BIC"]),
                "ratio_k": float(full_diag["ratio_k"]) if full_diag["ratio_k"] == full_diag["ratio_k"] else np.nan,
                "delta_k_rel": float(full_diag["delta_k_rel"]) if full_diag["delta_k_rel"] == full_diag["delta_k_rel"] else np.nan,
                "accepted_two": bool(use_two),
                "fails_positive": not bool(full_diag["cond_positive"]),
                "fails_R2_floor": not bool(full_diag["cond_R2_floor"]),
                "fails_order": not bool(full_diag["cond_order"]),
                "fails_ratio": not bool(full_diag["cond_ratio"]),
                "fails_contrast": not bool(full_diag["cond_contrast"]),
                "fails_model_choice": not bool(full_diag["cond_model_choice"]),
            })

            if use_two:
                best_model = "two"
                n_kept_two += 1
                k1 = float(two["k1"])
                k2 = float(two["k2"])
                k_fast = float(max(k1, k2))
                k_slow = float(min(k1, k2))
                k_unique = np.nan

                diag = {
                    "BIC_single": float(full_diag["BIC_single"]),
                    "BIC_two": float(full_diag["BIC_two"]),
                    "delta_BIC": float(full_diag["delta_BIC"]),
                    "R2_two": float(full_diag["R2_two"]),
                    "R2_1": float(full_diag["R2_1"]),
                    "R2_2": float(full_diag["R2_2"]),
                    "delta_R2": float(full_diag["delta_R2"]),
                    "ratio_k": float(full_diag["ratio_k"]) if full_diag["ratio_k"] == full_diag["ratio_k"] else np.nan,
                    "delta_k_rel": float(full_diag["delta_k_rel"]) if full_diag["delta_k_rel"] == full_diag["delta_k_rel"] else np.nan,
                    "t_break_sec": float(two["t_break_sec"]),
                }

        # stock résultats (tableau global)
        rows.append({
            "name": name,
            "n_points": int(len(t_sec)),
            "dt_sec_nominal": float(dt_sec_nominal),
            "duration_h": float(t_sec[-1] / 3600.0) if len(t_sec) >= 2 else 0.0,
            "Q_max_ls": float(np.max(Q_clean_raw) * 1000.0),
            "Qb_m3s_used": float(Qb_used) if Qb_used == Qb_used else np.nan,
            "model": best_model,
            "k_unique_s-1": k_unique,
            "k_fast_s-1": k_fast,
            "k_slow_s-1": k_slow,
            "R2_single": float(single["R2"]),
            "R2_two": diag["R2_two"],
            "R2_1": diag["R2_1"],
            "R2_2": diag["R2_2"],
            "delta_R2": diag["delta_R2"],
            "BIC_single": diag["BIC_single"],
            "BIC_two": diag["BIC_two"],
            "delta_BIC": diag["delta_BIC"],
            "ratio_k": diag["ratio_k"],
            "delta_k_rel": diag["delta_k_rel"],
            "t_break_h": float(diag["t_break_sec"] / 3600.0) if diag["t_break_sec"] == diag["t_break_sec"] else np.nan,
            "delta_lnQ": float(delta_lnQ),
        })

        # QA plot (toujours sur t_sec et lnQ réellement utilisés)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(t_sec / 3600.0, lnQ, s=10, color="tab:blue", label="ln(Q) (après nettoyage)")

        pred_single = single["intercept"] + single["slope"] * t_sec
        ax.plot(t_sec / 3600.0, pred_single, color="grey", linestyle="--",
                label=f"1 seg (k={single['k']:.2e}, R²={single['R2']:.3f})")

        if best_model == "two" and two is not None:
            kb = two["k_break_idx"]
            t1 = t_sec[: kb + 1]
            t2 = t_sec[kb + 1 :]
            ax.plot(t1 / 3600.0, two["intercept1"] + two["slope1"] * t1,
                    color="tab:orange", label=f"Seg1 (k={two['k1']:.2e}, R²={two.get('R2_1', np.nan):.2f})")
            ax.plot(t2 / 3600.0, two["intercept2"] + two["slope2"] * t2,
                    color="tab:green", label=f"Seg2 (k={two['k2']:.2e}, R²={two.get('R2_2', np.nan):.2f})")
            ax.axvline(two["t_break_sec"] / 3600.0, color="k", linestyle=":", alpha=0.7, label="Rupture")

        ax.set_xlabel("Temps depuis début décrue (h) [temps réel]")
        ax.set_ylabel("ln(Q) (ln(m³/s))" + (" (Q-Qb)" if bool(CFG.get("remove_baseflow", False)) else ""))
        title = f"{name} — modèle : {best_model} (MODE={MODE})"
        if bool(CFG.get("remove_baseflow", False)) and (Qb_used == Qb_used):
            title += f" | Qb={Qb_used:.2e} m³/s"
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(qa_dir / f"{name}_lnQ_QA.png", dpi=200)
        plt.close(fig)

    # exports
    if not rows:
        print("[WARN] Aucun résultat exploitable.")
        return

    df_res = pd.DataFrame(rows)
    out_xlsx = out_dir / f"recessions_and_k_PRO_{MODE}.xlsx"
    df_res.to_excel(out_xlsx, index=False)
    print(f"\n[OK] Tableau global enregistré : {out_xlsx}")

    if rows_two_candidates:
        df_cand = pd.DataFrame(rows_two_candidates)
        out_cand = out_dir / f"two_segment_candidates_{MODE}.xlsx"
        df_cand.to_excel(out_cand, index=False)
        print(f"[OK] Candidats 2 segments exportés : {out_cand}")

    print(f"\n[INFO] Candidats 2 segments (fit possible) : {n_candidate_two}")
    print(f"[INFO] Retenus 2 segments (après critères) : {n_kept_two}")

    print("\n=== REJECTIONS 2 segments (comptage) ===")
    if n_candidate_two > 0:
        for k, v in rej_counts.items():
            print(f"  {k:12s}: {v}")
    else:
        print("  Aucun candidat 2 segments.")

    print("\n[FIN OK]")


if __name__ == "__main__":
    main()
