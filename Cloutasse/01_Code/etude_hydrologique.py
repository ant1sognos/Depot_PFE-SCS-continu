# -*- coding: utf-8 -*-
"""
Script de découpage d'événements pluie-ruissellement pour le bassin du Cloutasse.

Ce script implémente la méthodologie décrite dans la Section 2.3 du rapport PFE :
  1. Détection de noyaux d'activité pluie-débit
  2. Extension temporelle avec estimation de baseflow local
  3. Filtrage qualité
  4. Calcul des signatures volumétriques (hp, hr)
  5. Génération des visualisations

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ======================================================================
# CONFIGURATION GLOBALE
# ======================================================================

class Config:
    """Configuration globale pour le découpage d'événements."""
    
    # Bassin versant
    A_BV_M2 = 0.81e6  # Surface du bassin (m²)
    
    # Baseflow
    BASE_PRE_HOURS = 6.0  # Fenêtre pour estimer Q0 avant événement (h)
    BASEFLOW_MARGIN_LS = 1.0  # Marge au-dessus de Q0 pour "en charge" (L/s)
    
    # Détection
    P_THR_MM = 0.01  # Seuil de pluie par pas de temps (mm)
    MIN_DQ_RESPONSE_LS = 0.5  # ΔQ minimum entre base et pic (L/s)
    MIN_AMP_LS = 0.5  # Amplitude globale minimale de Q (L/s)
    
    # Fusion et extension temporelle
    RAIN_GAP_HOURS = 4.0  # Gap max pour fusionner 2 blocs de pluie (h)
    POST_DRY_HOURS = 12.0  # Durée de calme pour clôturer (h)
    PRE_HOURS = 6.0  # Fenêtre max pour remonter avant la pluie (h)
    MAX_PEAK_DELAY_HOURS = 96.0  # Délai max entre 1ère pluie et pic (h)
    
    # Filtrage qualité
    MIN_P_TOT_MM = 0.01  # Pluie totale minimale (mm)
    MIN_Q_AMP_LS = 1.0  # Amplitude de débit minimale (L/s)
    MAX_DURATION_H = 1400.0  # Durée maximale d'un événement (h)
    
    # Pluie antécédente
    ANTECEDENT_DAYS = 3  # Durée pour calculer P_ante (jours)
    
    # Séquences humides (mémoire hydrologique)
    MAX_GAP_SEQUENCE_HOURS = 72.0  # Gap max pour rester dans une séquence (h)
    P_ANTE_WET_THR = 10.0  # Seuil P_ante pour sol "humide" (mm)
    
    # Mode de fonctionnement
    USE_EXISTING_EVENTS = False  # Si True, recharge les événements existants


# ======================================================================
# UTILITAIRES - BASEFLOW
# ======================================================================

def estimate_global_baseflow(q: np.ndarray) -> float:
    """
    Estime un débit de base global par quantile 10% (fallback).
    
    Parameters
    ----------
    q : np.ndarray
        Série de débits (L/s)
        
    Returns
    -------
    float
        Débit de base global (L/s)
    """
    q_finite = q[np.isfinite(q)]
    if q_finite.size == 0:
        return 0.0
    return float(np.nanpercentile(q_finite, 10.0))


def estimate_local_baseflow(
    q: np.ndarray,
    i_start_rain: int,
    dt: float,
    pre_base_hours: float = Config.BASE_PRE_HOURS,
    global_fallback: float = 0.0
) -> float:
    """
    Estime le débit de base local avant un événement (médiane sur fenêtre).
    
    Parameters
    ----------
    q : np.ndarray
        Série complète de débits (L/s)
    i_start_rain : int
        Indice de début de pluie
    dt : float
        Pas de temps (secondes)
    pre_base_hours : float
        Durée de la fenêtre avant l'événement (heures)
    global_fallback : float
        Valeur de secours si la fenêtre est vide
        
    Returns
    -------
    float
        Débit de base local Q0 (L/s)
    """
    pre_steps = int(pre_base_hours * 3600.0 / dt)
    i_pre_start = max(0, i_start_rain - pre_steps)
    Q_pre = q[i_pre_start:i_start_rain]
    
    if Q_pre.size > 0 and np.any(np.isfinite(Q_pre)):
        return float(np.nanmedian(Q_pre))
    else:
        return global_fallback


# ======================================================================
# ÉTAPE 1 : DÉTECTION DES NOYAUX D'ACTIVITÉ
# ======================================================================

def detect_rain_blocks(
    P: np.ndarray,
    P_thr_mm: float = Config.P_THR_MM
) -> List[Tuple[int, int]]:
    """
    Détecte les segments de pluie contigus (Section 2.3.1 - étape 1).
    
    Parameters
    ----------
    P : np.ndarray
        Série de pluie par pas de temps (mm)
    P_thr_mm : float
        Seuil de pluie significative (mm)
        
    Returns
    -------
    List[Tuple[int, int]]
        Liste de segments (i_start, i_end) de pluie continue
    """
    raining = P > P_thr_mm
    rain_idx = np.where(raining)[0]
    
    if rain_idx.size == 0:
        return []
    
    segments = []
    start = rain_idx[0]
    prev = rain_idx[0]
    
    for idx in rain_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = prev = idx
    segments.append((start, prev))
    
    return segments


def merge_rain_blocks(
    segments: List[Tuple[int, int]],
    gap_hours: float,
    dt: float
) -> List[Tuple[int, int]]:
    """
    Fusionne les blocs de pluie séparés par un gap court (même épisode synoptique).
    
    Parameters
    ----------
    segments : List[Tuple[int, int]]
        Blocs de pluie contigus
    gap_hours : float
        Gap maximal pour fusionner (heures)
    dt : float
        Pas de temps (secondes)
        
    Returns
    -------
    List[Tuple[int, int]]
        Blocs fusionnés
    """
    if not segments:
        return []
    
    gap_steps = int(gap_hours * 3600.0 / dt)
    merged = []
    
    for s, e in segments:
        if not merged:
            merged.append([s, e])
        else:
            last_s, last_e = merged[-1]
            if s <= last_e + gap_steps:
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])
    
    return [(s, e) for s, e in merged]


# ======================================================================
# ÉTAPE 2 : EXTENSION TEMPORELLE AVEC BASEFLOW LOCAL
# ======================================================================

def extend_event_with_baseflow(
    P: np.ndarray,
    q: np.ndarray,
    s_rain: int,
    e_rain: int,
    dt: float,
    Q0_global: float,
    pre_hours: float = Config.PRE_HOURS,
    post_dry_hours: float = Config.POST_DRY_HOURS,
    pre_base_hours: float = Config.BASE_PRE_HOURS,
    baseflow_margin_ls: float = Config.BASEFLOW_MARGIN_LS,
    P_thr_mm: float = Config.P_THR_MM
) -> Tuple[int, int]:
    """
    Étend un bloc de pluie en événement complet avec estimation de Q0 local
    (Section 2.3.1 - étape 2).
    
    L'extension se fait :
      - Vers l'amont si Q > Q0 + marge (bassin en charge) ou s'il pleut encore
      - Vers l'aval jusqu'à retour durable à Q0
    
    Parameters
    ----------
    P : np.ndarray
        Série de pluie (mm)
    q : np.ndarray
        Série de débit (L/s)
    s_rain, e_rain : int
        Indices de début/fin du bloc de pluie fusionné
    dt : float
        Pas de temps (secondes)
    Q0_global : float
        Débit de base global (fallback)
    pre_hours, post_dry_hours, pre_base_hours : float
        Paramètres temporels (heures)
    baseflow_margin_ls : float
        Marge au-dessus de Q0 (L/s)
    P_thr_mm : float
        Seuil de pluie (mm)
        
    Returns
    -------
    Tuple[int, int]
        Indices (i0, i1) de l'événement étendu
    """
    n = len(P)
    
    # Estimation du débit de base local Q0_evt
    Q0_evt = estimate_local_baseflow(q, s_rain, dt, pre_base_hours, Q0_global)
    
    # Extension amont : remonter si Q en charge ou pluie résiduelle
    i0 = s_rain
    j = s_rain - 1
    pre_steps = int(pre_hours * 3600.0 / dt)
    steps_back = 0
    
    while j >= 0 and steps_back < pre_steps:
        q_high = np.isfinite(q[j]) and q[j] > Q0_evt + baseflow_margin_ls
        rain_active = P[j] > 0.0
        
        if q_high or rain_active:
            i0 = j
            steps_back += 1
            j -= 1
        else:
            break
    
    # Extension aval : prolonger jusqu'à retour durable à Q0
    i1 = e_rain
    j = e_rain + 1
    dry_run = 0
    post_dry_steps = int(post_dry_hours * 3600.0 / dt)
    max_j = min(n - 1, e_rain + 10 * post_dry_steps)
    
    while j <= max_j:
        cond_q = (not np.isfinite(q[j])) or (q[j] <= Q0_evt + baseflow_margin_ls)
        cond_p = (not np.isfinite(P[j])) or (P[j] <= P_thr_mm)
        
        if cond_q and cond_p:
            dry_run += 1
            if dry_run >= post_dry_steps:
                i1 = j - dry_run
                break
        else:
            dry_run = 0
            i1 = j
        j += 1
    
    i0 = max(i0, 0)
    i1 = min(i1, n - 1)
    
    return (i0, i1)


# ======================================================================
# ÉTAPE 3 : FILTRAGE QUALITÉ
# ======================================================================

def filter_by_amplitude(
    events: List[Tuple[int, int]],
    q: np.ndarray,
    min_amp_ls: float = Config.MIN_AMP_LS
) -> List[Tuple[int, int]]:
    """
    Filtre les événements par amplitude de débit minimale.
    
    Parameters
    ----------
    events : List[Tuple[int, int]]
        Événements candidats
    q : np.ndarray
        Série de débits (L/s)
    min_amp_ls : float
        Amplitude minimale requise (L/s)
        
    Returns
    -------
    List[Tuple[int, int]]
        Événements filtrés
    """
    filtered = []
    for s, e in events:
        q_seg = q[s:e + 1]
        if not np.any(np.isfinite(q_seg)):
            continue
        amp = float(np.nanmax(q_seg) - np.nanmin(q_seg))
        if amp >= min_amp_ls:
            filtered.append((s, e))
    return filtered


def filter_by_response(
    events: List[Tuple[int, int]],
    P: np.ndarray,
    q: np.ndarray,
    dt: float,
    min_dQ_ls: float = Config.MIN_DQ_RESPONSE_LS,
    max_peak_delay_hours: float = Config.MAX_PEAK_DELAY_HOURS,
    P_thr_mm: float = Config.P_THR_MM
) -> List[Tuple[int, int]]:
    """
    Filtre sur la réponse nette du débit après la 1ère pluie.
    
    Vérifie qu'il y a bien un ΔQ >= min_dQ_ls dans la fenêtre de temps
    après la première pluie (Section 2.3.1 - filtre final).
    
    Parameters
    ----------
    events : List[Tuple[int, int]]
        Événements candidats
    P : np.ndarray
        Série de pluie (mm)
    q : np.ndarray
        Série de débits (L/s)
    dt : float
        Pas de temps (secondes)
    min_dQ_ls : float
        ΔQ minimum requis (L/s)
    max_peak_delay_hours : float
        Fenêtre de recherche du pic (heures)
    P_thr_mm : float
        Seuil de pluie (mm)
        
    Returns
    -------
    List[Tuple[int, int]]
        Événements avec réponse hydrologique significative
    """
    peak_window_steps = int(max_peak_delay_hours * 3600.0 / dt)
    filtered = []
    
    for s, e in events:
        rain_mask_evt = P[s:e + 1] > P_thr_mm
        if not np.any(rain_mask_evt):
            continue
        
        first_rain_local = np.where(rain_mask_evt)[0][0]
        i_first_rain = s + first_rain_local
        i_win_end = min(e, i_first_rain + peak_window_steps)
        
        if i_win_end <= i_first_rain:
            continue
        
        q_ref = q[i_first_rain]
        q_seg_after = q[i_first_rain:i_win_end + 1]
        
        if not np.any(np.isfinite(q_seg_after)):
            continue
        
        q_peak_after = float(np.nanmax(q_seg_after))
        dQ = q_peak_after - q_ref
        
        if dQ >= min_dQ_ls:
            filtered.append((int(s), int(e)))
    
    return filtered


def filter_clean_events(
    df: pd.DataFrame,
    events: List[Tuple[int, int]],
    dt: float,
    min_P_tot_mm: float = Config.MIN_P_TOT_MM,
    min_Q_amp_ls: float = Config.MIN_Q_AMP_LS,
    max_duration_h: float = Config.MAX_DURATION_H
) -> List[Tuple[int, int]]:
    """
    Filtrage "propre" léger pour éliminer les événements aberrants.
    
    Critères :
      - Pluie totale minimale
      - Amplitude de débit minimale
      - Durée maximale acceptable
      
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame annuel avec colonnes P_mm et Q_ls
    events : List[Tuple[int, int]]
        Événements à filtrer
    dt : float
        Pas de temps (secondes)
    min_P_tot_mm, min_Q_amp_ls : float
        Seuils minimaux
    max_duration_h : float
        Durée maximale (heures)
        
    Returns
    -------
    List[Tuple[int, int]]
        Événements propres
    """
    Q_all = pd.to_numeric(df["Q_ls"], errors="coerce").values
    P_all = pd.to_numeric(df["P_mm"], errors="coerce").values
    n = len(df)
    
    clean = []
    for i0, i1 in events:
        if not (0 <= i0 < n and 0 <= i1 < n and i1 >= i0):
            continue
        
        Q_evt = Q_all[i0:i1 + 1]
        P_evt = P_all[i0:i1 + 1]
        
        # Filtre pluie totale
        Ptot = float(np.nansum(P_evt))
        if (not np.isfinite(Ptot)) or (Ptot < min_P_tot_mm):
            continue
        
        # Filtre amplitude débit
        if np.any(np.isfinite(Q_evt)):
            Qamp = float(np.nanmax(Q_evt) - np.nanmin(Q_evt))
        else:
            continue
        
        if (not np.isfinite(Qamp)) or (Qamp < min_Q_amp_ls):
            continue
        
        # Filtre durée
        duration_h = (i1 - i0 + 1) * dt / 3600.0
        if duration_h > max_duration_h:
            continue
        
        clean.append((i0, i1))
    
    return clean


# ======================================================================
# PIPELINE COMPLET DE DÉTECTION
# ======================================================================

def detect_scs_events(
    P: np.ndarray,
    q: np.ndarray,
    dt: float,
    config: Config = Config()
) -> List[Tuple[int, int]]:
    """
    Pipeline complet de détection d'événements pluie-ruissellement.
    
    Implémente la procédure décrite en Section 2.3.1 du rapport :
      1. Détection de blocs de pluie contigus
      2. Fusion des blocs proches (même épisode synoptique)
      3. Extension temporelle avec baseflow local par événement
      4. Fusion d'événements qui se chevauchent
      5. Filtre amplitude de débit
      6. Filtre réponse hydrologique (ΔQ après pluie)
    
    Parameters
    ----------
    P : np.ndarray
        Série de pluie par pas de temps (mm)
    q : np.ndarray
        Série de débit (L/s)
    dt : float
        Pas de temps (secondes)
    config : Config
        Configuration des paramètres
        
    Returns
    -------
    List[Tuple[int, int]]
        Liste d'événements (i0, i1)
    """
    # Étape 1 : Détecter les segments de pluie contigus
    rain_segments = detect_rain_blocks(P, config.P_THR_MM)
    if not rain_segments:
        return []
    
    # Étape 2 : Fusionner les blocs proches
    merged_rain = merge_rain_blocks(rain_segments, config.RAIN_GAP_HOURS, dt)
    if not merged_rain:
        return []
    
    # Débit de base global (fallback)
    Q0_global = estimate_global_baseflow(q)
    
    # Étape 3 : Extension temporelle de chaque bloc avec Q0 local
    extended = []
    for s_rain, e_rain in merged_rain:
        i0, i1 = extend_event_with_baseflow(
            P, q, s_rain, e_rain, dt, Q0_global,
            config.PRE_HOURS,
            config.POST_DRY_HOURS,
            config.BASE_PRE_HOURS,
            config.BASEFLOW_MARGIN_LS,
            config.P_THR_MM
        )
        if i1 > i0:
            extended.append((i0, i1))
    
    if not extended:
        return []
    
    # Étape 4 : Fusion des événements qui se chevauchent
    extended.sort(key=lambda x: x[0])
    merged_events = []
    gap_steps = int(config.RAIN_GAP_HOURS * 3600.0 / dt)
    
    for s, e in extended:
        if not merged_events:
            merged_events.append([s, e])
        else:
            last_s, last_e = merged_events[-1]
            if s <= last_e + gap_steps:
                merged_events[-1][1] = max(last_e, e)
            else:
                merged_events.append([s, e])
    
    events = [(s, e) for s, e in merged_events]
    
    # Étape 5 : Filtre amplitude
    events = filter_by_amplitude(events, q, config.MIN_AMP_LS)
    if not events:
        return []
    
    # Étape 6 : Filtre réponse hydrologique
    events = filter_by_response(
        events, P, q, dt,
        config.MIN_DQ_RESPONSE_LS,
        config.MAX_PEAK_DELAY_HOURS,
        config.P_THR_MM
    )
    
    return events


# ======================================================================
# CALCUL DES SIGNATURES VOLUMÉTRIQUES (Section 2.3.2)
# ======================================================================

def compute_event_signatures(
    df: pd.DataFrame,
    events: List[Tuple[int, int]],
    dt: float,
    area_m2: float,
    pre_base_hours: float = Config.BASE_PRE_HOURS,
    antecedent_days: float = Config.ANTECEDENT_DAYS,
    year: Optional[int] = None
) -> pd.DataFrame:
    """
    Calcule les signatures volumétriques pour chaque événement (Section 2.3.2).
    
    Pour chaque événement :
      - hp (mm) : pluie totale sur l'événement
      - hr (mm) : lame de ruissellement excédentaire (au-dessus de Q0)
      - V_excess_m3 : volume excédentaire
      - P_ante_mm : pluie antécédente
      - Q0_Ls : débit de base estimé
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame annuel avec P_mm et Q_ls
    events : List[Tuple[int, int]]
        Liste d'événements
    dt : float
        Pas de temps (secondes)
    area_m2 : float
        Surface du bassin (m²)
    pre_base_hours : float
        Fenêtre pour estimer Q0 (heures)
    antecedent_days : float
        Durée pluie antécédente (jours)
    year : Optional[int]
        Année (pour la colonne de sortie)
        
    Returns
    -------
    pd.DataFrame
        Tableau des signatures avec colonnes :
        year, event_id, i0, i1, hp_mm, hr_mm, P_ante_mm, Q0_Ls, V_excess_m3
    """
    Q_all = pd.to_numeric(df["Q_ls"], errors="coerce").values
    P_all = pd.to_numeric(df["P_mm"], errors="coerce").values
    
    pre_steps = int(pre_base_hours * 3600.0 / dt)
    ante_steps = int(antecedent_days * 24.0 * 3600.0 / dt)
    Q0_global = estimate_global_baseflow(Q_all)
    
    info_rows = []
    
    for idx_evt, (i0, i1) in enumerate(events, start=1):
        P_evt = P_all[i0:i1 + 1]
        Q_evt = Q_all[i0:i1 + 1]
        
        # hp : pluie totale
        hp_mm = float(np.nansum(P_evt))
        
        # Estimation Q0 local
        Q0 = estimate_local_baseflow(Q_all, i0, dt, pre_base_hours, Q0_global)
        
        # Excès de débit (au-dessus de Q0)
        Q_excess = Q_evt - Q0
        Q_excess[~np.isfinite(Q_excess)] = 0.0
        Q_excess[Q_excess < 0.0] = 0.0
        
        # Volume excédentaire (m³)
        V_excess_m3 = Q_excess.sum() * dt / 1000.0
        
        # Lame de ruissellement hr (mm)
        hr_mm = 1000.0 * V_excess_m3 / area_m2
        
        # Pluie antécédente
        i_ante_start = max(0, i0 - ante_steps)
        P_ante_mm = float(np.nansum(P_all[i_ante_start:i0]))
        
        info_rows.append({
            "year": year,
            "event_id": idx_evt,
            "i0": i0,
            "i1": i1,
            "hp_mm": hp_mm,
            "hr_mm": hr_mm,
            "P_ante_mm": P_ante_mm,
            "Q0_Ls": Q0,
            "V_excess_m3": V_excess_m3,
        })
    
    return pd.DataFrame(info_rows)


# ======================================================================
# EXPORT DES ÉVÉNEMENTS
# ======================================================================

def export_events_to_csv(
    df: pd.DataFrame,
    events: List[Tuple[int, int]],
    out_dir: Path,
    year: int
) -> None:
    """
    Export chaque événement dans un fichier CSV individuel.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame annuel complet
    events : List[Tuple[int, int]]
        Liste d'événements
    out_dir : Path
        Répertoire de sortie
    year : int
        Année
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, (i0, i1) in enumerate(events, 1):
        df_evt = df.iloc[i0:i1 + 1]
        df_evt.to_csv(out_dir / f"event_{year}_{k:03d}.csv",
                      sep=";", index=False)


def load_events_from_meta(meta_path: Path) -> Dict[int, List[Tuple[int, int]]]:
    """
    Recharge les événements depuis hr_hp_events.csv.
    
    Parameters
    ----------
    meta_path : Path
        Chemin vers hr_hp_events.csv
        
    Returns
    -------
    Dict[int, List[Tuple[int, int]]]
        Dictionnaire {year: [(i0, i1), ...]}
    """
    dfm = pd.read_csv(meta_path, sep=";")
    events_by_year = {}
    for _, row in dfm.iterrows():
        y = int(row["year"])
        i0 = int(row["i0"])
        i1 = int(row["i1"])
        events_by_year.setdefault(y, []).append((i0, i1))
    
    for y in events_by_year:
        events_by_year[y].sort(key=lambda e: e[0])
    return events_by_year


# ======================================================================
# ENRICHISSEMENT DES SIGNATURES
# ======================================================================

def add_preflow_class(
    info_df: pd.DataFrame,
    col: str = "Q0_Ls"
) -> pd.DataFrame:
    """
    Ajoute une colonne 'preflow_class' selon le débit de base Q0.
    
    Classification :
      - très sec : Q0 < 1 L/s
      - sec : 1-5 L/s
      - humide : 5-15 L/s
      - très humide : >15 L/s
    """
    bins = [0, 1, 5, 15, np.inf]
    labels = [
        "très sec (Q0 < 1 L/s)",
        "sec (1–5 L/s)",
        "humide (5–15 L/s)",
        "très humide (>15 L/s)"
    ]
    info_df = info_df.copy()
    info_df["preflow_class"] = pd.cut(
        info_df[col].astype(float),
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    return info_df


def month_to_hydro_season(month: int) -> str:
    """Convertit un mois (1-12) en saison hydrologique."""
    if month in (12, 1, 2):
        return "Hiver"
    elif month in (3, 4, 5):
        return "Printemps"
    elif month in (6, 7, 8):
        return "Été"
    else:
        return "Automne"


def add_hydro_season(
    info_df: pd.DataFrame,
    df_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Ajoute une colonne 'season' selon le mois de début d'événement.
    """
    info = info_df.copy()
    seasons = []
    
    for _, row in info.iterrows():
        year = int(row["year"])
        i0 = int(row["i0"])
        dfy = df_full[df_full["year"] == year].reset_index(drop=True)
        
        if 0 <= i0 < len(dfy):
            date_evt = dfy.loc[i0, "dateQ"]
            season = month_to_hydro_season(date_evt.month)
        else:
            season = "Inconnu"
        seasons.append(season)
    
    info["season"] = seasons
    return info


def add_event_memory_sequence(
    info_df: pd.DataFrame,
    df_full: pd.DataFrame,
    max_gap_hours: float = Config.MAX_GAP_SEQUENCE_HOURS,
    P_ante_wet_thr: float = Config.P_ANTE_WET_THR
) -> pd.DataFrame:
    """
    Ajoute des indicateurs de 'mémoire hydrologique' entre événements.
    
    Colonnes ajoutées :
      - t_start, t_end, t_peak : datetimes
      - delta_hours_prev_end_to_start : temps depuis fin événement précédent
      - delta_hours_prev_peak_to_peak : temps entre pics
      - sequence_id : ID de séquence humide
      - rank_in_sequence : rang dans la séquence
      - sequence_size : taille de la séquence
      - n_prev_in_sequence : nombre d'événements précédents dans la séquence
      - memory_class : classification qualitative
    
    Parameters
    ----------
    info_df : pd.DataFrame
        Signatures d'événements
    df_full : pd.DataFrame
        DataFrame complet multi-années
    max_gap_hours : float
        Gap max pour rester dans une séquence (heures)
    P_ante_wet_thr : float
        Seuil P_ante pour sol "humide" (mm)
        
    Returns
    -------
    pd.DataFrame
        Signatures enrichies
    """
    info = info_df.copy().reset_index().rename(columns={"index": "orig_index"})
    info = info.sort_values(["year", "i0", "i1"]).reset_index(drop=True)
    
    # DataFrames par année
    years_full = sorted(df_full["year"].unique())
    df_by_year = {
        y: df_full[df_full["year"] == y].reset_index(drop=True)
        for y in years_full
    }
    
    # Extraction des timestamps
    t_start_list = []
    t_end_list = []
    t_peak_list = []
    
    for _, row in info.iterrows():
        y = int(row["year"])
        i0 = int(row["i0"])
        i1 = int(row["i1"])
        dfy = df_by_year[y]
        sub = dfy.iloc[i0:i1 + 1]
        
        t0 = sub["dateQ"].iloc[0]
        t1 = sub["dateQ"].iloc[-1]
        
        Q_evt = pd.to_numeric(sub["Q_ls"], errors="coerce").values
        if np.any(np.isfinite(Q_evt)):
            ip_local = int(np.nanargmax(Q_evt))
        else:
            ip_local = 0
        t_peak = sub["dateQ"].iloc[ip_local]
        
        t_start_list.append(t0)
        t_end_list.append(t1)
        t_peak_list.append(t_peak)
    
    info["t_start"] = pd.to_datetime(t_start_list)
    info["t_end"] = pd.to_datetime(t_end_list)
    info["t_peak"] = pd.to_datetime(t_peak_list)
    
    # Calcul des deltas
    n_evt = len(info)
    delta_prev_end_h = [np.nan] * n_evt
    delta_prev_peak_h = [np.nan] * n_evt
    
    for i in range(1, n_evt):
        dt_end = (info.loc[i, "t_start"] - info.loc[i - 1, "t_end"]).total_seconds() / 3600.0
        dt_peak = (info.loc[i, "t_peak"] - info.loc[i - 1, "t_peak"]).total_seconds() / 3600.0
        delta_prev_end_h[i] = dt_end
        delta_prev_peak_h[i] = dt_peak
    
    info["delta_hours_prev_end_to_start"] = delta_prev_end_h
    info["delta_hours_prev_peak_to_peak"] = delta_prev_peak_h
    
    # Construction des séquences
    sequence_id = np.zeros(n_evt, dtype=int)
    rank_in_seq = np.zeros(n_evt, dtype=int)
    
    current_seq = 1
    sequence_id[0] = current_seq
    rank_in_seq[0] = 1
    
    for i in range(1, n_evt):
        gap = delta_prev_end_h[i]
        if np.isfinite(gap) and (gap <= max_gap_hours):
            sequence_id[i] = current_seq
            rank_in_seq[i] = rank_in_seq[i - 1] + 1
        else:
            current_seq += 1
            sequence_id[i] = current_seq
            rank_in_seq[i] = 1
    
    info["sequence_id"] = sequence_id
    info["rank_in_sequence"] = rank_in_seq
    info["sequence_size"] = info.groupby("sequence_id")["sequence_id"].transform("count")
    info["n_prev_in_sequence"] = info["rank_in_sequence"] - 1
    
    # Classification de mémoire
    mem_classes = []
    for i, row in info.iterrows():
        gap = row["delta_hours_prev_end_to_start"]
        Pante = row.get("P_ante_mm", np.nan)
        
        if not np.isfinite(gap):
            mem_classes.append("début série")
            continue
        
        is_long_gap = gap > max_gap_hours
        is_wet = (np.isfinite(Pante) and (Pante >= P_ante_wet_thr))
        
        if is_long_gap and (not is_wet):
            mem_classes.append("isolé & sec")
        elif is_long_gap and is_wet:
            mem_classes.append("isolé mais sol humide")
        elif (not is_long_gap) and (not is_wet):
            mem_classes.append("enchaîné, peu de Pante")
        else:
            mem_classes.append("enchaîné, Pante élevée")
    
    info["memory_class"] = mem_classes
    
    # Retour à l'ordre d'origine
    info = info.sort_values("orig_index").reset_index(drop=True)
    info = info.drop(columns=["orig_index"])
    
    return info


# ======================================================================
# VISUALISATIONS - ÉVÉNEMENTS INDIVIDUELS
# ======================================================================

def plot_single_event(
    df_evt: pd.DataFrame,
    out_path: Path,
    year: int,
    event_id: int,
    i0: int,
    i1: int
) -> None:
    """
    Trace P et Q pour un événement individuel.
    
    Parameters
    ----------
    df_evt : pd.DataFrame
        Sous-ensemble de données pour l'événement
    out_path : Path
        Chemin de sortie du PNG
    year, event_id : int
        Identifiants
    i0, i1 : int
        Indices de début/fin
    """
    t = df_evt["dateQ"]
    Q = df_evt["Q_ls"].astype(float)
    P = df_evt["P_mm"].astype(float)
    
    Ptot = float(np.nansum(P))
    Qmax = float(np.nanmax(Q)) if np.any(np.isfinite(Q)) else np.nan
    
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    ax1.plot(t, Q, 'b-', label="Q_obs (L/s)")
    ax1.set_ylabel("Débit Q (L/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    ax2 = ax1.twinx()
    ax2.bar(t, P, width=0.002, alpha=0.3, color="orange", label="Pluie (mm/pas)")
    ax2.set_ylabel("Pluie (mm/pas)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    
    # Inverser axe pluie
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymax, ymin)
    
    date_fmt = DateFormatter("%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate()
    
    title = (f"Évènement {event_id:03d} — {year} | "
             f"P_tot={Ptot:.1f} mm | Q_max={Qmax:.1f} L/s "
             f"(indices {i0}-{i1})")
    ax1.set_title(title)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_events(
    df: pd.DataFrame,
    events: List[Tuple[int, int]],
    out_dir: Path,
    year: int
) -> None:
    """
    Génère les plots P-Q pour tous les événements d'une année.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for k, (i0, i1) in enumerate(events, 1):
        df_evt = df.iloc[i0:i1 + 1]
        out_png = out_dir / f"event_{year}_{k:03d}.png"
        plot_single_event(df_evt, out_png, year, k, i0, i1)


# ======================================================================
# VISUALISATIONS GLOBALES - hr vs hp
# ======================================================================

def plot_hr_vs_hp_simple(
    hp_all: np.ndarray,
    hr_all: np.ndarray,
    out_dir: Path
) -> None:
    """Nuage h_r vs h_p simple (sans labels)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(hp_all, hr_all, alpha=0.6, marker='x')
    ax.set_xlabel("h_p (mm) – lame de pluie événementielle")
    ax.set_ylabel("h_r (mm) – lame de ruissellement événementielle")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("h_r = f(h_p)")
    
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_hr_vs_hp_simple.png", dpi=200)
    plt.close(fig)
    print(f"[OK] Plot hr vs hp simple sauvegardé")


def plot_hr_vs_hp_colored_pante(
    info_df: pd.DataFrame,
    out_dir: Path,
    antecedent_days: float
) -> None:
    """Nuage h_r vs h_p coloré par pluie antécédente avec labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    hp = info_df["hp_mm"].astype(float).values
    hr = info_df["hr_mm"].astype(float).values
    P_ante = info_df["P_ante_mm"].astype(float).values
    event_ids = info_df["event_id"].astype(int).values
    years = info_df["year"].astype(int).values
    
    labels = [f"{eid}_{(y % 100):02d}" for eid, y in zip(event_ids, years)]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(hp, hr, c=P_ante, cmap="viridis",
                    alpha=0.8, marker="o", vmin=0, vmax=15)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"Pluie antécédente sur {antecedent_days:.0f} jours (mm)")
    
    for x, y, lab in zip(hp, hr, labels):
        ax.text(x, y, lab, fontsize=7, alpha=0.8)
    
    ax.set_xlabel("h_p (mm) – lame de pluie événementielle")
    ax.set_ylabel("h_r (mm) – lame de ruissellement événementielle")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"h_r = f(h_p) coloré par pluie antécédente ({antecedent_days:.0f}j)")
    
    fig.tight_layout()
    fig.savefig(out_dir / "hr_vs_hp_colored_Pante.png", dpi=200)
    plt.close(fig)
    print(f"[OK] Plot coloré P_ante sauvegardé")


def plot_hr_vs_hp_by_season(
    info_df: pd.DataFrame,
    df_full: pd.DataFrame,
    out_dir: Path
) -> None:
    """4 sous-graphes h_r vs h_p par saison hydrologique."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    info = add_hydro_season(info_df, df_full)
    seasons_order = ["Automne", "Hiver", "Printemps", "Été"]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    
    for ax, season in zip(axes, seasons_order):
        sub = info[info["season"] == season]
        if sub.empty:
            ax.set_title(f"{season} (0 évts)")
            ax.grid(True, linestyle="--", alpha=0.4)
            continue
        
        hp = sub["hp_mm"].astype(float).values
        hr = sub["hr_mm"].astype(float).values
        ids = sub["event_id"].astype(int).values
        years = sub["year"].astype(int).values
        
        labels = [f"{eid}_{(y % 100):02d}" for eid, y in zip(ids, years)]
        
        ax.scatter(hp, hr, alpha=0.7, marker="x")
        for x, yv, lab in zip(hp, hr, labels):
            ax.text(x, yv, lab, fontsize=8, alpha=0.7)
        
        ax.set_title(f"{season} (N={len(sub)})")
        ax.grid(True, linestyle="--", alpha=0.4)
    
    for ax in axes[2:]:
        ax.set_xlabel("h_p (mm)")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("h_r (mm)")
    
    fig.suptitle("h_r = f(h_p) par saison hydrologique", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(out_dir / "hr_vs_hp_by_season.png", dpi=200)
    plt.close(fig)
    print(f"[OK] Plot par saison sauvegardé")


def plot_hr_vs_hp_by_preflow(
    info_df: pd.DataFrame,
    out_dir: Path
) -> None:
    """Nuage h_r vs h_p classé par classe de débit de base (Q0)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    class_values = info_df["preflow_class"].dropna().unique()
    class_values = sorted(class_values, key=lambda x: str(x))
    markers = ["o", "s", "D", "^"]
    
    for cls, marker in zip(class_values, markers):
        sub = info_df[info_df["preflow_class"] == cls]
        hp = sub["hp_mm"].astype(float).values
        hr = sub["hr_mm"].astype(float).values
        ids = sub["event_id"].astype(int).values
        years = sub["year"].astype(int).values
        labels_evt = [f"{eid}_{(y % 100):02d}" for eid, y in zip(ids, years)]
        
        ax.scatter(hp, hr, marker=marker, alpha=0.8, label=str(cls))
        for x, yv, lab in zip(hp, hr, labels_evt):
            ax.text(x, yv, lab, fontsize=7, alpha=0.7)
    
    ax.set_xlabel("h_p (mm)")
    ax.set_ylabel("h_r (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("h_r = f(h_p) classé par état d'écoulement antérieur (Q0)")
    ax.legend(title="Classes de pré-écoulement", loc="upper left")
    
    fig.tight_layout()
    fig.savefig(out_dir / "hr_vs_hp_by_preflow.png", dpi=200)
    plt.close(fig)
    print(f"[OK] Plot par Q0 sauvegardé")


def plot_hr_vs_hp_by_memory(
    info_df: pd.DataFrame,
    out_dir: Path
) -> None:
    """Nuage h_r vs h_p coloré par n_prev_in_sequence (mémoire)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    hp = info_df["hp_mm"].astype(float).values
    hr = info_df["hr_mm"].astype(float).values
    n_prev = info_df["n_prev_in_sequence"].astype(float).values
    years = info_df["year"].astype(int).values
    ev_ids = info_df["event_id"].astype(int).values
    
    labels = [f"{eid}_{(y % 100):02d}" for eid, y in zip(ev_ids, years)]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(hp, hr, c=n_prev, cmap="plasma", alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Nombre d'évènements précédents dans la séquence humide")
    
    for x, yv, lab in zip(hp, hr, labels):
        ax.text(x, yv, lab, fontsize=7, alpha=0.7)
    
    ax.set_xlabel("h_p (mm)")
    ax.set_ylabel("h_r (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("h_r = f(h_p) coloré par la 'mémoire' d'évènements")
    
    fig.tight_layout()
    fig.savefig(out_dir / "hr_vs_hp_colored_memory.png", dpi=200)
    plt.close(fig)
    print(f"[OK] Plot mémoire sauvegardé")


# ======================================================================
# PIPELINE PRINCIPAL
# ======================================================================

def main():
    """
    Pipeline principal d'analyse événementielle du bassin du Cloutasse.
    
    Étapes :
      1. Lecture des données chronologiques complètes
      2. Détection ou rechargement des événements
      3. Calcul des signatures volumétriques (hp, hr, P_ante)
      4. Enrichissement (saison, Q0 class, mémoire)
      5. Export CSV et génération des visualisations
    """
    # Configuration des chemins
    base = Path(__file__).resolve().parent
    data_path = base.parent / "02_Data" / "PQ_BV_Cloutasse_interp.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
    
    # Lecture données complètes
    print("[INFO] Lecture des données...")
    df = pd.read_csv(data_path, sep=";")
    df["dateP"] = pd.to_datetime(df["dateP"])
    df["dateQ"] = pd.to_datetime(df["dateQ"])
    df = df.sort_values("dateQ").reset_index(drop=True)
    
    # Calcul du pas de temps
    dt = float(
        df["dateQ"].sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
        .median()
    )
    print(f"[INFO] Pas de temps détecté : dt = {dt:.1f} s")
    
    # Extraction des années
    df["year"] = df["dateQ"].dt.year
    years = sorted(df["year"].unique())
    print(f"[INFO] Années disponibles : {years}")
    
    # Répertoires de sortie
    all_event_dir = base.parent / "03_Plots" / "Etude_hydrologique"
    all_event_dir.mkdir(parents=True, exist_ok=True)
    meta_path = all_event_dir / "hr_hp_events.csv"
    
    # Configuration
    config = Config()
    
    # Mode de fonctionnement
    if config.USE_EXISTING_EVENTS:
        if not meta_path.exists():
            raise FileNotFoundError(
                f"USE_EXISTING_EVENTS=True mais {meta_path} introuvable.\n"
                "Lance d'abord avec USE_EXISTING_EVENTS=False."
            )
        print("[INFO] Mode POST-TRAITEMENT : rechargement des événements.")
        events_by_year = load_events_from_meta(meta_path)
    else:
        print("[INFO] Mode COMPLET : détection + filtrage des événements.")
        events_by_year = {}
    
    # Traitement par année
    info_all = []
    
    for year in years:
        print(f"\n=== Année {year} ===")
        dfy = df[df["year"] == year].reset_index(drop=True)
        
        if config.USE_EXISTING_EVENTS:
            events = events_by_year.get(year, [])
            print(f"[INFO] Événements rechargés : {len(events)}")
        else:
            P = pd.to_numeric(dfy["P_mm"], errors="coerce").fillna(0.0).values
            Q = pd.to_numeric(dfy["Q_ls"], errors="coerce").fillna(0.0).values
            
            # Détection des événements
            events_raw = detect_scs_events(P, Q, dt, config)
            print(f"[INFO] Événements bruts détectés : {len(events_raw)}")
            
            # Filtrage qualité
            events = filter_clean_events(dfy, events_raw, dt,
                                        config.MIN_P_TOT_MM,
                                        config.MIN_Q_AMP_LS,
                                        config.MAX_DURATION_H)
            print(f"[INFO] Événements propres retenus : {len(events)}")
            events_by_year[year] = events
        
        if not events:
            print(f"[WARN] Aucun événement pour {year}")
            continue
        
        # Export CSV individuels
        data_events_dir = base.parent / "02_Data" / "all_events" / str(year)
        export_events_to_csv(dfy, events, data_events_dir, year)
        print(f"[OK] {len(events)} événements exportés en CSV")
        
        # Plots P-Q individuels
        year_fig_dir = all_event_dir / str(year) / "figures"
        plot_events(dfy, events, year_fig_dir, year)
        print(f"[OK] Plots P-Q sauvegardés dans {year_fig_dir}")
        
        # Calcul des signatures
        info_year = compute_event_signatures(
            dfy, events, dt, config.A_BV_M2,
            config.BASE_PRE_HOURS,
            config.ANTECEDENT_DAYS,
            year
        )
        info_all.append(info_year)
    
    # Consolidation globale
    if not info_all:
        print("[WARN] Aucun événement : fin du traitement.")
        return
    
    print("\n=== Post-traitement global ===")
    info_df = pd.concat(info_all, ignore_index=True)
    
    # Enrichissement
    info_df = add_preflow_class(info_df, col="Q0_Ls")
    info_df = add_event_memory_sequence(
        info_df, df,
        config.MAX_GAP_SEQUENCE_HOURS,
        config.P_ANTE_WET_THR
    )
    
    # Export tableau final
    out_csv = all_event_dir / "hr_hp_events.csv"
    info_df.to_csv(out_csv, sep=";", index=False)
    print(f"[OK] Tableau des signatures sauvegardé : {out_csv}")
    
    # Visualisations globales
    print("\n=== Génération des visualisations globales ===")
    
    hp_all = info_df["hp_mm"].values
    hr_all = info_df["hr_mm"].values
    
    plot_hr_vs_hp_simple(hp_all, hr_all, all_event_dir)
    plot_hr_vs_hp_colored_pante(info_df, all_event_dir, config.ANTECEDENT_DAYS)
    plot_hr_vs_hp_by_season(info_df, df, all_event_dir)
    plot_hr_vs_hp_by_preflow(info_df, all_event_dir)
    plot_hr_vs_hp_by_memory(info_df, all_event_dir)
    
    print("\n[FIN] Traitement terminé avec succès.")


if __name__ == "__main__":
    main()