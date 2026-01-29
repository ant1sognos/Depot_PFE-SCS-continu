# Modélisation Hydrologique Continue - Site CSR Lyon
## Guide d'Utilisation

---

## Vue d'Ensemble

Ce projet développe une chaîne de modélisation hydrologique continue pour le site CSR Lyon (OTHU), dans le cadre du projet de fin d'étude "Interprétation et développement du modèle SCS pour la modélisation de séries chronologiques continues" (Antoine Sognos, 2025-2026).

### Composantes du Projet

  Pré-traitement des données   :
- Découpage événementiel de la série continue
- Extraction des constantes de vidange (k_seepage, k_runoff)
- Analyse des vitesses d'infiltration

  Modélisation   :
- Modèle A : SCS sans routage
- Modèle B : SCS avec routage rapide
- Modèle D : SCS avec cascade de routage (spécifique CSR)

  Simulation et analyse   :
- Simulation annuelle continue avec validation croisée temporelle
- Analyse des performances événementielles et mensuelles

### Site d'Application : CSR Lyon (OTHU)

- Type : parking expérimental urbain
- Surface : 94 m²
- Instrumentation : Pluie + Q_ruiss (ruissellement) + Q_inf (drainage interne)
- Période de données : 2022-2024
- Particularité : observation de deux flux simultanés permettant une meilleure contrainte du modèle

## Architecture du Projet

### Structure des Dossiers

   
CSR_Lyon/
├── 01_Code/
│   ├── modele_A.py                              # SCS sans routage
│   ├── modele_B.py                              # SCS + routage rapide
│   ├── modele_D.py                              # SCS + cascade (CSR spécifique)
│   │
│   ├── decoupe_event_plot_cumul_vs_pluie.py    # Découpage événementiel
│   ├── analyse_infiltration_velocity_from_events.py
│   │
│   ├── decoupage_calage_k_seepage_csr.py       # Extraction k_seepage
│   ├── decoupage_calage_k_runn_off_csr.py      # Extraction k_runoff
│   │
│   ├── run_annuel_modele_D.py                  # Simulation annuelle
│   └── analyse_annuelle_modele_D.py            # Analyse post-simulation
│
├── 02_Data/
│   ├── Donnees_serie_complete_2022-2024_corrigee_AS.csv
│   └── all_events1/
│       ├── hr_hp_events.csv
│       └── YYYY/event_YYYY_NNN.csv
│
└── 03_Plots/
    ├── Modele_A_SansRoutage/
    ├── Avec Routage/                           # Modèle B
    ├── Parking_CSR_CASCADE/                    # Modèle D
    ├── calib_H1_valid_H2/                      # Simulation annuelle
    ├── Etude_hydrologique/
    └── Identification constantes vidange CSR/
   

---

## PARTIE 1 : PRÉ-TRAITEMENT DES DONNÉES

### 1.1 Découpage Événementiel

  Script   :  decoupe_event_plot_cumul_vs_pluie.py 

  Objectif   : Extraire des événements pluie-ruissellement depuis la série continue.

  Critères de sélection   :
- Début : P=0, Q_ruiss=0, Q_inf=0 pendant 60 min minimum
- Fin : retour à Q_ruiss=0 et Q_inf=0 pendant 60 min minimum
- Séparation : 1h minimum entre événements

  Paramètres principaux   :
   python
Q0_RUISS_LH = 0.8       # Seuil débit nul ruissellement (L/h)
Q0_INF_LH = 0.8         # Seuil débit nul infiltration (L/h)
START_ZERO_MIN = 60.0   # Durée conditions initiales nulles (min)
END_ZERO_MIN = 60.0     # Durée retour à l'état de repos (min)
MIN_HP_MM = 0.05        # Hauteur de pluie minimale
   

  Entrée   :
   
02_Data/Donnees_serie_complete_2022-2024_corrigee_AS.csv
Colonnes : Date ; Hauteur_de_pluie_mm ; Q_inf_LH ; Q_ruiss_LH
   

  Sorties   :
   
02_Data/all_events1/YYYY/event_YYYY_NNN.csv
02_Data/all_events1/hr_hp_events.csv
03_Plots/Etude_hydrologique/YYYY/event_ .png
   

  Utilisation   :
   bash
cd 01_Code/
python decoupe_event_plot_cumul_vs_pluie.py
   

---

### 1.2 Extraction des Constantes de Vidange

Ces analyses permettent d'obtenir des bornes réalistes pour la calibration des modèles.

#### 1.2.1 Constante de Drainage (k_seepage)

  Script   :  decoupage_calage_k_seepage_csr.py 

  Principe   :
- Détection de segments de décrue de Q_inf en période sèche
- Régression linéaire : ln(Q_inf) = a + bt
- Extraction : k_seepage = -b (s⁻¹)

  Paramètres   :
   python
Q_TARGET_COL = "Q_inf_LH"
rain_thr_mm = 0.1       # Seuil pluie nulle
q_min_LH = 5.0          # Seuil débit minimal
min_pts = 15            # Longueur minimale régression
r2_min = 0.9            # Qualité minimale R²
   

  Sorties   :
   
03_Plots/Identification constantes vidange CSR/k_seepage/
    recessions_simple_events.xlsx
    QA/ .png
   

  Résultats CSR   :
- N = 272 segments analysés
- k_seepage ∈ [1.3×10⁻⁵, 8.3×10⁻⁴] s⁻¹
- Médiane ≈ 4.8×10⁻⁵ s⁻¹ (t₁/₂ ≈ 4h)

---

#### 1.2.2 Constante de Routage (k_runoff)

  Script   :  decoupage_calage_k_runn_off_csr.py 

  Principe   :
- Détection de décrues de Q_ruiss avec retour à zéro stable
- Régression : ln(Q_ruiss) = a + bt
- Extraction : k_runoff = -b (s⁻¹)

  Critères   :
   python
RAIN_THRESH_MM = 0.0            # Strictement sec
Q_ZERO_THRESH_M3S = 2e-7        # Seuil Q~0
N_ZERO_END = 5                  # Points consécutifs ~0
ALLOW_SMALL_BUMPS = True        # Tolère petites variations
   

  Sorties   :
   
03_Plots/Identification constantes runoff CSR/
    recessions_onek_from_events.xlsx
    Recessions_QA/ .png
   

  Résultats CSR   :
- N = 51 segments analysés
- k_runoff ∈ [8.3×10⁻⁴, 4.0×10⁻³] s⁻¹
- Médiane ≈ 2.8×10⁻³ s⁻¹ (t₁/₂ ≈ 4 min)

---

### 1.3 Analyse des Vitesses d'Infiltration

  Script   :  analyse_infiltration_velocity_from_events.py 

  Objectif   : Calculer v_inf(t) = Q_inf(t) / A [mm/h] pour chaque événement.

  Métriques calculées   :
- Statistiques : moyenne, médiane, p95, maximum
- Temps actif d'infiltration
- Relations avec hauteur de pluie et débit de ruissellement

  Configuration   :
   python
A_SITE_M2 = 94.0
CLIP_QINF_NEGATIVE = True
TOP_N = 30                  # Nombre d'événements tracés
   

  Sorties   :
   
03_Plots/Etude_hydrologique/INFILTRATION_VELOCITY/
    infiltration_velocity_events.xlsx
     .png (distributions, relations)
    EVENTS/ .png (courbes individuelles)
   

---

## PARTIE 2 : MODÈLES HYDROLOGIQUES

### Vue d'Ensemble

| Modèle | Structure | Sorties | Paramètres |
|--------|-----------|---------|------------|
| A | SCS sans routage | Q_ruiss | k_infiltr, k_seepage |
| B | SCS + routage rapide | Q_ruiss | k_infiltr, k_seepage, k_runoff |
| D | SCS + cascade (hr1→hr2) | Q_ruiss + Q_inf | k_infiltr, k_seepage, k_runoff1, k_runoff2 |

### Noyau SCS Commun

  Réservoirs   :
1.   ha   : Abstraction initiale ∈ [0, Ia]
2.   hs   : Stockage sol ∈ [0, S]
3.   Génération ruissellement   : rgen = max(q - infil, 0)

  Processus   :
- Infiltration : infil = k_infiltr × (1 - hs/S)²
- Drainage : seep = k_seepage × hs
- Pluie nette : q disponible après remplissage ha

---

### 2.1 Modèle A : Sans Routage

  Fichier   :  modele_A.py 

  Principe   : Production = Transfert (pas de temps de transfert)
- Q_mod = rgen × A_BV

  Paramètres fixes   :
   python
I_A_FIXED = 0.002      # m
S_FIXED = 0.13         # m
A_BV_M2 = 94.0         # m²
DT_S = 300.0           # s
   

  Paramètres calibrés   :
- k_infiltr [m/s] : bornes 0.5-10 mm/h
- k_seepage [s⁻¹] : bornes 10⁻⁷-10⁻⁴

  Sorties   :
   
03_Plots/Modele_A_SansRoutage/<event>/
    Qmod_Qobs_P.png
    Etats_reservoirs.png
    Cumuls_mm.png
   

  Limite   : Réponse instantanée, pics abruptes, NSE souvent négatives.

---

### 2.2 Modèle B : Avec Routage

  Fichier   :  modele_B.py 

  Ajout   : Réservoir de routage hr
- r_out = k_runoff × hr
- Q_mod = r_out × A_BV

  Paramètre supplémentaire   :
- k_runoff [s⁻¹] : constante de routage (t₁/₂ = ln(2)/k_runoff)

  Sorties   :
   
03_Plots/Avec Routage/<event>/
    cumuls_V_obs_V_mod.png
    Q_mod_vs_Q_obs_P_haut.png
    etats_reservoirs_runoff.png
   

  Amélioration   : Lissage des hydrogrammes, montées/décrues plus réalistes.

---

### 2.3 Modèle D : Cascade (Spécifique CSR)

  Fichier   :  modele_D.py 

  Architecture   :
   
Pluie nette q → Infiltration → hs → seep = Q_inf
              → Ruissellement → hr1 → hr2 → Q_ruiss
   

  Cascade de routage   :
- hr1 : routage rapide (k_runoff1, t₁/₂ ~ 3-5 min)
- hr2 : queue de décrue (k_runoff2, t₁/₂ ~ 15-30 min)

  Particularité : Substepping   :
   python
DT_INTERNAL = 30.0     # s (pas interne)
   
Si dt_obs = 120 s, utilise 4 sous-pas pour la stabilité numérique.

  Paramètres   :
- k_infiltr [m/s]
- k_seepage [s⁻¹]
- k_runoff1 [s⁻¹]
- k_runoff2 [s⁻¹]

  Bornes recommandées   :
   python
KINF_MIN_MM_H = 0.2, KINF_MAX_MM_H = 5.0
KSEEP_MIN = 1e-5, KSEEP_MAX = 1e-4
K_RUNOFF1_MIN = 1e-3, K_RUNOFF1_MAX = 5e-3
K_RUNOFF2_MIN = 1e-4, K_RUNOFF2_MAX = 1e-3
   

  Format CSV   :
   
date ; P_mm ; Q_ruiss_LH ; Q_inf_LH
   

  Sorties   :
   
03_Plots/Parking_CSR_CASCADE/<event>/
    cumuls_V_obs_V_mod.png
    Qruiss_obs_vs_mod_LH_P.png
    Qinf_obs_vs_mod_LH_P.png
    etats_reservoirs.png
    cumuls_mm.png
   

  Avantage   : La contrainte simultanée sur Q_ruiss et Q_inf réduit l'équifinalité des paramètres.

---

## PARTIE 3 : SIMULATION ANNUELLE ET VALIDATION

### 3.1 Simulation Continue

  Script   :  run_annuel_modele_D.py 

  Période   : Année hydrologique 2022-10-01 → 2023-09-30

  Validation croisée temporelle   :
- H1 : 01/10/2022 → 31/03/2023 (semestre hivernal)
- H2 : 01/04/2023 → 30/09/2023 (semestre estival)

  Deux configurations   :
1. Calage H1 → Validation H2
2. Calage H2 → Validation H1

  Warm-up   :
- 1 mois au début de chaque période de calage
- Exclu de la fonction objectif
- Neutralise les conditions initiales arbitraires

  Calibration hiérarchisée (2 étapes)   :

Étape 1 : Calage Q_inf
- Paramètres : k_infiltr + k_seepage
- Objectif : erreur volumique relative + log-RMSE forme

Étape 2 : Calage Q_ruiss
- Paramètres : k_runoff1 + k_runoff2
- Objectif : RMSE(Q_ruiss) + pénalité forme Q_inf

  Configuration   :
   python
CONTINUOUS_CSV_REL = "Donnees_serie_complete_2022-2024_corrigee_AS.csv"
YEAR_START = "2022-10-01"
YEAR_END = "2023-09-30"
WARMUP_MONTHS = 1
   

  Sorties   :
   
03_Plots/<split_name>/
    simulation.csv          # Série complète (date, period, P, Q_obs, Q_mod, états)
    params.xlsx            # Paramètres optimaux
    metrics_periods.xlsx   # NSE, KGE par période
    metrics_monthly.xlsx   # Bilans mensuels
   

  Temps de calcul   : 10-30 min (parallélisation ProcessPoolExecutor)

---

### 3.2 Analyse Post-Simulation

  Script   :  analyse_annuelle_modele_D.py 

  Workflow   :
1. Lecture de simulation.csv
2. Chargement métadonnées événements
3. Calcul métriques par événement
4. Génération plots (optionnel)

  Configuration   :
   python
SPLIT_NAME = "calib_H1_valid_H2"
PLOT_MODE = "topN"
TOP_N = 25
   

  Métriques calculées   :
- NSE (Nash-Sutcliffe)
- KGE (Kling-Gupta)
- RMSE
- Biais volumique
- Volumes cumulés (P, Q_ruiss, Q_inf)

  Sorties   :
   
03_Plots/<split_name>/
    metrics_events.xlsx
    EVENTS_PNG/<event_id>.png
   

---

## PARTIE 4 : CALIBRATION

### 4.1 Méthode

  Algorithme   : Multi-start + Powell (scipy.optimize.minimize)
- Espace log10 pour explorer les ordres de grandeur
- N_STARTS répétitions depuis points aléatoires
- Optimisation sans gradient

  Fonction objectif   :
   
CSR (Modèle D) : RMSE(Q_ruiss_mod, Q_ruiss_obs) en L/h
   

### 4.2 Bornes Recommandées

| Paramètre | Ordre de grandeur | t₁/₂ | Processus |
|-----------|-------------------|------|-----------|
| k_infiltr | 0.2-5 mm/h | - | Capacité infiltration |
| k_seepage | 10⁻⁵-10⁻⁴ s⁻¹ | 2-20 h | Drainage interne |
| k_runoff1 | 10⁻³-5×10⁻³ s⁻¹ | 3-14 min | Cascade rapide |
| k_runoff2 | 10⁻⁴-10⁻³ s⁻¹ | 15-30 min | Queue décrue |

  Formule demi-vie   : t₁/₂ = ln(2) / k ≈ 0.693 / k

### 4.3 Métriques de Performance

  NSE   (Nash-Sutcliffe) :
   
NSE = 1 - Σ(Q_obs - Q_mod)² / Σ(Q_obs - Q̄_obs)²
   
- Sensible au timing et aux pics
- NSE < 0 : modèle moins performant que la moyenne
- NSE > 0.5 : acceptable ; NSE > 0.7 : bon

  KGE   (Kling-Gupta) :
   
KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
   
- r = corrélation, α = variabilité, β = biais
- Moins sensible aux décalages temporels
- KGE > 0 : acceptable ; KGE > 0.5 : bon

---

## PARTIE 5 : INTERPRÉTATION PHYSIQUE

### 5.1 Paramètres de Stockage

| Paramètre | Signification | Valeur CSR |
|-----------|---------------|------------|
| Ia | Abstraction initiale | 0.002 m |
| S | Capacité rétention sol | 0.13 m |

### 5.2 Capacité d'Infiltration

  Conversion   : k_infiltr [m/s] × 3.6×10⁶ = capacité [mm/h]

  Exemples   :
- 1.0×10⁻⁶ m/s → 3.6 mm/h (argileux)
- 1.0×10⁻⁵ m/s → 36 mm/h (limoneux)
- 1.0×10⁻⁴ m/s → 360 mm/h (sableux)

### 5.3 Constantes de Vidange

| k [s⁻¹] | t₁/₂ | Processus |
|---------|------|-----------|
| 10⁻² | ~1 min | Routage très rapide |
| 10⁻³ | ~10 min | Routage rapide |
| 10⁻⁴ | ~2 h | Transfert subsurface |
| 10⁻⁵ | ~20 h | Drainage interne |

### 5.4 Bilan de Masse

  Équation   :
   
P_tot = ET_tot + Infil_tot + Seep_tot + Runoff_tot + ΔStock + ε
   

  Critère de validité   : ε < 1% de P_tot

  Exemple de sortie   :
   
=== BILAN DE MASSE ===
P_tot                  = 25.34 mm
Ruissellement généré   = 10.12 mm (39.9%)
Seepage profond        = 8.45 mm (33.3%)
ETP sur Ia             = 1.23 mm (4.9%)
ΔStock                 = 5.52 mm (21.8%)
Erreur fermeture       = 0.0023 mm (0.009%)
   

---

## PARTIE 6 : WORKFLOW TYPE

### Analyse Complète CSR

  Étape 1 : Découpage événementiel  
   bash
python decoupe_event_plot_cumul_vs_pluie.py
   

  Étape 2 : Extraction signatures (optionnel mais recommandé)  
   bash
python decoupage_calage_k_seepage_csr.py
python decoupage_calage_k_runn_off_csr.py
   

  Étape 3 : Modélisation événement  
   bash
# Éditer modele_D.py : EVENT_CSV_REL, bornes
python modele_D.py
   

  Étape 4 : Simulation annuelle  
   bash
# Éditer run_annuel_modele_D.py : SPLIT_CONFIGS, bornes
python run_annuel_modele_D.py
   

  Étape 5 : Analyse  
   bash
python analyse_annuelle_modele_D.py
   

### Paramètres Typiques CSR

   python
# Configuration site
A_BV_M2 = 94.0
I_A_FIXED = 0.002
S_FIXED = 0.13
DT_INTERNAL = 30.0

# Valeurs observées (médiane)
k_infiltr ≈ 0.5-2 mm/h
k_seepage ≈ 4.8×10⁻⁵ s⁻¹ (t₁/₂ ~ 4h)
k_runoff1 ≈ 4.5×10⁻³ s⁻¹ (t₁/₂ ~ 3 min)
k_runoff2 ≈ 5×10⁻⁴ s⁻¹ (t₁/₂ ~ 23 min)
   

---

## PARTIE 7 : LIMITATIONS ET HYPOTHÈSES

### 7.1 Limitations Méthodologiques

  Modèle A   :
- Pas de temps de transfert
- Pics instantanés non réalistes
- Utilisation recommandée pour tests conceptuels uniquement

  Modèle B   :
- Volume parfois imparfait
- Sensible aux conditions initiales

  Modèle D   :
- Variabilité saisonnière des paramètres observée en validation croisée
- k_infiltr et k_runoff2 montrent une sensibilité au contexte saisonnier

### 7.2 Hypothèses du Modèle

  Noyau SCS   :
- Loi HSM pour infiltration (quadratique)
- Seepage proportionnel à hs (linéaire)

  Routage   :
- Lois linéaires (k × h)
- Approche globale (pas de routage distribué)

  ETP   :
- Application sur ha uniquement
- Impact limité sur événements courts

  Stabilité numérique   :
- dt = 120 s (observations CSR)
- Substepping dt = 30 s (Modèle D)

---

## PARTIE 8 : DÉPENDANCES


  Version   : Python ≥ 3.8

  Librairies   :
numpy >= 1.21
pandas >= 1.3
scipy >= 1.7
matplotlib >= 3.4
openpyxl >= 3.0
   


