# Modèles SCS Continus - Guide d'Utilisation

## Vue d'ensemble

Ce document décrit l'utilisation des quatre modèles hydrologiques SCS continus développés dans le cadre du projet de fin d'étude (PFE) sur la modélisation de séries chronologiques continues. Ces modèles constituent une progression contrôlée de complexité, partageant tous un noyau SCS continu commun mais se différenciant par leurs opérateurs de transfert.

### Référence au rapport
Ce README est basé sur le rapport de fin d'étude "Interprétation et développement du modèle SCS pour la modélisation de séries chronologiques continues" (Antoine Sognos, 2025-2026).

---

## Architecture Générale

### Structure des dossiers attendue

   
projet/
├── 01_Code/
│   ├── modele_A.py    # SCS continu sans routage
│   ├── modele_B.py    # + réservoir de routage de surface
│   ├── modele_C.py    # + compartiment lent (voie souterraine)
│   └── modele_D.py    # routage en cascade (spécifique CSR)
├── 02_Data/
│   ├── PQ_BV_Cloutasse.csv        # Données bassin Cloutasse
│   ├── ETP_SAFRAN_J.csv           # ETP journalière (pour Cloutasse)
│   └── all_events1/               # Événements CSR
│       └── 2024/
│           └── event_2024_014.csv
└── 03_Plots/
    ├── Modele_A_SansRoutage/
    ├── Avec Routage/
    ├── Avec Routage & Réservoir Laggé/
    └── Parking_CSR_CASCADE/
   

---

## Les Quatre Modèles : Progression Conceptuelle

### Tableau Récapitulatif (cf. Tableau A.1 du rapport)

| Modèle | Noyau commun | Réservoirs spécifiques | Sortie(s) visée(s) | Paramètres ajoutés |
|--------|--------------|------------------------|--------------------|--------------------|
|  A  | Production SCS continue (ha, hs) | Stock de surface (non routé) | Q (Cloutasse) ou Qruiss (CSR) | Ia, S, k_infiltr, k_seepage |
|  B  | Même noyau SCS | + Routage surface : hr avec r_out = k_runoff × hr | Q (Cloutasse) ou Qruiss (CSR) | + k_runoff |
|  C  | Noyau SCS + Routage B | + Réservoir lent hsub (q_sub = k_sub × hsub) | Q total (rapide + lent) | + k_sub, α_sub |
|  D  | Noyau SCS | Cascade routage : hr1 → hr2 | Qruiss ET Qinf (CSR) | k_runoff1, k_runoff2 |

### Principe Commun : Le Noyau SCS Continu

Tous les modèles partagent la  production SCS continue  :

1. Abstraction initiale : réservoir ha ∈ [0, Ia]
   - Tant que ha < Ia → pas de pluie nette
   - Dès que ha = Ia → excédent devient pluie nette q

2.  Stockage sol  : réservoir hs ∈ [0, S]
   - Infiltration potentielle selon loi HSM : dépend de hs/S
   - Vidange interne : kseepage (drainage profond)

3.  Génération de ruissellement  : rgen = max(q - infil, 0)

La différence entre modèles réside dans  comment rgen est transformé en débit observé .

---

## Modèle A : SCS Continu Sans Routage

### Description

-  Structure la plus simple  : production = transfert
- Aucun temps de transfert explicite
- Q_mod = rgen × A_BV

### Fichier
 modele_A.py 

### Paramètres

 Fixes  (à définir dans le script) :
   python
I_A_FIXED = 0.002  # m (capacité abstraction initiale)
S_FIXED = 0.13     # m (capacité stockage sol)
A_BV_M2 = 94.0     # m² (surface bassin versant)
DT_S = 300.0       # s (pas de temps)
   

 À calibrer  :
-  k_infiltr  (m/s) : capacité d'infiltration
  - Bornes :  KINF_MIN_MM_H  à  KINF_MAX_MM_H  (en mm/h)
-  k_seepage  (s⁻¹) : constante de vidange interne
  - Bornes :  KSEEP_MIN  à  KSEEP_MAX 

### Configuration Événement

   python
# Chemin relatif depuis 02_Data/
EVENT_CSV_REL = "all_events1/2024/event_2024_014.csv"

# Options calibration
DO_CALIBRATION = True
N_STARTS = 30  # Nombre de points de départ multistart
   

### Format CSV Attendu

 Colonnes obligatoires  :
-  date  ou  dateP  (datetime)
-  P_mm  (mm/pas)
-  Q_m3s ,  Q_LH ,  Q_lh ,  Q_LS , ou  Q_ls  (selon unité)

### Utilisation

   bash
cd 01_Scripts/
python modele_A.py
   

### Sorties

 Console  :
- Paramètres optimaux
- J_opt (log-RMSE)
- Bilan de masse
- Bilan volumétrique

 Figures  (dans  03_Plots/Modele_A_SansRoutage/<event_name>/ ) :
1.  Qmod_Qobs_P.png  : Hydrogramme + pluie
2.  Etats_reservoirs.png  : ha, hs, hr
3.  Cumuls_mm.png  : P, ET, infiltration, seepage, ruissellement

### Limites Connues (cf. Section 3.3.1 du rapport)

>  Réponse trop instantanée  : absence de temps de transfert explicite
> - Pics abruptes non réalistes
> - NSE souvent négatives
> - Nécessite ajout routage → voir Modèle B

---

## Modèle B : Avec Réservoir de Routage de Surface

### Description

-  Ajout d'un temps de transfert explicite 
- Réservoir de surface hr
- r_out = k_runoff × hr (loi linéaire)
- Q_mod = r_out × A_BV

### Fichier
 modele_B.py 

### Paramètres Supplémentaires

 À calibrer  (en plus de A) :
-  k_runoff  (s⁻¹) : constante de routage rapide
  - Interprétation : t₁/₂ = ln(2)/k_runoff
  - Ordre de grandeur : quelques minutes à quelques heures

### Configuration

   python
EVENT_CSV_REL = "PQ_BV_Cloutasse.csv"  # Exemple Cloutasse
I_A_FIXED = 0.002
S_FIXED = 0.02  # (à adapter selon bassin)
DO_CALIBRATION = True
N_STARTS = 30
   

### Format CSV Cloutasse

 Colonnes  :
-  dateP  (datetime)
-  P_mm  (mm/pas de 5 min)
-  Q_ls  (L/s) ou autre unité Q

 + ETP (optionnel)  :
   python
USE_ETP = True  # Active lecture ETP_SAFRAN_J.csv
   

### Utilisation

   bash
python modele_B.py
   

### Sorties

 Figures  (dans  03_Plots/Avec Routage/<event_name>/ ) :
1.  cumuls_V_obs_V_mod.png  : Volumes cumulés
2.  Q_mod_vs_Q_obs_P_haut.png  : Hydrogramme + pluie inversée
3.  etats_reservoirs_runoff.png  : ha, hs, hr
4.  cumuls_P_ETP_infil_seep_runoff_event2.png 

### Amélioration par rapport à A

>   Lisse les hydrogrammes  : montée/décrue plus réalistes
> ️ Mais volume peut rester imparfait (cf. Fig. 3.4 rapport)

---

## Modèle C : Avec Compartiment Lent (Écoulement Subsurfacique)

### Description

-  Séparation rapide/lent explicite 
- Réservoir lent hsub alimenté par fraction α_sub de l'infiltration
- q_sub = k_sub × hsub
- Q_mod = (r_out + q_sub) × A_BV

### Fichier
 modele_C.py 

### Paramètres Supplémentaires

 À calibrer  (en plus de B) :
-  k_sub  (s⁻¹) : constante compartiment lent
-  α_sub  ∈ [0, 1] : fraction infiltration → voie lente

### Chemin de l'Eau (cf. Section 2.2.3 du rapport)

   
Pluie nette q
    ↓
Infiltration totale (infil)
    ├─→ α_sub × infil → hsub → q_sub → exutoire
    └─→ (1 - α_sub) × infil → hs → seepage profond

Excédent non infiltré → r_gen → hr → r_out → exutoire

Q_total = r_out + q_sub
   

### Configuration

   python
EVENT_CSV_REL = "PQ_BV_Cloutasse.csv"
I_A_FIXED = 0.002
S_FIXED = 0.02
DO_CALIBRATION = True
N_STARTS = 30

# Bornes k_sub (ordre de grandeur intermédiaire)
KSUB_MIN = 1e-7
KSUB_MAX = 1e-3
   

### Utilisation

   bash
python modele_C.py
   

### Sorties

 Figures  (dans  03_Plots/Avec Routage & Réservoir Laggé/<event_name>/ ) :
1.  cumuls_V_obs_V_mod.png 
2.  Q_mod_vs_Q_obs_P_runoff_top.png 
3.  etats_reservoirs_runoff_baseflow.png  : ha, hs, hr,  hsub 
4.  cumuls_P_ETP_infil_seep_runoff_event.png 

### Limite (cf. Section 3.3.1 du rapport)

>  Problème d'identifiabilité  : sans observation de flux interne (Qinf), 
> compensation possible entre k_runoff, k_sub et α_sub
> → Préférer Modèle D sur site CSR

---

## Modèle D : Routage en Cascade (Spécifique CSR)

### Description

-  Conçu pour site CSR  : deux flux observés (Qruiss, Qinf)
- Cascade routage rapide : hr1 → hr2
- Calibration hiérarchisée en 2 étapes (cf. Section 2.5.6 du rapport)

### Fichier
 modele_D.py 

### Paramètres

 À calibrer  :
- k_infiltr  (m/s)
- k_runoff1  (s⁻¹) : première cascade (très rapide)
- k_runoff2  (s⁻¹) : seconde cascade (queue décrue)
- k_seepage  (s⁻¹) : drainage interne

### Particularité : Substepping

python
DT_INTERNAL = 30.0  # s (pas interne pour stabilité)
# Si dt_obs = 120 s → 4 sous-pas internes


### Configuration

   python
EVENT_CSV_REL = "all_events1/2024/event_2024_014.csv"
A_BV_M2 = 94.0       # CSR parking
I_A_FIXED = 0.002
S_FIXED = 0.13
DT_INTERNAL = 30.0
DO_CALIBRATION = True
N_STARTS = 30
INFIL_FROM_SURFACE = True  # Infiltration peut puiser dans hr1
   

### Format CSV CSR

 Colonnes obligatoires  :
-  date  (format : YYYY-MM-DD HH:MM:SS)
-  P_mm  (mm/pas de 2 min)
-  Q_ruiss_LH  (L/h) : débit ruisselé
-  Q_inf_LH  (L/h) : débit drainé interne

### Utilisation

   bash
python modele_D.py
   

### Sorties

 Console  :
- Paramètres optimaux avec t₁/₂ en minutes/heures
- Bilan volumétrique séparé Qruiss / Qinf
- Bilan de masse

 Figures  (dans  03_Plots/Parking_CSR_CASCADE/<event_name>/ ) :
1.  cumuls_V_obs_V_mod.png 
2.  Qruiss_obs_vs_mod_LH_P.png 
3.  Qinf_obs_vs_mod_LH_P.png  (comparaison drainage)
4.  etats_reservoirs.png  : ha, hs, hr1, hr2
5.  cumuls_mm.png 

### Avantages (cf. Section 3.3.2 du rapport)

>   Cascade  : reproduit simultanément pic rapide + queue étalée
>   Contrainte Qinf  : réduit équifinalité, améliore identiabilité
>   Meilleure restitution dynamique  sur CSR

---

## Calibration : Aspects Communs

### Méthode

 Algorithme  : Multi-start + Powell (scipy.optimize.minimize)
- Espace des paramètres :  log10  pour explorer ordres de grandeur
- N_STARTS répétitions depuis points aléatoires
- Méthode de Powell (sans gradient)

### Fonction Objectif

 Selon le site  (cf. Section 2.4 du rapport) :

1.  Cloutasse  (modèles A, B, C) :
      
   J = RMSE(log(Q_mod + ε), log(Q_obs + ε))
      
   Raison : forte variabilité ordres de grandeur

2.  CSR  (modèle D) :
      
   J = RMSE(Q_ruiss_mod, Q_ruiss_obs)  [en L/h]
      
   Raison : ordres de grandeur plus resserrés

### Bornes Recommandées

 D'après signatures observées  (cf. Sections 3.1.2 et 3.2.2 du rapport) :

| Paramètre | Site | Ordre de grandeur | Interprétation |
|-----------|------|-------------------|----------------|
| k_infiltr | CSR | 0.2–5 mm/h | Capacité infiltration |
| k_seepage | Cloutasse | 10⁻⁷–10⁻⁴ s⁻¹ | t₁/₂ ~ heures à jours |
| k_seepage | CSR | 10⁻⁵–10⁻⁴ s⁻¹ | t₁/₂ ~ quelques heures |
| k_runoff | Cloutasse | 10⁻⁵–10⁻³ s⁻¹ | t₁/₂ ~ 10 min à 3h |
| k_runoff1/2 | CSR | 10⁻³–10⁻² s⁻¹ | t₁/₂ ~ 3–14 min |

---

## Signatures Hydrologiques : Utilisation pour Bornage

### Constantes de Vidange Issues des Décrues

 Principe  (cf. Section 2.3.3 du rapport) :
- Détection segments décrue en période sèche
- Ajustement linéaire : ln(Q) = a + bt → k = -b
- Filtrage R² > 0.8

 Résultats Cloutasse  (Tableau A.2) :
-  182/185 décrues  : un seul régime → k ∈ [1.9×10⁻⁶, 3.3×10⁻⁴] s⁻¹
-  3/185 décrues  : double régime → k_rapide ∈ [6.3×10⁻⁵, 7.0×10⁻⁴] s⁻¹

 Résultats CSR  (Tableau A.3) :
-  Qruiss  (51 segments) : k_runoff ∈ [8.3×10⁻⁴, 4.0×10⁻³] s⁻¹
-  Qinf  (272 segments) : k_seepage ∈ [1.3×10⁻⁵, 8.3×10⁻⁴] s⁻¹

### Relation Événementielle hp vs hr

 Usage  : diagnostic global efficacité pluie→ruissellement
- Dispersion importante = variabilité conditions antécédentes
- Pas de relation fonctionnelle unique (cf. Fig. 3.1 et A.1)

---

## Validation Croisée Temporelle (CSR)

### Procédure (cf. Section 2.5.4 du rapport)

 Année hydrologique 2022-2023  (01/10/2022 → 30/09/2023)
-  H1  : 01/10/2022 → 31/03/2023
-  H2  : 01/04/2023 → 30/09/2023

 Deux configurations  :
1. Calage H1 → Validation H2
2. Calage H2 → Validation H1

 Warm-up  : 1 mois au début de chaque période de calage
- Exclus de la fonction objectif
- Neutralise arbitraire conditions initiales

 Continuité  : état final calage = état initial validation

### Résultats (Modèle D, Tableau A.4)

 Paramètres stables  :
- k_runoff1 : ~2.5 min
- k_seepage : ~4h

 Paramètres sensibles  :
- k_infiltr : varie selon saison
- k_runoff2 : varie selon saison

>   Interprétation  (Section 3.4.2) : non-stationnarité saisonnière
> Le modèle prédit correctement  quand  l'eau s'écoule, mais reste
> sensible au contexte pour prédire  combien  s'écoule.

---

## Diagnostic des Performances

### Métriques Utilisées

 NSE  (Nash-Sutcliffe Efficiency) :
   
NSE = 1 - Σ(Q_obs - Q_mod)² / Σ(Q_obs - Q̄_obs)²
   
- Très sensible au timing et aux pics
- NSE < 0 : modèle pire que moyenne

 KGE  (Kling-Gupta Efficiency) :
- Combine corrélation, biais, variabilité
- Moins pénalisant que NSE sur décalages temporels

 RMSE  :
- Version linéaire (CSR) ou log (Cloutasse)

### Lecture des Résultats

 Modèle A  (Fig. 3.6 rapport) :
- NSE mensuelles : moyenne ~-0.92 (très dégradées)
- KGE mensuelles : médiane ~0.19 (positives mais faibles)
- Bilan volumique : surestimation systématique

 Modèle D  (Fig. 3.13 rapport) :
- NSE/KGE stables entre calage et validation
- Biais volumique en validation (sous-estimation)
- Dynamique temporelle préservée

---
     
## Exemples d'Utilisation

### Exemple 1 : Événement CSR avec Modèle D

   python
# Dans modele_D.py, configurer :
EVENT_CSV_REL = "all_events1/2024/event_2024_014.csv"
A_BV_M2 = 94.0
I_A_FIXED = 0.002
S_FIXED = 0.13
DT_INTERNAL = 30.0
DO_CALIBRATION = True
N_STARTS = 30

# Bornes adaptées CSR (Section 3.2.2 rapport)
KINF_MIN_MM_H = 0.2
KINF_MAX_MM_H = 5.0
KSEEP_INIT = 1.0e-5  # t1/2 ~ 4h
K_RUNOFF1_INIT = 2.5e-3  # t1/2 ~ 4.6 min
K_RUNOFF2_INIT = 5.0e-4  # t1/2 ~ 23 min

# Exécuter
python modele_D.py
   

 Résultat attendu  :
- RMSE_opt faible (< 10 L/h typiquement)
- Volumes Qruiss bien reproduits
- Qinf cohérent (ordre de grandeur heures)

### Exemple 2 : Bassin Cloutasse avec Modèle B

   python
# Dans modele_B.py :
EVENT_CSV_REL = "PQ_BV_Cloutasse.csv"
A_BV_M2 = 810000.0  # 0.81 km² = 810000 m²
I_A_FIXED = 0.002
S_FIXED = 0.07  # À adapter selon analyse sol
USE_ETP = True  # (mais vérifier implémentation lecture)
DO_CALIBRATION = True
N_STARTS = 30

# Bornes (Section 3.1.2 rapport)
KINF_MIN_MM_H = 0.5
KINF_MAX_MM_H = 10.0
KSEEP_MIN = 1e-7
KSEEP_MAX = 1e-4
KRUNOFF_MIN = 1e-6
KRUNOFF_MAX = 1e-3

python modele_B.py
   

 Attention  :
- Vérifier colonne débit :  Q_ls  (L/s)
- NSE peut être négative (réponse naturelle variable)
- KGE souvent plus informatif

---

## Interprétation Physique des Paramètres

### Paramètres de Stockage

| Paramètre | Signification | Ordre de grandeur | Site |
|-----------|---------------|-------------------|------|
|  Ia  | Abstraction initiale (interception, dépressions) | 0.002–0.005 m | Universel |
|  S  | Capacité rétention sol | 0.02–0.13 m | Selon lithologie |

### Constantes de Vidange

 Relation demi-vie  :
   
t₁/₂ = ln(2) / k ≈ 0.693 / k
   

| k (s⁻¹) | t₁/₂ | Processus typique |
|---------|------|-------------------|
| 10⁻² | ~1 min | Routage très rapide (CSR) |
| 10⁻³ | ~10 min | Routage rapide surface |
| 10⁻⁴ | ~2 h | Transfert subsurface lent |
| 10⁻⁵ | ~20 h | Drainage interne |
| 10⁻⁶ | ~8 jours | Vidange profonde |

### Capacité d'Infiltration

   
k_infiltr [m/s] × 3600 × 1000 = capacité [mm/h]
   

 Exemples  :
- k_infiltr = 1.0×10⁻⁶ m/s → 3.6 mm/h (sol argileux)
- k_infiltr = 1.0×10⁻⁵ m/s → 36 mm/h (sol limoneux)
- k_infiltr = 1.0×10⁻⁴ m/s → 360 mm/h (sol sableux)

---

## Bilan de Masse : Vérification

### Principe (tous modèles)

 Équation de continuité  :
   
P_tot = ET_tot + Infil_tot + Seep_tot + Runoff_tot + ΔStock + ε
   

Où :
-  P_tot  : pluie cumulée
-  ET_tot  : évapotranspiration effective (sur Ia)
-  Infil_tot  : infiltration dans sol
-  Seep_tot  : percolation profonde
-  Runoff_tot  : ruissellement à l'exutoire
-  ΔStock  : variation stockage (Ia + sol + surface)
-  ε  : erreur de fermeture (doit être < 1% de P_tot)

### Affichage Console

   
=== BILAN DE MASSE ===
P_tot                  = 25.34 mm
Ruissellement généré    = 10.12 mm 
Seepage profond         = 8.45 mm
ETP sur Ia              = 1.23 mm
ΔStock (Ia+sol+surf)    = 5.52 mm
Erreur fermeture         = 0.0023 mm (0.009 %)
   
