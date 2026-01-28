# Modélisation hydrologique continue - Bassin versant du Cloutasse

## Contexte

Ce dépôt contient les scripts Python développés dans le cadre d'un projet de fin d'études en ingénierie hydrologique (Polytech Montpellier, 2024-2025). L'objectif est d'évaluer des extensions continues du modèle SCS (Soil Conservation Service) pour la modélisation pluie-ruissellement sur chroniques longues à pas de temps fin.

Le travail porte sur le bassin versant naturel du Cloutasse (Mont-Lozère, 0.81 km²).

L'analyse complète est disponible dans le rapport `AS_PFE_SCS.pdf` présent à la racine du dépôt.

## Structure du dépôt

```
.
├── 01_Scripts/              Scripts Python
├── 02_Data/                 Données d'entrée (incluses)
├── 03_Plots/                Figures générées (vide initialement)
├── 04_Outputs/              Résultats de calibration (vide initialement)
└── AS_PFE_SCS.pdf          Rapport complet
```

## Prérequis

Python 3.10 ou supérieur avec les bibliothèques suivantes :

```
numpy
pandas
matplotlib
scipy
```

Installation :
```bash
pip install numpy pandas matplotlib scipy
```

## Utilisation

Le traitement suit une chaîne logique. Les scripts peuvent être exécutés dans l'ordre suivant :

### 1. Interpolation des données

```bash
python 01_Scripts/interpolation.py
```

Comble les lacunes dans la série de débit (limite : 60 minutes). Génère `PQ_BV_Cloutasse_interp.csv`.

### 2. Découpage des événements

```bash
python 01_Scripts/etude_hydrologique.py
```

Détecte les événements pluie-ruissellement et calcule les signatures volumétriques (hauteur de pluie hp, hauteur ruisselée hr). Les résultats sont exportés dans `03_Plots/Etude_hydrologique/`.

Paramètres principaux (classe `Config` dans le script) :
- `A_BV_M2` : surface du bassin (m²)
- `P_THR_MM` : seuil de pluie significative
- `POST_DRY_HOURS` : durée de calme pour clôturer un événement
- `USE_EXISTING_EVENTS` : mode rechargement (défaut : `False`)

### 3. Extraction des constantes de vidange

```bash
python 01_Scripts/decoupe_calage_k_runoff_k_sub.py
```

Identifie les phases de décrue et estime les constantes de vidange k (s⁻¹) par régression linéaire sur ln(Q). Trois modes de fonctionnement disponibles (variable `MODE` dans le script) :

- `strict` : critères rigides, segments de haute qualité
- `loose` : compromis qualité/quantité
- `loose_two` : optimisé pour détecter les décrues à double pente

Sorties : tableaux CSV et graphiques annotés dans `03_Plots/Constantes_vidange/`.

### 4. Calibration des modèles

Trois variantes du modèle SCS continu sont implémentées :

#### Modèle A : SCS continu sans routage

```bash
python 01_Scripts/modele_A.py
```

Version de référence. Production SCS continue (abstraction Ia, sol S, infiltration HSM) sans temps de transfert explicite. Réponse instantanée : Q_mod = r_gen × A_BV.

Paramètres calés : Ia, S, k_infiltr, k_seepage

#### Modèle B : SCS continu avec routage de surface

```bash
python 01_Scripts/modele_B.py
```

Ajout d'un réservoir de routage (h_r) avec loi linéaire : r_out = k_runoff × h_r. Permet de lisser la réponse et d'introduire un temps de transfert.

Paramètres calés : k_infiltr, k_seepage, k_runoff (Ia et S fixés)

#### Modèle C : SCS continu avec compartiment lent

```bash
python 01_Scripts/modele_C.py
```

Extension du modèle B avec un réservoir souterrain (h_sub) alimenté par une fraction α_sub de l'infiltration. Débit total : Q_mod = (r_out + q_sub) × A_BV.

Paramètres calés : k_infiltr, k_seepage, k_runoff, k_sub, α_sub

## Méthode de calibration

Les trois modèles utilisent une optimisation multi-start (méthode de Powell, scipy.optimize.minimize) pour limiter les minima locaux. Les paramètres sont bornés à partir des signatures hydrologiques indépendantes (constantes de vidange observées, bilans volumétriques événementiels).

Fonction objectif : RMSE sur log(Q + ε)

Les résultats incluent systématiquement :
- Hydrogrammes événementiels observés vs modélisés
- Bilans de masse (fermeture contrôlée)
- Métriques de performance (NSE, KGE, RMSE)
- Évolution des stocks internes

## Résultats

Les performances dépendent de la structure du modèle. Le modèle A produit des réponses trop instantanées. L'introduction d'un routage (modèle B) améliore la restitution temporelle. Le modèle C permet de séparer les échelles de temps mais présente des difficultés d'identifiabilité (absence de flux interne observé).

Les détails sont présentés dans le chapitre 3 du rapport.

## Limitations

Ce travail présente plusieurs limites méthodologiques :

- Hypothèse de stationnarité des paramètres sur l'année hydrologique
- Équifinalité des paramètres internes en l'absence de mesures de flux intermédiaires
- Simplifications structurelles (modèle global, pas de variabilité spatiale)
- Sensibilité aux données d'entrée (qualité de l'interpolation, ETP journalière)

Les résultats doivent être interprétés dans ce contexte. Le rapport complet propose une analyse critique détaillée.

## Notes techniques

### Bilan de masse

Tous les modèles garantissent la fermeture du bilan :

```
P_tot = ET_tot + Seep_tot + ΔStorage + Closure_error
```

Avec :
- P_tot : pluie totale (m)
- ET_tot : évapotranspiration cumulée (m)
- Seep_tot : percolation profonde (m)
- ΔStorage : variation des stocks (h_a, h_s, h_r, h_sub)
- Closure_error : erreur numérique (< 1e-6 m)

Le ruissellement r_gen ne figure pas dans le bilan global car il constitue un flux interne (transfert vers h_r).

### Constantes de vidange

Les constantes k (s⁻¹) sont liées aux temps de demi-vie par t_1/2 = ln(2) / k.

Ordres de grandeur typiques :
- Routage rapide : k ~ 1e-4 s⁻¹ → t_1/2 ~ 2 h
- Drainage lent : k ~ 1e-6 s⁻¹ → t_1/2 ~ 8 jours

## Références

Le modèle SCS continu s'appuie sur les travaux de :

- Mockus, V. (1956). Hydrology. USDA Soil Conservation Service, National Engineering Handbook, Section 4.
- Guinot, V. et al. (travaux en cours, UMR HSM)

La méthodologie de découpage événementiel et d'analyse des décrues est détaillée dans la Section 2 du rapport.

## Encadrement

Ce travail a été réalisé sous la direction de :

- Vincent Guinot (UMR HSM)
- Luc Neppel (UMR HSM)
- Violetta Montoya Coronado (UMR HSM)

Les données du bassin du Cloutasse proviennent de l'Observatoire Hydro-météorologique Méditerranéen Cévennes-Vivarais (OHM-CV).

## Licence

Ce code est mis à disposition à des fins pédagogiques et de recherche. Toute réutilisation doit citer la source (rapport PFE + dépôt GitHub) et tenir compte des limitations méthodologiques avant application à un autre contexte.

---

Antoine Sognos  
Polytech Montpellier — Département Sciences et Technologies de l'Eau  
Année universitaire 2024-2025
