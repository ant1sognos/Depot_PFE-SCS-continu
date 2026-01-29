# Modélisation hydrologique continue SCS–HSM  
## Deux sites : Cloutasse (BV naturel) et CSR Lyon (ouvrage urbain OTHU)

Ce dépôt regroupe les codes Python développés dans le cadre du PFE « Interprétation et développement du modèle SCS pour la modélisation de séries chronologiques continues ». Il propose une chaîne complète de traitement et de modélisation continue, appliquée à deux sites contrastés afin de comparer l’effet de la structure du modèle (production seule, routage, compartiment lent/cascade) sur la restitution des hydrogrammes et l’identifiabilité des paramètres.
Le rapport complet et l’analyse des résultats sont disponibles dans `AS_PFE_SCS.pdf`.

## Sites et objectifs

### 1) Bassin versant du Cloutasse (0.81 km², naturel)
- Données : pluie + débit à l’exutoire (un seul flux observé).
- Objectif : comparer des extensions continues du noyau SCS–HSM avec complexification progressive :
  - Modèle A : sans routage (référence conceptuelle)
  - Modèle B : routage de surface (temps de transfert)
  - Modèle C : ajout d’une voie lente (compartiment subsurface)
- Chaîne : interpolation limitée du débit → découpage événements → estimation de constantes de vidange → calibration et diagnostics événementiels.

Guide détaillé : `Cloutasse/README.md`.

### 2) CSR Lyon (94 m², parking expérimental OTHU)
- Données : pluie + deux flux observés (Q_ruiss et Q_inf).
- Objectif : exploiter la double contrainte (Q_ruiss, Q_inf) pour réduire l’équifinalité et tester une structure adaptée au site :
  - Modèle A : sans routage
  - Modèle B : routage simple
  - Modèle D : cascade de routage (spécifique CSR) + substepping
- Chaîne : découpage événements → signatures (k_seepage, k_runoff) → simulation annuelle continue avec validation croisée temporelle → analyse mensuelle et événementielle.

Guide détaillé : `CSR_Lyon/README.md`.

---

## Ce que permet le dépôt

1) Construire des événements pluie–réponse à partir de séries continues, et générer des diagnostics.
2) Extraire des constantes de vidange (décrues) pour encadrer des bornes réalistes de calibration.
3) Comparer plusieurs structures SCS–HSM continues (A/B/C/D) sur événements.
4) Sur CSR : exécuter une simulation annuelle continue avec warm-up, splits temporels (H1/H2) et calibration hiérarchisée, puis produire des métriques (NSE, KGE, RMSE, biais volumique) et des bilans.

---

## Organisation
├─ AS_PFE_SCS.pdf
├─ Cloutasse/
│ └─ README.md
└─ CSR_Lyon/
└─ README.md


## Démarrage rapide

CSR (chaîne complète annuelle) :

cd CSR_Lyon/01_Code
python decoupe_event_plot_cumul_vs_pluie.py
python run_annuel_modele_D.py
python analyse_annuelle_modele_D.py


Cloutasse (comparaison événementielle A/B/C) :

cd Cloutasse/01_Scripts
python interpolation.py
python etude_hydrologique.py
python decoupe_calage_k_runoff_k_sub.py
python modele_A.py
python modele_B.py
python modele_C.p
