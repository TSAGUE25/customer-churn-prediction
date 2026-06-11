# Customer Churn Prediction

> Prédiction du churn client sur un portefeuille Telecom/Finance — pipeline ML complet avec explicabilité et optimisation du seuil de décision.  
> **Stack :** Python · scikit-learn · pandas · matplotlib · seaborn

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-Emmanuel%20TSAGUE-purple)](https://github.com/TSAGUE25)

---

## Table des matières

1. [Contexte métier](#contexte-métier)
2. [Problème](#problème)
3. [Données utilisées](#données-utilisées)
4. [Architecture du dépôt](#architecture-du-dépôt)
5. [Méthodes et pipeline](#méthodes-et-pipeline)
6. [Résultats obtenus](#résultats-obtenus)
7. [Valeur métier](#valeur-métier)
8. [Installation](#installation)

---

## Contexte métier

Dans les secteurs Telecom, Finance et SaaS, **perdre un client coûte 5 à 25× plus cher** que d'en fidéliser un. Le churn (résiliation) est donc une priorité stratégique : identifier en avance les clients susceptibles de partir permet de déclencher des actions de rétention ciblées avant qu'il ne soit trop tard.

Ce projet simule le cas d'un opérateur Telecom gérant **5 000 clients** sur des contrats mensuels, annuels et biannuels, avec des profils d'usage variés (internet, support technique, canal d'acquisition).

---

## Problème

> *"Comment identifier automatiquement les clients les plus susceptibles de résilier leur contrat dans les 30 prochains jours, et quantifier l'impact financier de ne pas agir ?"*

**Traduction analytique :**
1. Construire un modèle de classification binaire (`churn = 1`)
2. Comparer plusieurs algorithmes (LogisticRegression, RandomForest, GradientBoosting)
3. Optimiser le seuil de décision pour un coût asymétrique (FN >> FP)
4. Identifier les leviers actionnables (feature importance, analyse segmentée)
5. Quantifier le chiffre d'affaires mensuel exposé

---

## Données utilisées

> Données entièrement simulées — aucune donnée réelle ou confidentielle.

**`data_sample/customers.csv`** — 5 000 clients, 15 variables :

| Variable | Description |
|----------|-------------|
| `tenure_months` | Ancienneté en mois (1–120) |
| `monthly_charge` | Montant mensuel en € (20–150) |
| `num_complaints` | Nombre de réclamations (0–10) |
| `contract_type` | Monthly / Annual / Biannual |
| `acquisition_channel` | Web / Store / Agent / Referral |
| `has_tech_support` | Option support technique (0/1) |
| `last_interaction_days` | Jours depuis la dernière interaction |
| `churn` | **Cible** : 0 = Fidèle, 1 = Churner |

Taux de churn : **~22%** (déséquilibre géré par `class_weight='balanced'`).

---

## Architecture du dépôt

```
customer-churn-prediction/
│
├── data_sample/
│   └── schema_reference.md          # Schéma complet, règles de génération
│   └── customers.csv                # Généré au premier run du notebook
│
├── src/
│   ├── __init__.py
│   ├── churn_predictor.py           # ChurnPredictor + generate_churn_data()
│   └── visualization.py            # ChurnVisualizer — 8 figures
│
├── notebooks/
│   └── 01_churn_analysis.py        # Pipeline complet (11 sections)
│
├── docs/
│   ├── methodologie.md             # Pipeline, SHAP, seuil asymétrique
│   ├── dictionnaire_donnees.md     # Métriques, glossaire métier
│   └── guide_utilisateur.md        # Prise en main, exemples de code
│
├── figures/                        # 8 figures PNG générées
├── reports/                        # Rapport Markdown auto-généré
├── requirements.txt
└── .gitignore
```

### `ChurnPredictor` — méthodes principales

| Méthode | Rôle |
|---------|------|
| `prepare_and_split()` | Feature engineering + train/test split stratifié |
| `compare_models()` | Entraîne et compare LR, RF, GB sur AUC-ROC |
| `train(model_name, cv)` | GridSearchCV sur le modèle choisi |
| `evaluate()` | AUC-ROC, confusion matrix, rapport de classification |
| `optimal_threshold(cost_fn, cost_fp)` | Seuil minimisant le coût asymétrique FN/FP |
| `feature_importance()` | Importance Gini ou coefs selon le modèle |
| `predict_at_risk(threshold)` | Liste des clients à risque avec probabilités |
| `summary_report()` | Dictionnaire de métriques clés |

---

## Méthodes et pipeline

```
customers.csv (5 000 lignes)
        │
        ▼
   Feature Engineering
   (charge_per_month_ratio, complaint_rate)
        │
        ▼
ColumnTransformer
├── StandardScaler  → 11 variables numériques
└── OneHotEncoder   → 4 variables catégorielles
        │
        ▼
Comparaison 3 modèles (GridSearchCV × StratifiedKFold)
├── Logistic Regression
├── Random Forest         ← meilleur AUC
└── Gradient Boosting
        │
        ▼
Évaluation : AUC-ROC · Confusion Matrix · Classification Report
        │
        ├──→ Feature Importance (top 15)
        ├──→ Seuil optimal (coût asymétrique FN×5 + FP×1)
        └──→ Clients à risque (prob ≥ seuil) + CA exposé
                │
                ▼
        Rapport Markdown + 8 figures
```

---

## Résultats obtenus

> Résultats sur le dataset simulé de 5 000 clients.

| Modèle | AUC-ROC |
|--------|---------|
| Régression Logistique | ~0.771 |
| **Random Forest** | **~0.823** |
| Gradient Boosting | ~0.811 |

| Indicateur clé | Valeur |
|----------------|--------|
| AUC-ROC test (RF) | **~0.823** |
| Seuil optimal (FN×5 + FP×1) | **~0.28** |
| Clients à risque identifiés | **~650 / 5 000** |
| CA mensuel exposé | **~52 000 €/mois** |

**Top 3 leviers de churn identifiés :**
1. `contract_type = Monthly` — taux de churn 3× supérieur aux contrats annuels
2. `num_complaints` — chaque réclamation supplémentaire augmente la probabilité de churn de ~3%
3. `last_interaction_days` — les clients sans contact depuis > 90 jours churne 2× plus

### Visualisations

**Courbes ROC — comparaison des 3 modèles :**

![ROC Curves](figures/fig1_roc_curves.png)

**Matrice de confusion (Random Forest) :**

![Confusion Matrix](figures/fig2_confusion_matrix.png)

**Importance des features :**

![Feature Importance](figures/fig3_feature_importance.png)

**Taux de churn par type de contrat :**

![Churn by Contract](figures/fig4_churn_by_contract.png)

**Taux de churn par canal d'acquisition :**

![Churn by Channel](figures/fig5_churn_by_channel.png)

**Distribution du montant mensuel — Churners vs Fidèles :**

![Monthly Charge Distribution](figures/fig6_monthly_charge_distribution.png)

**Optimisation du seuil de décision (coût asymétrique) :**

![Threshold Optimization](figures/fig7_threshold_optimization.png)

**Taux de churn par ancienneté :**

![Tenure vs Churn](figures/fig8_tenure_vs_churn.png)

---

## Valeur métier

### Pour un directeur commercial / CRM
- **Priorisation des actions de rétention** : contacter les 650 clients à risque avant la fin du mois
- **ROI immédiat** : 52 000 €/mois de CA potentiellement sauvé avec un taux de succès de 30% → 15 600 €/mois
- **Personnalisation** : les features SHAP indiquent quoi offrir à chaque segment

### Pour un Data Scientist
- Pipeline sklearn réutilisable sur n'importe quel dataset de churn (changer les CSV)
- Comparaison systématique de 3 familles de modèles
- Gestion correcte du déséquilibre de classes + optimisation du seuil métier
- Séparation propre entre ingestion, modélisation et visualisation (OOP)

### Exemple de calcul ROI
```
CA mensuel exposé  : 52 000 €
Taux de rétention  : 30% (campagne ciblée)
CA sauvé / mois    : 15 600 €
Coût campagne      : ~3 000 € (emails + offres)
ROI mensuel        : 420%
```

---

## Installation

```bash
git clone https://github.com/TSAGUE25/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt
python notebooks/01_churn_analysis.py
```

**Sorties générées :**
- `data_sample/customers.csv` — dataset 5 000 clients (si absent)
- `figures/` — 8 graphiques PNG
- `reports/churn_report.md` — rapport Markdown complet


## Contributors

| Nom | Role | GitHub |
|-----|------|--------|
| **TSAGUE Emmanuel** | Data Scientist - auteur principal | [@TSAGUE25](https://github.com/TSAGUE25) |

---

*Auteur : Emmanuel TSAGUE — Data Scientist / Data Analyst*  
*Données : entièrement simulées — aucune donnée réelle ou confidentielle*
