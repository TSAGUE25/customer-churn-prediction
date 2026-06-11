# Méthodologie — Customer Churn Prediction

## 1. Feature Engineering

Deux variables dérivées créées avant modélisation :

```
charge_per_month_ratio = monthly_charge / (tenure_months + 1)
complaint_rate         = num_complaints / (tenure_months + 1)
```

Justification : un client récent avec beaucoup de plaintes est plus à risque qu'un client ancien avec le même nombre en absolu.

---

## 2. Pipeline sklearn

```
ColumnTransformer
├── StandardScaler → variables numériques (11)
└── OneHotEncoder  → variables catégorielles (4 → ~12 dummies)
         │
         ▼
    Classifier (LR / RF / GB)
```

Le pipeline garantit l'absence de fuite de données (data leakage) entre train et test.

---

## 3. Comparaison de modèles

| Modèle | Avantages | Limites |
|--------|-----------|---------|
| Régression Logistique | Interprétable, rapide | Linéaire, sensible aux corrélations |
| Random Forest | Robuste, feature importance native | Moins interprétable, lent |
| Gradient Boosting | Souvent meilleur AUC | Risque overfitting, hyperparamètres |

Métrique principale : **AUC-ROC** (insensible au déséquilibre de classes).

---

## 4. Déséquilibre de classes

Taux de churn ~22% → déséquilibre modéré. Gestion par `class_weight='balanced'` (LR, RF) qui pondère les erreurs sur la classe minoritaire.

---

## 5. Seuil de décision optimal (coût asymétrique)

Le seuil par défaut (0.5) n'est pas optimal en métier. Un faux négatif (churner raté) coûte bien plus qu'un faux positif (offre de rétention envoyée à tort) :

```
Coût total = FN × cost_fn + FP × cost_fp
           = FN × 5 + FP × 1
```

Le seuil optimal minimise ce coût sur 200 valeurs candidates entre 0.01 et 0.99.

---

## 6. Interprétation SHAP

SHAP (SHapley Additive exPlanations) décompose la prédiction de chaque client en contributions individuelles par feature. Contrairement à l'importance globale (Gini), SHAP donne l'impact directionnel (positif / négatif) par observation.

---

## 7. Limites

- Données simulées sans corrélations complexes réelles
- Pas de variable temporelle (séries temporelles de consommation)
- Le modèle ne capture pas les événements déclencheurs (incident, déménagement)
- Améliorations : XGBoost natif, LGBM, calibration des probabilités (Platt scaling)
