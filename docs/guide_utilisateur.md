# Guide utilisateur — Customer Churn Prediction

## Prérequis

```bash
pip install -r requirements.txt
```

## Lancer l'analyse complète

```bash
cd customer-churn-prediction
python notebooks/01_churn_analysis.py
```

L'exécution génère automatiquement `data_sample/customers.csv` (5 000 clients) si absent.

**Sorties :**
- 8 figures dans `figures/`
- Rapport Markdown dans `reports/churn_report.md`

## Utilisation directe des classes

```python
import pandas as pd
from src.churn_predictor import ChurnPredictor, generate_churn_data
from src.visualization import ChurnVisualizer

df = generate_churn_data(n=5000)

predictor = ChurnPredictor(df)
predictor.prepare_and_split()

# Comparer les 3 modèles
print(predictor.compare_models())

# Entraîner le meilleur
predictor.train(model_name='random_forest')
results = predictor.evaluate()
print(f"AUC-ROC : {results['auc_roc']}")

# Seuil optimal (coût asymétrique)
threshold, cost = predictor.optimal_threshold(cost_fn=5, cost_fp=1)
print(f"Seuil : {threshold:.3f}")

# Clients à risque
at_risk = predictor.predict_at_risk(threshold=threshold)
print(f"{len(at_risk)} clients à risque — {at_risk['monthly_charge'].sum():,.0f} €/mois exposés")

# Importance features
importance = predictor.feature_importance()
print(importance.head(10))

# Visualisations
viz = ChurnVisualizer(output_dir='figures')
viz.plot_roc_curves(predictor._all_results)
viz.plot_churn_by_contract(df)
```

## Structure des sorties

| Fichier | Contenu |
|---------|---------|
| `figures/fig1_roc_curves.png` | Courbes ROC des 3 modèles |
| `figures/fig2_confusion_matrix.png` | Matrice de confusion |
| `figures/fig3_feature_importance.png` | Top 15 features |
| `figures/fig4_churn_by_contract.png` | Churn par type de contrat |
| `figures/fig5_churn_by_channel.png` | Churn par canal d'acquisition |
| `figures/fig6_monthly_charge_distribution.png` | Distribution montant mensuel |
| `figures/fig7_threshold_optimization.png` | Courbe coût vs seuil |
| `figures/fig8_tenure_vs_churn.png` | Churn par ancienneté |
| `reports/churn_report.md` | Rapport Markdown complet |

## Adapter à vos données

Remplacer `generate_churn_data()` par votre propre CSV. Les colonnes requises sont décrites dans `data_sample/schema_reference.md`. Ajuster `NUMERIC_FEATURES` et `CATEGORICAL_FEATURES` dans `ChurnPredictor` si vos noms de colonnes diffèrent.
