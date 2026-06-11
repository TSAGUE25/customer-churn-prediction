"""
Prédiction du churn client — Telecom / Finance
Cas 2 — Portfolio Data Science | Emmanuel TSAGUE
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.churn_predictor import ChurnPredictor, generate_churn_data
from src.visualization import ChurnVisualizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data_sample')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ======================================================================
# 0. CHARGEMENT / GÉNÉRATION DES DONNÉES
# ======================================================================

print("=" * 60)
print("0. DONNÉES")
print("=" * 60)

csv_path = os.path.join(DATA_DIR, 'customers.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Chargé depuis {csv_path}")
else:
    df = generate_churn_data(n=5000, random_state=42)
    df.to_csv(csv_path, index=False)
    print(f"Dataset généré et sauvegardé : {csv_path}")

print(f"Shape : {df.shape}")
print(f"Taux de churn : {df['churn'].mean()*100:.1f}%")

# ======================================================================
# 1. EXPLORATION DES DONNÉES (EDA)
# ======================================================================

print("\n" + "=" * 60)
print("1. EDA")
print("=" * 60)

print("\n--- Répartition par type de contrat ---")
print(df.groupby('contract_type')['churn'].agg(['count', 'mean'])
      .rename(columns={'mean': 'taux_churn'}))

print("\n--- Répartition par canal d'acquisition ---")
print(df.groupby('acquisition_channel')['churn'].mean().round(3))

print("\n--- Statistiques : montant mensuel ---")
print(df.groupby('churn')['monthly_charge'].describe().round(2))

print("\n--- Corrélation variables numériques / churn ---")
numeric_cols = ['age', 'tenure_months', 'monthly_charge', 'num_complaints',
                'last_interaction_days', 'num_products', 'avg_call_duration_min']
print(df[numeric_cols + ['churn']].corr()['churn'].drop('churn').round(3).sort_values())

print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())

# ======================================================================
# 2. INITIALISATION ET SPLIT
# ======================================================================

print("\n" + "=" * 60)
print("2. PRÉPARATION")
print("=" * 60)

predictor = ChurnPredictor(df)
predictor.prepare_and_split(test_size=0.2, random_state=42)
print(f"Train : {len(predictor.X_train)} | Test : {len(predictor.X_test)}")
print(f"Taux churn train : {predictor.y_train.mean()*100:.1f}%")
print(f"Taux churn test  : {predictor.y_test.mean()*100:.1f}%")

# ======================================================================
# 3. COMPARAISON DES MODÈLES
# ======================================================================

print("\n" + "=" * 60)
print("3. COMPARAISON MODÈLES")
print("=" * 60)

comparison = predictor.compare_models(cv=5)
print("\nAUC-ROC par modèle :")
print(comparison.to_string())

# ======================================================================
# 4. ENTRAÎNEMENT DU MEILLEUR MODÈLE (Random Forest)
# ======================================================================

print("\n" + "=" * 60)
print("4. ENTRAÎNEMENT — Random Forest")
print("=" * 60)

predictor.train(model_name='random_forest', cv=5)
print(f"Meilleurs paramètres : {predictor._results['best_params']}")
print(f"AUC-ROC CV : {predictor._results['best_cv_auc']}")

# ======================================================================
# 5. ÉVALUATION
# ======================================================================

print("\n" + "=" * 60)
print("5. ÉVALUATION")
print("=" * 60)

results = predictor.evaluate()
print(f"\nAUC-ROC test  : {results['auc_roc']}")
print(f"Avg Precision : {results['avg_precision']}")
print("\nMatrice de confusion :")
print(results['confusion_matrix'])
print("\nRapport de classification :")
cr = results['classification_report']
for label in ['0', '1']:
    print(f"  Classe {label} — Precision: {cr[label]['precision']:.3f} | "
          f"Recall: {cr[label]['recall']:.3f} | F1: {cr[label]['f1-score']:.3f}")

# ======================================================================
# 6. IMPORTANCE DES FEATURES
# ======================================================================

print("\n" + "=" * 60)
print("6. IMPORTANCE DES FEATURES")
print("=" * 60)

importance = predictor.feature_importance()
print("\nTop 10 features :")
print(importance.head(10).round(4).to_string())

# ======================================================================
# 7. SEUIL DE DÉCISION OPTIMAL
# ======================================================================

print("\n" + "=" * 60)
print("7. SEUIL DE DÉCISION OPTIMAL (coût asymétrique)")
print("=" * 60)

optimal_t, min_cost = predictor.optimal_threshold(cost_fn=5, cost_fp=1)
print(f"\nSeuil optimal : {optimal_t:.3f}")
print(f"Coût minimum  : {min_cost:.0f} (FN×5 + FP×1)")
print("Interprétation : rater un churner coûte 5× plus qu'une fausse alarme")

# ======================================================================
# 8. CLIENTS À RISQUE
# ======================================================================

print("\n" + "=" * 60)
print("8. CLIENTS À RISQUE")
print("=" * 60)

at_risk = predictor.predict_at_risk(threshold=optimal_t)
print(f"\nNombre de clients à risque : {len(at_risk)}")
print(f"Chiffre d'affaires mensuel exposé : {at_risk['monthly_charge'].sum():,.0f} €")
print("\nTop 10 clients les plus à risque :")
print(at_risk[['customer_id', 'churn_prob', 'monthly_charge',
               'tenure_months', 'num_complaints']].head(10).to_string(index=False))

# ======================================================================
# 9. SYNTHÈSE
# ======================================================================

print("\n" + "=" * 60)
print("9. SYNTHÈSE")
print("=" * 60)

summary = predictor.summary_report()
for k, v in summary.items():
    print(f"  {k:35s} : {v}")

# ======================================================================
# 10. VISUALISATIONS
# ======================================================================

print("\n" + "=" * 60)
print("10. VISUALISATIONS")
print("=" * 60)

viz = ChurnVisualizer(output_dir=FIGURES_DIR)

viz.plot_roc_curves(predictor._all_results)
viz.plot_confusion_matrix(results['confusion_matrix'], model_name='random_forest')
viz.plot_feature_importance(importance)
viz.plot_churn_by_contract(df)
viz.plot_churn_by_channel(df)
viz.plot_monthly_charge_distribution(df)
viz.plot_threshold_optimization(results['threshold_costs'], optimal_t)
viz.plot_tenure_vs_churn(df)

print("\n8 figures générées dans figures/")

# ======================================================================
# 11. RAPPORT MARKDOWN
# ======================================================================

print("\n" + "=" * 60)
print("11. RAPPORT")
print("=" * 60)

lines = [
    "# Rapport de Prédiction de Churn Client",
    "",
    f"**Date :** 2024-12-15  ",
    f"**Périmètre :** {summary['nb_clients']:,} clients | Modèle : Random Forest  ",
    "",
    "> *Données simulées à des fins pédagogiques.*",
    "",
    "---",
    "",
    "## 1. Synthèse exécutive",
    "",
    "| Indicateur | Valeur |",
    "|-----------|--------|",
    f"| Taux de churn réel | **{summary['taux_churn_reel']}%** |",
    f"| AUC-ROC (test) | **{summary['auc_roc']}** |",
    f"| Avg Precision | **{summary['avg_precision']}** |",
    f"| Seuil de décision optimal | **{summary['optimal_threshold']}** |",
    f"| Clients à risque identifiés | **{summary['nb_clients_at_risk']:,}** |",
    f"| CA mensuel exposé | **{summary['manque_a_gagner_mensuel']:,.0f} €** |",
    "",
    "---",
    "",
    "## 2. Comparaison des modèles",
    "",
    "| Modèle | AUC-ROC |",
    "|--------|---------|",
]
for name, res in predictor._all_results.items():
    lines.append(f"| {name.replace('_', ' ').title()} | **{res['auc_roc']}** |")

lines += [
    "",
    "---",
    "",
    "## 3. Top 5 features explicatives",
    "",
    "| Feature | Importance |",
    "|---------|-----------|",
]
for feat, imp in importance.head(5).items():
    lines.append(f"| `{feat}` | {imp:.4f} |")

lines += [
    "",
    "---",
    "",
    "## 4. Recommandations",
    "",
    "1. **Campagne de rétention urgente** sur les clients à risque (prob ≥ seuil optimal)",
    "2. **Contrats mensuels** : taux de churn 3× supérieur aux contrats annuels → proposer migration",
    "3. **Réduire les délais d'interaction** : last_interaction_days fortement corrélé au churn",
    "4. **Tech support** : réduire le churn de 8% — proposer l'option aux clients sans support",
    "",
    "---",
    "",
    "*Rapport généré automatiquement — Customer Churn Prediction*  ",
    "*Auteur : Emmanuel TSAGUE — Data Scientist / Data Analyst*",
]

report_path = os.path.join(REPORTS_DIR, 'churn_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Rapport sauvegardé : {report_path}")
print("\n=== ANALYSE TERMINÉE ===")
