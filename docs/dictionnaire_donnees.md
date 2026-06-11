# Dictionnaire des données — Customer Churn Prediction

> Données entièrement simulées à des fins pédagogiques.

---

## Churn

**Définition :** Résiliation par un client de son contrat ou abonnement.  
**Variable cible :** `churn = 1` si résiliation dans la fenêtre d'observation, `0` sinon.  
**Taux observé :** ~22% sur le dataset simulé (vs 15–35% observé en industrie telecom).

---

## Métriques modèle

| Métrique | Formule | Interprétation |
|---------|---------|---------------|
| **AUC-ROC** | Aire sous la courbe ROC | 1.0 = parfait, 0.5 = aléatoire — insensible au déséquilibre |
| **Average Precision** | Aire sous la courbe Précision-Rappel | Plus informative que AUC quand classes déséquilibrées |
| **F1-score** | 2 × (P × R) / (P + R) | Équilibre précision/rappel — dépend du seuil |
| **Précision** | TP / (TP + FP) | Parmi les prédits churners, combien le sont vraiment |
| **Rappel** | TP / (TP + FN) | Parmi les vrais churners, combien sont détectés |

---

## Glossaire métier

| Terme | Définition |
|-------|-----------|
| **Churn voluntaire** | Résiliation à l'initiative du client |
| **Churn involontaire** | Résiliation pour impayé |
| **Seuil de décision** | Probabilité au-delà de laquelle on classe « churner » |
| **Cost of False Negative** | Valeur à vie du client perdu (LTV) |
| **Cost of False Positive** | Coût d'une action de rétention inutile |
| **LTV** | Lifetime Value = revenu moyen mensuel × durée de vie espérée |
| **Taux de rétention** | 1 - taux de churn sur une période |

---

## Variables les plus discriminantes (résultats modèle)

| Variable | Impact sur churn | Direction |
|----------|-----------------|-----------|
| `contract_type = Monthly` | Fort | Positif (augmente le churn) |
| `num_complaints` | Élevé | Positif |
| `last_interaction_days` | Élevé | Positif |
| `tenure_months` | Modéré | Négatif (fidélise) |
| `has_tech_support = 1` | Modéré | Négatif (réduit le churn) |
| `num_products` | Modéré | Négatif (cross-sell fidélise) |
