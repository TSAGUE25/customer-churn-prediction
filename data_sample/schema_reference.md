# Schéma de référence — Customer Churn Prediction

> Données entièrement simulées à des fins pédagogiques.

## Table : `customers.csv`

| Colonne | Type | Description | Valeurs / Contraintes |
|---------|------|-------------|----------------------|
| `customer_id` | string | Identifiant unique | C00001–C05000 |
| `age` | integer | Âge du client | 18–75 |
| `tenure_months` | integer | Ancienneté en mois | 1–120 |
| `monthly_charge` | float | Montant mensuel (€) | 20–150 |
| `total_charges` | float | Charges totales cumulées (€) | tenure × monthly × bruit |
| `num_complaints` | integer | Nombre de réclamations | 0–10 |
| `contract_type` | string | Type de contrat | Monthly, Annual, Biannual |
| `acquisition_channel` | string | Canal d'acquisition | Web, Store, Agent, Referral |
| `internet_service` | string | Type de connexion | DSL, Fiber, None |
| `has_tech_support` | integer | Option support technique | 0 = Non, 1 = Oui |
| `num_products` | integer | Nombre de produits souscrits | 1–4 |
| `payment_method` | string | Mode de paiement | Auto, Manual |
| `last_interaction_days` | integer | Jours depuis dernière interaction | 1–365 |
| `avg_call_duration_min` | float | Durée moyenne des appels (min) | 0–30 |
| `churn` | integer | **Variable cible** — résiliation | 0 = Fidèle, 1 = Churner |

**Cardinalité :** 5 000 clients

---

## Variables dérivées (feature engineering)

| Variable | Formule | Justification |
|----------|---------|---------------|
| `charge_per_month_ratio` | `monthly_charge / (tenure_months + 1)` | Charge relative à l'ancienneté |
| `complaint_rate` | `num_complaints / (tenure_months + 1)` | Fréquence de réclamations |

---

## Distribution cibles

- Taux de churn global : **~22%** (déséquilibre géré par `class_weight='balanced'`)
- Churn Monthly : ~35% | Annual : ~15% | Biannual : ~8%

---

## Règles de génération

1. La probabilité de churn augmente avec `num_complaints` et `last_interaction_days`
2. Les contrats mensuels ont un taux de churn 3× supérieur aux contrats biannuels
3. L'absence de tech support augmente la probabilité de churn de ~5%
4. Le paiement manuel (vs automatique) corrèle positivement avec le churn
