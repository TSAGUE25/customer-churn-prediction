import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


def generate_churn_data(n=5000, random_state=42):
    """Génère un dataset client fictif cohérent avec le comportement de churn."""
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(1, 121, n)
    monthly_charge = rng.uniform(20, 150, n).round(2)
    num_complaints = rng.integers(0, 11, n)
    age = rng.integers(18, 76, n)
    num_products = rng.integers(1, 5, n)
    last_interaction_days = rng.integers(1, 366, n)
    avg_call_duration = rng.uniform(0, 30, n).round(1)
    has_tech_support = rng.integers(0, 2, n)

    contract_type = rng.choice(['Monthly', 'Annual', 'Biannual'],
                                n, p=[0.55, 0.30, 0.15])
    acquisition_channel = rng.choice(['Web', 'Store', 'Agent', 'Referral'],
                                      n, p=[0.35, 0.25, 0.25, 0.15])
    internet_service = rng.choice(['DSL', 'Fiber', 'None'],
                                   n, p=[0.40, 0.45, 0.15])
    payment_method = rng.choice(['Auto', 'Manual'], n, p=[0.60, 0.40])

    total_charges = (monthly_charge * tenure * rng.uniform(0.95, 1.05, n)).round(2)

    churn_prob = (
        0.05
        + 0.003 * num_complaints
        + 0.002 * last_interaction_days / 30
        - 0.001 * tenure
        + 0.002 * monthly_charge / 10
        + 0.15 * (contract_type == 'Monthly').astype(float)
        - 0.08 * (contract_type == 'Biannual').astype(float)
        - 0.05 * has_tech_support
        + 0.03 * (payment_method == 'Manual').astype(float)
        - 0.02 * num_products
    )
    churn_prob = np.clip(churn_prob + rng.normal(0, 0.04, n), 0.02, 0.92)
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'customer_id': [f'C{i:05d}' for i in range(1, n + 1)],
        'age': age,
        'tenure_months': tenure,
        'monthly_charge': monthly_charge,
        'total_charges': total_charges,
        'num_complaints': num_complaints,
        'contract_type': contract_type,
        'acquisition_channel': acquisition_channel,
        'internet_service': internet_service,
        'has_tech_support': has_tech_support,
        'num_products': num_products,
        'payment_method': payment_method,
        'last_interaction_days': last_interaction_days,
        'avg_call_duration_min': avg_call_duration,
        'churn': churn,
    })
    return df


class ChurnPredictor:
    """Modélisation du churn client — LogisticRegression, RandomForest, GradientBoosting."""

    NUMERIC_FEATURES = [
        'age', 'tenure_months', 'monthly_charge', 'total_charges',
        'num_complaints', 'num_products', 'last_interaction_days',
        'avg_call_duration_min', 'has_tech_support',
        'charge_per_month_ratio', 'complaint_rate',
    ]
    CATEGORICAL_FEATURES = [
        'contract_type', 'acquisition_channel', 'internet_service', 'payment_method'
    ]

    PARAM_GRIDS = {
        'logistic': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__solver': ['lbfgs'],
            'classifier__max_iter': [500],
        },
        'random_forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [6, 10, None],
            'classifier__min_samples_leaf': [5, 10],
        },
        'gradient_boosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5],
        },
    }

    def __init__(self, df):
        self.df = df.copy()
        self.pipeline = None
        self.model_name = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self._results = {}
        self._all_results = {}

    def _engineer_features(self, df):
        df = df.copy()
        df['charge_per_month_ratio'] = df['monthly_charge'] / (df['tenure_months'] + 1)
        df['complaint_rate'] = df['num_complaints'] / (df['tenure_months'] + 1)
        return df

    def _build_pipeline(self, model_name):
        estimators = {
            'logistic': LogisticRegression(random_state=42, class_weight='balanced'),
            'random_forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
        }
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             self.CATEGORICAL_FEATURES),
        ])
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', estimators[model_name]),
        ])

    def prepare_and_split(self, test_size=0.2, random_state=42):
        df = self._engineer_features(self.df)
        features = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
        X = df[features]
        y = df['churn']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def train(self, model_name='random_forest', cv=5):
        if self.X_train is None:
            self.prepare_and_split()
        self.model_name = model_name
        pipe = self._build_pipeline(model_name)
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        gs = GridSearchCV(
            pipe, self.PARAM_GRIDS[model_name],
            scoring='roc_auc', cv=cv_strategy, n_jobs=-1, verbose=0
        )
        gs.fit(self.X_train, self.y_train)
        self.pipeline = gs.best_estimator_
        self._results = {
            'best_params': gs.best_params_,
            'best_cv_auc': round(gs.best_score_, 4),
        }
        return self

    def evaluate(self):
        if self.pipeline is None:
            raise RuntimeError("Appeler train() d'abord.")
        y_prob = self.pipeline.predict_proba(self.X_test)[:, 1]
        y_pred = self.pipeline.predict(self.X_test)
        auc = roc_auc_score(self.y_test, y_prob)
        ap = average_precision_score(self.y_test, y_prob)
        fpr, tpr, thresholds_roc = roc_curve(self.y_test, y_prob)
        prec, rec, thresholds_pr = precision_recall_curve(self.y_test, y_prob)
        self._results.update({
            'auc_roc': round(auc, 4),
            'avg_precision': round(ap, 4),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'fpr': fpr, 'tpr': tpr, 'thresholds_roc': thresholds_roc,
            'precision': prec, 'recall': rec,
            'y_prob': y_prob, 'y_pred': y_pred,
        })
        return self._results

    def compare_models(self, cv=5):
        if self.X_train is None:
            self.prepare_and_split()
        for name in self.PARAM_GRIDS:
            pipe = self._build_pipeline(name)
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            gs = GridSearchCV(pipe, self.PARAM_GRIDS[name],
                              scoring='roc_auc', cv=cv_strategy, n_jobs=-1)
            gs.fit(self.X_train, self.y_train)
            best = gs.best_estimator_
            y_prob = best.predict_proba(self.X_test)[:, 1]
            self._all_results[name] = {
                'pipeline': best,
                'auc_roc': round(roc_auc_score(self.y_test, y_prob), 4),
                'fpr': roc_curve(self.y_test, y_prob)[0],
                'tpr': roc_curve(self.y_test, y_prob)[1],
                'y_prob': y_prob,
            }
        return pd.DataFrame({
            name: {'AUC-ROC': v['auc_roc']} for name, v in self._all_results.items()
        }).T

    def optimal_threshold(self, cost_fn=5, cost_fp=1):
        if 'y_prob' not in self._results:
            self.evaluate()
        y_prob = self._results['y_prob']
        thresholds = np.linspace(0.01, 0.99, 200)
        costs = []
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            cm = confusion_matrix(self.y_test, y_pred_t)
            tn, fp, fn, tp = cm.ravel()
            total_cost = fn * cost_fn + fp * cost_fp
            costs.append(total_cost)
        best_idx = np.argmin(costs)
        self._results['optimal_threshold'] = thresholds[best_idx]
        self._results['threshold_costs'] = list(zip(thresholds, costs))
        return thresholds[best_idx], costs[best_idx]

    def feature_importance(self):
        if self.pipeline is None:
            raise RuntimeError("Appeler train() d'abord.")
        clf = self.pipeline.named_steps['classifier']
        pre = self.pipeline.named_steps['preprocessor']
        cat_cols = pre.transformers_[1][1].get_feature_names_out(self.CATEGORICAL_FEATURES)
        feature_names = self.NUMERIC_FEATURES + list(cat_cols)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        else:
            importances = np.abs(clf.coef_[0])
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def predict_at_risk(self, threshold=None):
        if self.pipeline is None:
            raise RuntimeError("Appeler train() d'abord.")
        if threshold is None:
            threshold = self._results.get('optimal_threshold', 0.5)
        df = self._engineer_features(self.df)
        X = df[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]
        probs = self.pipeline.predict_proba(X)[:, 1]
        result = self.df[['customer_id', 'monthly_charge', 'tenure_months', 'num_complaints']].copy()
        result['churn_prob'] = probs.round(3)
        result['at_risk'] = (probs >= threshold).astype(int)
        return result[result['at_risk'] == 1].sort_values('churn_prob', ascending=False)

    def summary_report(self):
        res = self._results
        at_risk = self.predict_at_risk()
        report = {
            'model': self.model_name,
            'nb_clients': len(self.df),
            'taux_churn_reel': round(self.df['churn'].mean() * 100, 1),
            'auc_roc': res.get('auc_roc'),
            'avg_precision': res.get('avg_precision'),
            'best_cv_auc': res.get('best_cv_auc'),
            'optimal_threshold': round(res.get('optimal_threshold', 0.5), 3),
            'nb_clients_at_risk': len(at_risk),
            'manque_a_gagner_mensuel': round(at_risk['monthly_charge'].sum(), 0),
        }
        return report
