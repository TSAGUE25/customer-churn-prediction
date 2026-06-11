import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, roc_auc_score


class ChurnVisualizer:
    """Visualisations pour l'analyse de churn client."""

    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})

    def _save(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure sauvegardée : {path}")
        return path

    def plot_roc_curves(self, all_results):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'logistic': '#1976D2', 'random_forest': '#388E3C', 'gradient_boosting': '#E53935'}
        labels = {'logistic': 'Régression Logistique', 'random_forest': 'Random Forest',
                  'gradient_boosting': 'Gradient Boosting'}
        for name, res in all_results.items():
            ax.plot(res['fpr'], res['tpr'],
                    label=f"{labels[name]} (AUC={res['auc_roc']:.3f})",
                    color=colors[name], linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC=0.500)')
        ax.set_xlabel('Taux de faux positifs', fontsize=12)
        ax.set_ylabel('Taux de vrais positifs', fontsize=12)
        ax.set_title('Courbes ROC — Comparaison des modèles', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        return self._save(fig, 'fig1_roc_curves.png')

    def plot_confusion_matrix(self, cm, model_name='random_forest'):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Fidèle', 'Churner'], yticklabels=['Fidèle', 'Churner'],
                    linewidths=0.5, cbar_kws={'label': 'Nombre de clients'})
        ax.set_xlabel('Prédiction', fontsize=12)
        ax.set_ylabel('Réalité', fontsize=12)
        ax.set_title(f'Matrice de confusion — {model_name.replace("_", " ").title()}',
                     fontsize=13, fontweight='bold')
        return self._save(fig, 'fig2_confusion_matrix.png')

    def plot_feature_importance(self, importance_series, top_n=15):
        top = importance_series.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#D32F2F' if i < 5 else '#F57C00' if i < 10 else '#FDD835'
                  for i in range(len(top))]
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor='white')
        ax.set_xlabel('Importance relative', fontsize=12)
        ax.set_title('Top features — Importance pour la prédiction de churn',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return self._save(fig, 'fig3_feature_importance.png')

    def plot_churn_by_contract(self, df):
        churn_rate = df.groupby('contract_type')['churn'].mean() * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#E53935', '#FB8C00', '#43A047']
        bars = ax.bar(churn_rate.index, churn_rate.values, color=colors, edgecolor='white', width=0.5)
        for bar, val in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Taux de churn (%)', fontsize=12)
        ax.set_xlabel('Type de contrat', fontsize=12)
        ax.set_title('Taux de churn par type de contrat', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        return self._save(fig, 'fig4_churn_by_contract.png')

    def plot_churn_by_channel(self, df):
        churn_rate = df.groupby('acquisition_channel')['churn'].mean() * 100
        fig, ax = plt.subplots(figsize=(9, 5))
        palette = {'Web': '#1565C0', 'Store': '#2E7D32', 'Agent': '#E65100', 'Referral': '#6A1B9A'}
        colors = [palette.get(c, 'grey') for c in churn_rate.index]
        ax.bar(churn_rate.index, churn_rate.values, color=colors, edgecolor='white', width=0.5)
        ax.axhline(df['churn'].mean() * 100, color='crimson', linestyle='--',
                   linewidth=1.5, label=f"Moyenne : {df['churn'].mean()*100:.1f}%")
        ax.set_ylabel('Taux de churn (%)', fontsize=12)
        ax.set_xlabel("Canal d'acquisition", fontsize=12)
        ax.set_title("Taux de churn par canal d'acquisition", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        return self._save(fig, 'fig5_churn_by_channel.png')

    def plot_monthly_charge_distribution(self, df):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df[df['churn'] == 0]['monthly_charge'], bins=40, alpha=0.6,
                color='#1976D2', label='Fidèle', density=True)
        ax.hist(df[df['churn'] == 1]['monthly_charge'], bins=40, alpha=0.6,
                color='#E53935', label='Churner', density=True)
        ax.set_xlabel('Montant mensuel (€)', fontsize=12)
        ax.set_ylabel('Densité', fontsize=12)
        ax.set_title('Distribution du montant mensuel — Churners vs Fidèles',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        return self._save(fig, 'fig6_monthly_charge_distribution.png')

    def plot_threshold_optimization(self, threshold_costs, optimal_threshold):
        thresholds, costs = zip(*threshold_costs)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(thresholds, costs, color='#1565C0', linewidth=2)
        ax.axvline(optimal_threshold, color='crimson', linestyle='--', linewidth=2,
                   label=f'Seuil optimal : {optimal_threshold:.2f}')
        ax.set_xlabel('Seuil de décision', fontsize=12)
        ax.set_ylabel('Coût total (FN×5 + FP×1)', fontsize=12)
        ax.set_title('Optimisation du seuil de décision (coût asymétrique)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        return self._save(fig, 'fig7_threshold_optimization.png')

    def plot_tenure_vs_churn(self, df):
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = [0, 12, 24, 48, 72, 121]
        labels = ['0-12m', '12-24m', '24-48m', '48-72m', '72m+']
        df = df.copy()
        df['tenure_bin'] = pd.cut(df['tenure_months'], bins=bins, labels=labels)
        churn_by_tenure = df.groupby('tenure_bin', observed=True)['churn'].mean() * 100
        ax.bar(churn_by_tenure.index, churn_by_tenure.values,
               color='#5C6BC0', edgecolor='white', width=0.6)
        ax.set_xlabel("Ancienneté client", fontsize=12)
        ax.set_ylabel("Taux de churn (%)", fontsize=12)
        ax.set_title("Taux de churn par tranche d'ancienneté", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        return self._save(fig, 'fig8_tenure_vs_churn.png')
