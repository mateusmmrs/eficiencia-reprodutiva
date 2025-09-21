"""
================================================================
MODELAGEM PREDITIVA — REPRODUÇÃO BOVINA
================================================================
Logistic Regression + Random Forest
Avaliação: ROC-AUC, Confusion Matrix, Precision/Recall
Validação Cruzada + Interpretação Biológica
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# === CONFIG ===
DATA_PATH = Path(__file__).parent.parent / "data" / "rebanho_simulado.csv"
PLOTS_PATH = Path(__file__).parent.parent / "plots"
MODELS_PATH = Path(__file__).parent.parent / "models"
MODELS_PATH.mkdir(exist_ok=True)

COLORS = {
    'primary': '#1B4332', 'accent': '#40916C',
    'light': '#74C69D', 'danger': '#DC2626',
    'bg': '#FAFAFA', 'text': '#1A1A1A', 'muted': '#6B7280',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'], 'axes.facecolor': COLORS['bg'],
    'axes.edgecolor': '#D1D5DB', 'font.family': 'sans-serif',
    'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.4,
    'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'xtick.color': COLORS['muted'], 'ytick.color': COLORS['muted'],
    'legend.fontsize': 11, 'legend.framealpha': 0.9,
})


def prepare_features(df):
    """Prepare feature matrix and target"""
    df = df.copy()

    # Encode categoricals
    le_raca = LabelEncoder()
    le_cat = LabelEncoder()
    le_proto = LabelEncoder()
    le_touro = LabelEncoder()

    df['raca_enc'] = le_raca.fit_transform(df['raca'])
    df['categoria_enc'] = le_cat.fit_transform(df['categoria'])
    df['protocolo_enc'] = le_proto.fit_transform(df['protocolo'])
    df['touro_enc'] = le_touro.fit_transform(df['touro'])

    features = ['ecc', 'dpp', 'peso_kg', 'thi', 'idade_anos',
                'raca_enc', 'categoria_enc', 'protocolo_enc', 'touro_enc']
    feature_names = ['ECC', 'DPP', 'Peso (kg)', 'THI', 'Idade',
                     'Raça', 'Categoria', 'Protocolo', 'Touro']

    X = df[features].values
    y = df['resultado_prenhez'].values

    return X, y, features, feature_names


def train_and_evaluate(X, y, feature_names):
    """Train both models and evaluate"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # === Logistic Regression ===
    print("\n🔬 REGRESSÃO LOGÍSTICA")
    print("-" * 40)

    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train)

    y_pred_lr = lr.predict(X_test_sc)
    y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

    auc_lr = roc_auc_score(y_test, y_prob_lr)
    print(f"   ROC-AUC: {auc_lr:.4f}")
    print(f"\n{classification_report(y_test, y_pred_lr, target_names=['Não Prenha', 'Prenha'])}")

    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_lr = cross_val_score(lr, X_train_sc, y_train, cv=cv, scoring='roc_auc')
    print(f"   CV ROC-AUC: {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")

    # Coefficients interpretation
    print("\n   📋 Coeficientes (interpretação):")
    coef_df = pd.DataFrame({
        'Variable': feature_names,
        'Coefficient': lr.coef_[0],
        'Odds Ratio': np.exp(lr.coef_[0])
    }).sort_values('Coefficient', ascending=False)
    print(coef_df.to_string(index=False))

    # === Random Forest ===
    print("\n\n🌲 RANDOM FOREST")
    print("-" * 40)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    auc_rf = roc_auc_score(y_test, y_prob_rf)
    print(f"   ROC-AUC: {auc_rf:.4f}")
    print(f"\n{classification_report(y_test, y_pred_rf, target_names=['Não Prenha', 'Prenha'])}")

    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"   CV ROC-AUC: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

    # Feature importance
    print("\n   📋 Importância das Features:")
    imp_df = pd.DataFrame({
        'Variable': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(imp_df.to_string(index=False))

    # Save models
    joblib.dump(lr, MODELS_PATH / "logistic_regression.pkl")
    joblib.dump(rf, MODELS_PATH / "random_forest.pkl")
    joblib.dump(scaler, MODELS_PATH / "scaler.pkl")
    print(f"\n   💾 Modelos salvos em: {MODELS_PATH}")

    return {
        'lr': lr, 'rf': rf, 'scaler': scaler,
        'X_test': X_test, 'X_test_sc': X_test_sc, 'y_test': y_test,
        'y_prob_lr': y_prob_lr, 'y_prob_rf': y_prob_rf,
        'y_pred_lr': y_pred_lr, 'y_pred_rf': y_pred_rf,
        'coef_df': coef_df, 'imp_df': imp_df,
        'auc_lr': auc_lr, 'auc_rf': auc_rf,
        'cv_lr': cv_scores_lr, 'cv_rf': cv_scores_rf,
    }


def plot_roc_curves(results):
    """ROC curves comparison"""
    fig, ax = plt.subplots(figsize=(9, 7))

    fpr_lr, tpr_lr, _ = roc_curve(results['y_test'], results['y_prob_lr'])
    fpr_rf, tpr_rf, _ = roc_curve(results['y_test'], results['y_prob_rf'])

    ax.plot(fpr_lr, tpr_lr, color=COLORS['primary'], linewidth=2.5,
            label=f"Logistic Regression (AUC = {results['auc_lr']:.3f})")
    ax.plot(fpr_rf, tpr_rf, color=COLORS['accent'], linewidth=2.5,
            label=f"Random Forest (AUC = {results['auc_rf']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Aleatório')

    ax.set_xlabel("Taxa de Falso Positivo", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiro Positivo", fontsize=12)
    ax.set_title("Curva ROC — Comparação de Modelos", pad=15)
    ax.legend(loc='lower right', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "08_roc_curves.png")
    plt.close()
    print("   ✅ 08 — Curvas ROC")


def plot_confusion_matrices(results):
    """Confusion matrix for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, y_pred, title in [
        (axes[0], results['y_pred_lr'], f"Regressão Logística\nAUC = {results['auc_lr']:.3f}"),
        (axes[1], results['y_pred_rf'], f"Random Forest\nAUC = {results['auc_rf']:.3f}"),
    ]:
        cm = confusion_matrix(results['y_test'], y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=['Não Prenha', 'Prenha'],
                    yticklabels=['Não Prenha', 'Prenha'],
                    linewidths=0.5, square=True,
                    annot_kws={'fontsize': 14})
        ax.set_xlabel("Predição", fontsize=12)
        ax.set_ylabel("Real", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=11)

    plt.suptitle("Matrizes de Confusão", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "09_confusion_matrix.png")
    plt.close()
    print("   ✅ 09 — Matrizes de confusão")


def plot_feature_importance(results):
    """Feature importance comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Logistic Regression coefficients
    coef = results['coef_df'].sort_values('Coefficient')
    colors = [COLORS['primary'] if v >= 0 else COLORS['danger'] for v in coef['Coefficient']]
    axes[0].barh(coef['Variable'], coef['Coefficient'], color=colors, edgecolor='white', height=0.5)
    axes[0].set_title("Coeficientes — Regressão Logística", fontsize=13, fontweight='bold')
    axes[0].axvline(x=0, color=COLORS['muted'], linewidth=0.5)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].tick_params(labelsize=11)
    for i, (var, val) in enumerate(zip(coef['Variable'], coef['Coefficient'])):
        axes[0].text(val + 0.01 * np.sign(val), i, f'{val:.3f}', va='center', fontsize=10)

    # Random Forest importance
    imp = results['imp_df'].sort_values('Importance')
    axes[1].barh(imp['Variable'], imp['Importance'] * 100, color=COLORS['accent'],
                 edgecolor='white', height=0.5)
    axes[1].set_title("Importância — Random Forest", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Importância (%)", fontsize=12)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].tick_params(labelsize=11)
    for i, (var, val) in enumerate(zip(imp['Variable'], imp['Importance'])):
        axes[1].text(val * 100 + 0.3, i, f'{val:.1%}', va='center', fontsize=10)

    plt.suptitle("Interpretação dos Modelos", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "10_feature_importance.png")
    plt.close()
    print("   ✅ 10 — Importância das features")


def main():
    print("\n🧠 MODELAGEM PREDITIVA")
    print("=" * 50)

    df = pd.read_csv(DATA_PATH)
    X, y, features, feature_names = prepare_features(df)

    print(f"📊 Features: {len(features)} | Amostras: {len(X)}")
    print(f"📊 Classe positiva (prenha): {y.mean():.1%}")

    results = train_and_evaluate(X, y, feature_names)

    print("\n🎨 Gerando gráficos de modelagem...")
    plot_roc_curves(results)
    plot_confusion_matrices(results)
    plot_feature_importance(results)

    # Summary
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS MODELOS")
    print(f"   Logistic Regression — AUC: {results['auc_lr']:.4f} | CV: {results['cv_lr'].mean():.4f}")
    print(f"   Random Forest       — AUC: {results['auc_rf']:.4f} | CV: {results['cv_rf'].mean():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
