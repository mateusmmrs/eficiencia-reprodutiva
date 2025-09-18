"""
================================================================
EDA — ANÁLISE EXPLORATÓRIA DE DADOS (v2 — PADRONIZADO)
================================================================
Paleta executiva consistente, fontes uniformes, sem overlap
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# === CONFIG ===
DATA_PATH = Path(__file__).parent.parent / "data" / "rebanho_simulado.csv"
PLOTS_PATH = Path(__file__).parent.parent / "plots"
PLOTS_PATH.mkdir(exist_ok=True)

# Paleta executiva
PAL = {
    'dark': '#1B4332', 'mid': '#2D6A4F', 'green': '#40916C',
    'light': '#74C69D', 'pale': '#B7E4C7',
    'bg': '#FAFAFA', 'text': '#1A1A1A', 'muted': '#6B7280',
    'red': '#DC2626', 'amber': '#F59E0B', 'blue': '#3B82F6',
}
BARS = [PAL['dark'], PAL['mid'], PAL['green'], PAL['light'], PAL['pale']]

def setup_style():
    plt.rcParams.update({
        'figure.facecolor': PAL['bg'],
        'axes.facecolor': PAL['bg'],
        'axes.edgecolor': '#D1D5DB',
        'axes.labelcolor': PAL['text'],
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.color': PAL['muted'],
        'ytick.color': PAL['muted'],
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.color': PAL['text'],
        'font.family': 'sans-serif',
        'font.size': 11,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.4,
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
    })

def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def load_data():
    return pd.read_csv(DATA_PATH)


def plot_01(df):
    """Taxa de prenhez por categoria"""
    fig, ax = plt.subplots(figsize=(9, 5))
    stats = df.groupby("categoria")["resultado_prenhez"].agg(["mean", "count", "std"])
    stats["ci"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean", ascending=True)

    bars = ax.barh(stats.index, stats["mean"] * 100, xerr=stats["ci"] * 100,
                   color=BARS[:3], edgecolor='white', height=0.5, capsize=5,
                   error_kw=dict(lw=1.5, capthick=1.5, color=PAL['muted']))

    for bar, val in zip(bars, stats["mean"]):
        ax.text(bar.get_width() + 2.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontweight='bold', fontsize=13, color=PAL['text'])

    ax.set_xlabel("Taxa de Prenhez (%)", fontsize=12)
    ax.set_title("Taxa de Prenhez por Categoria Animal", pad=15)
    ax.set_xlim(0, 88)
    clean_ax(ax)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "01_taxa_prenhez_categoria.png")
    plt.close()


def plot_02(df):
    """ECC vs prenhez"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(2.0, 9.5, 0.5)
    df2 = df.copy()
    df2['ecc_bin'] = pd.cut(df2['ecc'], bins=bins)
    grouped = df2.groupby('ecc_bin', observed=True)['resultado_prenhez'].agg(['mean', 'count'])
    grouped = grouped[grouped['count'] >= 20]

    x = range(len(grouped))
    colors = [PAL['red'] if v < 0.55 else PAL['amber'] if v < 0.65
              else PAL['green'] if v < 0.75 else PAL['dark']
              for v in grouped['mean']]

    bars = ax.bar(x, grouped['mean'] * 100, color=colors, edgecolor='white', width=0.65)

    for i, (bar, val) in enumerate(zip(bars, grouped['mean'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.0%}', ha='center', fontsize=10, fontweight='bold', color=PAL['text'])

    ax.set_xticks(x)
    ax.set_xticklabels([f'{b.left:.1f}–{b.right:.1f}' for b in grouped.index],
                        rotation=45, ha='right', fontsize=10)
    ax.set_xlabel("Escore de Condição Corporal (ECC)", fontsize=12)
    ax.set_ylabel("Taxa de Prenhez (%)", fontsize=12)
    ax.set_title("Impacto do ECC na Taxa de Prenhez", pad=15)
    ax.set_ylim(0, 100)
    ax.axhline(y=65, color=PAL['muted'], linestyle='--', alpha=0.6, linewidth=1, label='Benchmark 65%')
    ax.legend(loc='lower right', fontsize=11)
    clean_ax(ax)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "02_ecc_vs_prenhez.png")
    plt.close()


def plot_03(df):
    """Curva DPP"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df_p = df[df['categoria'] != 'Novilha'].copy()
    bins = np.arange(30, 160, 10)
    df_p['dpp_bin'] = pd.cut(df_p['dpp'], bins=bins)
    grouped = df_p.groupby('dpp_bin', observed=True)['resultado_prenhez'].agg(['mean', 'count'])
    grouped = grouped[grouped['count'] >= 15]
    mid = [(b.left + b.right) / 2 for b in grouped.index]

    ax.plot(mid, grouped['mean'] * 100, color=PAL['dark'], linewidth=2.5,
            marker='o', markersize=8, markerfacecolor='white', markeredgecolor=PAL['dark'], markeredgewidth=2)
    ax.fill_between(mid, grouped['mean'] * 100, alpha=0.08, color=PAL['green'])
    ax.axvspan(50, 80, alpha=0.06, color=PAL['green'], label='Zona Ótima (50–80 DPP)')

    ax.set_xlabel("Dias Pós-Parto (DPP)", fontsize=12)
    ax.set_ylabel("Taxa de Prenhez (%)", fontsize=12)
    ax.set_title("Curva DPP × Probabilidade de Prenhez", pad=15)
    ax.legend(loc='upper right', fontsize=11)
    clean_ax(ax)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "03_curva_dpp.png")
    plt.close()


def plot_04(df):
    """Performance por touro"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    stats = df.groupby("touro").agg(taxa=("resultado_prenhez", "mean"), n=("resultado_prenhez", "count"))
    stats = stats.sort_values("taxa", ascending=False)

    colors = [PAL['dark'] if v >= 0.70 else PAL['green'] if v >= 0.65
              else PAL['amber'] if v >= 0.60 else PAL['red']
              for v in stats['taxa']]

    bars = ax.bar(range(len(stats)), stats['taxa'] * 100, color=colors, edgecolor='white', width=0.6)

    for bar, val, n in zip(bars, stats['taxa'], stats['n']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f'{val:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 4,
                f'n={n}', ha='center', fontsize=8, color='white', alpha=0.8)

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Taxa de Prenhez (%)", fontsize=12)
    ax.set_title("Performance Reprodutiva por Touro", pad=15)
    ax.axhline(y=67, color=PAL['muted'], linestyle='--', alpha=0.5, lw=1, label='Média Geral')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 85)
    clean_ax(ax)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "04_performance_touro.png")
    plt.close()


def plot_05(df):
    """THI impacto"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(55, 90, 3)
    df2 = df.copy()
    df2['thi_bin'] = pd.cut(df2['thi'], bins=bins)
    grouped = df2.groupby('thi_bin', observed=True)['resultado_prenhez'].agg(['mean', 'count'])
    grouped = grouped[grouped['count'] >= 20]
    mid = [(b.left + b.right) / 2 for b in grouped.index]

    ax.plot(mid, grouped['mean'] * 100, color=PAL['dark'], linewidth=2.5,
            marker='s', markersize=8, markerfacecolor='white', markeredgecolor=PAL['dark'], markeredgewidth=2)
    ax.axvspan(78, 92, alpha=0.08, color=PAL['red'], label='Estresse Calórico (THI > 78)')

    ax.set_xlabel("Índice de Temperatura e Umidade (THI)", fontsize=12)
    ax.set_ylabel("Taxa de Prenhez (%)", fontsize=12)
    ax.set_title("Impacto do Estresse Calórico na Reprodução", pad=15)
    ax.legend(loc='lower left', fontsize=11)
    clean_ax(ax)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "05_thi_impacto.png")
    plt.close()


def plot_06(df):
    """Heatmap de correlação"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = ['ecc', 'dpp', 'peso_kg', 'thi', 'idade_anos',
            'custo_protocolo', 'resultado_prenhez', 'perda_gestacional']
    labels = ['ECC', 'DPP', 'Peso (kg)', 'THI', 'Idade',
              'Custo Proto.', 'Prenhez', 'Perda Gest.']
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 130, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', linewidths=0.5,
                xticklabels=labels, yticklabels=labels,
                square=True, ax=ax, cbar_kws={'shrink': 0.8},
                annot_kws={'fontsize': 11})
    ax.set_title("Correlação entre Variáveis Produtivas e Reprodutivas", pad=15, fontsize=15)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "06_correlacao.png")
    plt.close()


def plot_07(df):
    """Comparação fazendas"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    stats = df.groupby("fazenda").agg(
        taxa=("resultado_prenhez", "mean"),
        ecc=("ecc", "mean"),
        custo=("custo_manutencao_dia", "mean"),
    ).sort_values("taxa", ascending=False)

    short = [f.replace('Fazenda ', '') for f in stats.index]

    # Taxa
    axes[0].barh(short, stats['taxa'] * 100, color=BARS, edgecolor='white', height=0.5)
    axes[0].set_xlabel("Taxa de Prenhez (%)", fontsize=11)
    axes[0].set_title("Taxa de Prenhez", fontsize=13, fontweight='bold')
    for i, v in enumerate(stats['taxa']):
        axes[0].text(v * 100 + 0.5, i, f'{v:.1%}', va='center', fontweight='bold', fontsize=11)

    # ECC
    axes[1].barh(short, stats['ecc'], color=BARS, edgecolor='white', height=0.5)
    axes[1].set_xlabel("ECC Médio", fontsize=11)
    axes[1].set_title("ECC Médio", fontsize=13, fontweight='bold')
    for i, v in enumerate(stats['ecc']):
        axes[1].text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold', fontsize=11)

    # Custo
    axes[2].barh(short, stats['custo'], color=BARS, edgecolor='white', height=0.5)
    axes[2].set_xlabel("R$/dia", fontsize=11)
    axes[2].set_title("Custo Manutenção/dia", fontsize=13, fontweight='bold')
    for i, v in enumerate(stats['custo']):
        axes[2].text(v + 0.05, i, f'R$ {v:.2f}', va='center', fontweight='bold', fontsize=11)

    for ax in axes:
        clean_ax(ax)

    plt.suptitle("Comparação entre Fazendas", fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "07_comparacao_fazendas.png")
    plt.close()


def main():
    setup_style()
    df = load_data()
    print("🎨 Regenerando gráficos EDA padronizados...")
    plot_01(df); print("   ✅ 01")
    plot_02(df); print("   ✅ 02")
    plot_03(df); print("   ✅ 03")
    plot_04(df); print("   ✅ 04")
    plot_05(df); print("   ✅ 05")
    plot_06(df); print("   ✅ 06")
    plot_07(df); print("   ✅ 07")
    print("✅ EDA concluída!")

if __name__ == "__main__":
    main()
