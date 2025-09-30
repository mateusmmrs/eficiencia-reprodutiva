"""
================================================================
MODELAGEM ECONÔMICA + ELASTICIDADE + SENSIBILIDADE + SIMULADOR
================================================================
ROI Reprodutivo, Elasticidade da Taxa de Prenhez,
Análise de Sensibilidade, Simulador Estratégico
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

COLORS = {
    'primary': '#1B4332', 'secondary': '#2D6A4F', 'accent': '#40916C',
    'light': '#74C69D', 'lighter': '#B7E4C7',
    'bg': '#FAFAFA', 'text': '#1A1A1A', 'muted': '#6B7280',
    'danger': '#DC2626', 'warning': '#F59E0B', 'info': '#3B82F6',
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


def load_data():
    return pd.read_csv(DATA_PATH)


# ================================================================
# 1. MODELAGEM ECONÔMICA
# ================================================================

def economic_model(df):
    """Calcular métricas econômicas por animal"""
    df = df.copy()

    # Receita esperada = prenhez × valor do bezerro (ajustado por perda)
    df['receita_esperada'] = (
        df['resultado_prenhez'] *
        (1 - df['perda_gestacional']) *
        df['valor_bezerro']
    )

    # Custo total = protocolo + manutenção × dias_abertos + nutricional
    df['custo_manutencao_total'] = df['custo_manutencao_dia'] * df['dias_abertos']
    df['custo_total'] = (
        df['custo_protocolo'] +
        df['custo_manutencao_total'] +
        df['custo_nutricional']
    )

    # Margem bruta
    df['margem_bruta'] = df['receita_esperada'] - df['custo_total']

    # ROI reprodutivo (%)
    df['roi'] = np.where(
        df['custo_total'] > 0,
        ((df['receita_esperada'] - df['custo_total']) / df['custo_total']) * 100,
        0
    )

    return df


def print_economic_summary(df):
    """Print economic summary tables"""
    print("\n💰 MODELAGEM ECONÔMICA")
    print("=" * 60)

    # Overall
    total = len(df)
    print(f"\n📊 RESUMO GERAL (n={total})")
    print(f"   Taxa de Prenhez:        {df['resultado_prenhez'].mean():.1%}")
    print(f"   Receita Média/Vaca:     R$ {df['receita_esperada'].mean():,.2f}")
    print(f"   Custo Médio/Vaca:       R$ {df['custo_total'].mean():,.2f}")
    print(f"   Margem Bruta Média:     R$ {df['margem_bruta'].mean():,.2f}")
    print(f"   ROI Médio:              {df['roi'].mean():.1f}%")
    print(f"   Dias Abertos Médios:    {df['dias_abertos'].mean():.0f}")
    print(f"   Receita Total Rebanho:  R$ {df['receita_esperada'].sum():,.2f}")
    print(f"   Custo Total Rebanho:    R$ {df['custo_total'].sum():,.2f}")
    print(f"   Margem Total Rebanho:   R$ {df['margem_bruta'].sum():,.2f}")

    # Per farm
    print("\n📊 POR FAZENDA:")
    farm_summary = df.groupby('fazenda').agg(
        n=('id_animal', 'count'),
        taxa_prenhez=('resultado_prenhez', 'mean'),
        receita_media=('receita_esperada', 'mean'),
        custo_medio=('custo_total', 'mean'),
        margem_media=('margem_bruta', 'mean'),
        roi_medio=('roi', 'mean'),
    ).round(2)
    print(farm_summary.to_string())

    return farm_summary


# ================================================================
# 2. ELASTICIDADE DA TAXA DE PRENHEZ
# ================================================================

def elasticity_analysis(df):
    """Calculate elasticity of pregnancy rate on gross margin"""
    print("\n\n📐 ELASTICIDADE DA TAXA DE PRENHEZ")
    print("=" * 60)

    base_rate = df['resultado_prenhez'].mean()
    base_margin = df['margem_bruta'].mean()
    avg_bezerro = df['valor_bezerro'].mean()
    avg_custo_manut_dia = df['custo_manutencao_dia'].mean()

    scenarios = [1, 2, 3, 5, 10]
    results = []

    for delta_pp in scenarios:
        new_rate = base_rate + (delta_pp / 100)

        # New margin calculation
        new_receita = new_rate * avg_bezerro * 0.94  # 6% perda
        new_dias_abertos = 365 * (1 - new_rate)  # approx
        new_custo_manut = avg_custo_manut_dia * new_dias_abertos
        new_custo_total = df['custo_protocolo'].mean() + new_custo_manut + df['custo_nutricional'].mean()
        new_margin = new_receita - new_custo_total

        pct_margin_change = ((new_margin - base_margin) / abs(base_margin)) * 100
        pct_rate_change = (delta_pp / (base_rate * 100)) * 100
        elasticity = pct_margin_change / pct_rate_change if pct_rate_change != 0 else 0

        results.append({
            'delta_pp': delta_pp,
            'nova_taxa': new_rate,
            'margem_anterior': base_margin,
            'nova_margem': new_margin,
            'var_margem_pct': pct_margin_change,
            'elasticidade': elasticity,
            'ganho_por_vaca': new_margin - base_margin,
            'ganho_1000_vacas': (new_margin - base_margin) * 1000,
        })

    results_df = pd.DataFrame(results)

    print(f"\n   Base: Taxa = {base_rate:.1%} | Margem = R$ {base_margin:,.2f}/vaca")
    print(f"\n   {'Δ pp':>6} | {'Nova Taxa':>10} | {'Δ Margem':>12} | {'Elasticidade':>12} | {'Ganho/vaca':>12} | {'Ganho 1000 vacas':>18}")
    print("   " + "-" * 85)
    for _, r in results_df.iterrows():
        print(f"   {r['delta_pp']:>+4.0f}pp | {r['nova_taxa']:>9.1%} | {r['var_margem_pct']:>+10.1f}% | {r['elasticidade']:>11.2f}x | R$ {r['ganho_por_vaca']:>9,.2f} | R$ {r['ganho_1000_vacas']:>14,.2f}")

    return results_df


def plot_elasticity(results_df):
    """Visualize elasticity results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Gain per cow
    axes[0].bar(results_df['delta_pp'].astype(str) + 'pp',
                results_df['ganho_por_vaca'],
                color=COLORS['accent'], edgecolor='white', width=0.55)
    for i, v in enumerate(results_df['ganho_por_vaca']):
        axes[0].text(i, v + 5, f'R$ {v:,.0f}', ha='center', fontweight='bold', fontsize=11)
    axes[0].set_xlabel("Aumento na Taxa de Prenhez", fontsize=12)
    axes[0].set_ylabel("Ganho por Vaca (R$)", fontsize=12)
    axes[0].set_title("Ganho Marginal por Vaca Exposta", pad=15)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Gain for 1000 cows
    axes[1].bar(results_df['delta_pp'].astype(str) + 'pp',
                results_df['ganho_1000_vacas'],
                color=COLORS['primary'], edgecolor='white', width=0.55)
    for i, v in enumerate(results_df['ganho_1000_vacas']):
        axes[1].text(i, v + 2000, f'R$ {v:,.0f}', ha='center', fontweight='bold', fontsize=10)
    axes[1].set_xlabel("Aumento na Taxa de Prenhez", fontsize=12)
    axes[1].set_ylabel("Ganho Total (R$)", fontsize=12)
    axes[1].set_title("Impacto Financeiro — Rebanho de 1.000 Vacas", pad=15)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R$ {x:,.0f}'))
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle("Elasticidade Econômica da Taxa de Prenhez", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "11_elasticidade.png")
    plt.close()
    print("   ✅ 11 — Elasticidade")


# ================================================================
# 3. ANÁLISE DE SENSIBILIDADE
# ================================================================

def sensitivity_analysis(df):
    """Multi-variable sensitivity analysis"""
    print("\n\n🔀 ANÁLISE DE SENSIBILIDADE")
    print("=" * 60)

    base_margin = df['margem_bruta'].mean()
    base_rate = df['resultado_prenhez'].mean()
    base_bezerro = df['valor_bezerro'].mean()
    base_custo_dia = df['custo_manutencao_dia'].mean()
    base_perda = df[df['resultado_prenhez']==1]['perda_gestacional'].mean()

    # === Tornado chart ===
    variations = {
        'Taxa de Prenhez': (0.55, 0.75),
        'Valor do Bezerro': (1800, 3200),
        'Custo Manutenção/dia': (5, 12),
        'Perda Gestacional': (0.03, 0.12),
    }

    tornado_data = []
    for var, (low, high) in variations.items():
        if var == 'Taxa de Prenhez':
            margin_low = low * base_bezerro * (1 - base_perda) - (df['custo_protocolo'].mean() + base_custo_dia * 365 * (1 - low) + df['custo_nutricional'].mean())
            margin_high = high * base_bezerro * (1 - base_perda) - (df['custo_protocolo'].mean() + base_custo_dia * 365 * (1 - high) + df['custo_nutricional'].mean())
        elif var == 'Valor do Bezerro':
            margin_low = base_rate * low * (1 - base_perda) - df['custo_total'].mean()
            margin_high = base_rate * high * (1 - base_perda) - df['custo_total'].mean()
        elif var == 'Custo Manutenção/dia':
            margin_low = df['receita_esperada'].mean() - (df['custo_protocolo'].mean() + low * df['dias_abertos'].mean() + df['custo_nutricional'].mean())
            margin_high = df['receita_esperada'].mean() - (df['custo_protocolo'].mean() + high * df['dias_abertos'].mean() + df['custo_nutricional'].mean())
        elif var == 'Perda Gestacional':
            margin_low = base_rate * base_bezerro * (1 - low) - df['custo_total'].mean()
            margin_high = base_rate * base_bezerro * (1 - high) - df['custo_total'].mean()

        tornado_data.append({
            'variable': var,
            'low_val': low,
            'high_val': high,
            'margin_low': margin_low,
            'margin_high': margin_high,
            'range': abs(margin_high - margin_low),
        })

    tornado_df = pd.DataFrame(tornado_data).sort_values('range')

    print(f"\n   Base margin: R$ {base_margin:,.2f}")
    print(f"\n   {'Variável':<25} | {'Cenário Baixo':>15} | {'Cenário Alto':>15} | {'Range':>12}")
    print("   " + "-" * 75)
    for _, r in tornado_df.iterrows():
        print(f"   {r['variable']:<25} | R$ {r['margin_low']:>11,.2f} | R$ {r['margin_high']:>11,.2f} | R$ {r['range']:>8,.2f}")

    # === Heatmap: Taxa prenhez × Valor bezerro ===
    rates = np.arange(0.50, 0.80, 0.02)
    values = np.arange(1800, 3400, 200)
    heatmap_data = np.zeros((len(rates), len(values)))

    for i, rate in enumerate(rates):
        for j, val in enumerate(values):
            receita = rate * val * (1 - base_perda)
            custo = df['custo_protocolo'].mean() + base_custo_dia * 365 * (1 - rate) + df['custo_nutricional'].mean()
            heatmap_data[i, j] = receita - custo

    return tornado_df, rates, values, heatmap_data


def plot_tornado(tornado_df, base_margin):
    """Tornado chart"""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    y_pos = range(len(tornado_df))
    for i, (_, row) in enumerate(tornado_df.iterrows()):
        left = min(row['margin_low'], row['margin_high'])
        width = abs(row['margin_high'] - row['margin_low'])
        color = COLORS['accent'] if row['margin_high'] > row['margin_low'] else COLORS['danger']
        ax.barh(i, width, left=left, color=color, edgecolor='white', height=0.5, alpha=0.85)

    ax.axvline(x=base_margin, color=COLORS['text'], linewidth=1.5, linestyle='--', label=f'Base: R$ {base_margin:,.0f}')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tornado_df['variable'], fontsize=11)
    ax.set_xlabel("Margem Bruta por Vaca (R$)", fontsize=12)
    ax.set_title("Análise de Sensibilidade — Gráfico Tornado", pad=15)
    ax.legend(loc='upper right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R$ {x:,.0f}'))

    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "12_tornado.png")
    plt.close()
    print("   ✅ 12 — Tornado")


def plot_heatmap_sensitivity(rates, values, heatmap_data):
    """Heatmap: taxa prenhez × valor bezerro → margem"""
    fig, ax = plt.subplots(figsize=(11, 7))

    rate_labels = [f'{r:.0%}' for r in rates]
    value_labels = [f'R$ {v/1000:.1f}k' for v in values]

    sns.heatmap(heatmap_data, xticklabels=value_labels, yticklabels=rate_labels,
                annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Margem Bruta (R$/vaca)'},
                annot_kws={'fontsize': 10})

    ax.set_xlabel("Valor Médio do Bezerro", fontsize=12)
    ax.set_ylabel("Taxa de Prenhez", fontsize=12)
    ax.set_title("Heatmap de Margem Bruta — Cenários Combinados", pad=15)
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "13_heatmap_sensibilidade.png")
    plt.close()
    print("   ✅ 13 — Heatmap sensibilidade")


def plot_sensitivity_curve(df):
    """Sensitivity curve: margem vs taxa de prenhez"""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    base_bezerro = df['valor_bezerro'].mean()
    base_perda = df[df['resultado_prenhez']==1]['perda_gestacional'].mean()
    base_custo_dia = df['custo_manutencao_dia'].mean()

    rates = np.arange(0.40, 0.86, 0.01)
    margins = []
    for rate in rates:
        receita = rate * base_bezerro * (1 - base_perda)
        custo = df['custo_protocolo'].mean() + base_custo_dia * 365 * (1 - rate) + df['custo_nutricional'].mean()
        margins.append(receita - custo)

    ax.plot(rates * 100, margins, color=COLORS['primary'], linewidth=2.5)
    ax.fill_between(rates * 100, margins, alpha=0.08, color=COLORS['accent'])

    # Breakeven
    breakeven_idx = np.argmin(np.abs(np.array(margins)))
    if margins[breakeven_idx] < 50:
        ax.axvline(x=rates[breakeven_idx] * 100, color=COLORS['danger'], linestyle='--',
                   alpha=0.7, label=f'Breakeven ≈ {rates[breakeven_idx]:.0%}')

    # Current position
    current_rate = df['resultado_prenhez'].mean()
    current_margin = df['margem_bruta'].mean()
    ax.scatter([current_rate * 100], [current_margin], s=120, color=COLORS['danger'],
               zorder=5, label=f'Atual: {current_rate:.1%}')

    ax.set_xlabel("Taxa de Prenhez (%)", fontsize=12)
    ax.set_ylabel("Margem Bruta por Vaca (R$)", fontsize=12)
    ax.set_title("Curva de Sensibilidade — Taxa de Prenhez vs Margem", pad=15)
    ax.legend(loc='upper left', fontsize=11)
    ax.axhline(y=0, color=COLORS['muted'], linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R$ {x:,.0f}'))

    plt.tight_layout()
    fig.savefig(PLOTS_PATH / "14_curva_sensibilidade.png")
    plt.close()
    print("   ✅ 14 — Curva sensibilidade")


# ================================================================
# 4. SIMULADOR ESTRATÉGICO
# ================================================================

def strategic_simulator(df):
    """Simulate impact of +1pp increase in pregnancy rate"""
    print("\n\n🎯 SIMULADOR ESTRATÉGICO")
    print("=" * 60)

    base_rate = df['resultado_prenhez'].mean()
    avg_bezerro = df['valor_bezerro'].mean()
    base_perda = df[df['resultado_prenhez']==1]['perda_gestacional'].mean()
    base_custo_dia = df['custo_manutencao_dia'].mean()

    def calc_margin(rate):
        receita = rate * avg_bezerro * (1 - base_perda)
        custo = df['custo_protocolo'].mean() + base_custo_dia * 365 * (1 - rate) + df['custo_nutricional'].mean()
        return receita - custo

    base_margin = calc_margin(base_rate)

    print(f"\n   📌 'Qual o impacto financeiro anual ao aumentar 1pp na taxa de prenhez?'")
    print(f"\n   Base: Taxa = {base_rate:.1%} | Margem = R$ {base_margin:,.2f}/vaca")

    new_margin = calc_margin(base_rate + 0.01)
    gain_per_cow = new_margin - base_margin

    print(f"\n   ✅ Ganho por vaca exposta:       R$ {gain_per_cow:,.2f}")
    print(f"   ✅ Ganho para 500 vacas:          R$ {gain_per_cow * 500:,.2f}")
    print(f"   ✅ Ganho para 1.000 vacas:        R$ {gain_per_cow * 1000:,.2f}")
    print(f"   ✅ Ganho para 5.000 vacas:        R$ {gain_per_cow * 5000:,.2f}")

    # Annual projection
    print(f"\n   📊 PROJEÇÃO ACUMULADA (5 anos, +1pp/ano):")
    accumulated = 0
    for year in range(1, 6):
        rate = base_rate + (year * 0.01)
        margin = calc_margin(rate)
        annual_gain = (margin - base_margin) * 1000
        accumulated += annual_gain
        print(f"      Ano {year}: Taxa={rate:.1%} | Ganho anual: R$ {annual_gain:>10,.2f} | Acumulado: R$ {accumulated:>12,.2f}")

    return gain_per_cow


def main():
    print("\n💰 ANÁLISE ECONÔMICA COMPLETA")
    print("=" * 60)

    df = load_data()

    # 1. Economic model
    df = economic_model(df)
    farm_summary = print_economic_summary(df)

    # Save enriched dataset
    df.to_csv(DATA_PATH.parent / "rebanho_economico.csv", index=False)

    # 2. Elasticity
    elast_df = elasticity_analysis(df)
    plot_elasticity(elast_df)

    # 3. Sensitivity
    tornado_df, rates, values, heatmap = sensitivity_analysis(df)
    plot_tornado(tornado_df, df['margem_bruta'].mean())
    plot_heatmap_sensitivity(rates, values, heatmap)
    plot_sensitivity_curve(df)

    # 4. Strategic Simulator
    strategic_simulator(df)

    # Save summary for dashboard
    summary_data = {
        'taxa_prenhez': df['resultado_prenhez'].mean(),
        'margem_bruta_media': df['margem_bruta'].mean(),
        'roi_medio': df['roi'].mean(),
        'dias_abertos_medio': df['dias_abertos'].mean(),
        'receita_total': df['receita_esperada'].sum(),
        'custo_total': df['custo_total'].sum(),
        'n_animais': len(df),
    }
    pd.DataFrame([summary_data]).to_csv(DATA_PATH.parent / "summary_kpis.csv", index=False)
    farm_summary.to_csv(DATA_PATH.parent / "summary_fazendas.csv")

    print(f"\n✅ Análise econômica completa! Gráficos em: {PLOTS_PATH}")


if __name__ == "__main__":
    main()
