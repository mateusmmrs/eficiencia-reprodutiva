"""
================================================================
DASHBOARD EXECUTIVO — REPRODUÇÃO BOVINA
================================================================
Gera HTML interativo com Plotly para visualização executiva.
3 páginas: Visão Geral, Inteligência Reprodutiva, Inteligência Econômica
================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json

# === CONFIG ===
DATA_PATH = Path(__file__).parent.parent / "data"
DASHBOARD_PATH = Path(__file__).parent.parent / "dashboard"
DASHBOARD_PATH.mkdir(exist_ok=True)

COLORS = {
    'bg': '#0F1923',
    'card': '#1A2634',
    'border': '#2A3A4A',
    'primary': '#10B981',
    'secondary': '#3B82F6',
    'danger': '#EF4444',
    'warning': '#F59E0B',
    'text': '#F8FAFC',
    'muted': '#94A3B8',
    'accent_green': '#059669',
    'accent_blue': '#2563EB',
}


def load_data():
    df = pd.read_csv(DATA_PATH / "rebanho_economico.csv")
    kpis = pd.read_csv(DATA_PATH / "summary_kpis.csv").iloc[0]
    farms = pd.read_csv(DATA_PATH / "summary_fazendas.csv")
    return df, kpis, farms


def create_kpi_card(title, value, subtitle="", color="#10B981"):
    return f"""
    <div style="background:{COLORS['card']}; border:1px solid {COLORS['border']};
         border-radius:12px; padding:24px; text-align:center; border-top:3px solid {color};">
        <div style="font-size:13px; color:{COLORS['muted']}; text-transform:uppercase;
             letter-spacing:1px; margin-bottom:8px;">{title}</div>
        <div style="font-size:32px; font-weight:800; color:{color};">{value}</div>
        <div style="font-size:12px; color:{COLORS['muted']}; margin-top:4px;">{subtitle}</div>
    </div>"""


def build_dashboard(df, kpis, farms):
    """Build complete HTML dashboard"""

    taxa_prenhez = kpis['taxa_prenhez']
    margem = kpis['margem_bruta_media']
    roi = kpis['roi_medio']
    dias_abertos = kpis['dias_abertos_medio']

    # === CHART 1: Taxa prenhez por fazenda ===
    farm_stats = df.groupby('fazenda')['resultado_prenhez'].mean().sort_values()
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=farm_stats.index,
        x=farm_stats.values * 100,
        orientation='h',
        marker_color=[COLORS['primary'] if v > 0.66 else COLORS['warning'] for v in farm_stats.values],
        text=[f'{v:.1%}' for v in farm_stats.values],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=13, family='Inter'),
    ))
    fig1.update_layout(
        title=dict(text='Taxa de Prenhez por Fazenda', font=dict(size=16, color=COLORS['text'])),
        plot_bgcolor=COLORS['card'], paper_bgcolor=COLORS['card'],
        xaxis=dict(title='%', range=[0, 80], gridcolor=COLORS['border'], color=COLORS['muted']),
        yaxis=dict(color=COLORS['muted']),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        font=dict(family='Inter', color=COLORS['text']),
    )

    # === CHART 2: ECC vs prenhez ===
    bins = np.arange(2.5, 9.0, 0.5)
    df['ecc_bin'] = pd.cut(df['ecc'], bins=bins)
    ecc_stats = df.groupby('ecc_bin', observed=True)['resultado_prenhez'].mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[f'{b.left:.1f}-{b.right:.1f}' for b in ecc_stats.index],
        y=ecc_stats.values * 100,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=10, color=COLORS['primary'], line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(16,185,129,0.1)',
    ))
    fig2.update_layout(
        title=dict(text='ECC vs Taxa de Prenhez', font=dict(size=16, color=COLORS['text'])),
        plot_bgcolor=COLORS['card'], paper_bgcolor=COLORS['card'],
        xaxis=dict(title='ECC', gridcolor=COLORS['border'], color=COLORS['muted']),
        yaxis=dict(title='Taxa de Prenhez (%)', gridcolor=COLORS['border'], color=COLORS['muted']),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        font=dict(family='Inter', color=COLORS['text']),
    )
    df.drop(columns=['ecc_bin'], inplace=True, errors='ignore')

    # === CHART 3: Performance por touro ===
    touro_stats = df.groupby('touro')['resultado_prenhez'].mean().sort_values(ascending=False)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=touro_stats.index,
        y=touro_stats.values * 100,
        marker_color=[COLORS['primary'] if v > 0.68 else COLORS['secondary'] if v > 0.63
                      else COLORS['warning'] if v > 0.58 else COLORS['danger']
                      for v in touro_stats.values],
        text=[f'{v:.0%}' for v in touro_stats.values],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11),
    ))
    fig3.update_layout(
        title=dict(text='Performance Reprodutiva por Touro', font=dict(size=16, color=COLORS['text'])),
        plot_bgcolor=COLORS['card'], paper_bgcolor=COLORS['card'],
        xaxis=dict(gridcolor=COLORS['border'], color=COLORS['muted']),
        yaxis=dict(title='Taxa de Prenhez (%)', range=[0, 85], gridcolor=COLORS['border'], color=COLORS['muted']),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        font=dict(family='Inter', color=COLORS['text']),
    )

    # === CHART 4: Elasticidade ===
    base_rate = taxa_prenhez
    avg_bezerro = df['valor_bezerro'].mean()
    base_perda = 0.064
    base_custo_dia = df['custo_manutencao_dia'].mean()

    def calc_margin(rate):
        receita = rate * avg_bezerro * (1 - base_perda)
        custo = df['custo_protocolo'].mean() + base_custo_dia * 365 * (1 - rate) + df['custo_nutricional'].mean()
        return receita - custo

    rates = np.arange(0.45, 0.86, 0.01)
    margins = [calc_margin(r) for r in rates]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=rates * 100, y=margins,
        mode='lines', line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy', fillcolor='rgba(16,185,129,0.1)',
        name='Margem Bruta',
    ))
    fig4.add_trace(go.Scatter(
        x=[base_rate * 100], y=[calc_margin(base_rate)],
        mode='markers', marker=dict(size=14, color=COLORS['danger'], symbol='diamond'),
        name=f'Atual ({base_rate:.0%})',
    ))
    fig4.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'], opacity=0.5)
    fig4.update_layout(
        title=dict(text='Curva de Sensibilidade — Margem vs Taxa de Prenhez', font=dict(size=16, color=COLORS['text'])),
        plot_bgcolor=COLORS['card'], paper_bgcolor=COLORS['card'],
        xaxis=dict(title='Taxa de Prenhez (%)', gridcolor=COLORS['border'], color=COLORS['muted']),
        yaxis=dict(title='Margem Bruta (R$/vaca)', gridcolor=COLORS['border'], color=COLORS['muted']),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
        font=dict(family='Inter', color=COLORS['text']),
        legend=dict(font=dict(color=COLORS['text'])),
    )

    # === CHART 5: Heatmap sensibilidade ===
    sim_rates = np.arange(0.50, 0.78, 0.02)
    sim_values = np.arange(1800, 3400, 200)
    heatmap_data = np.zeros((len(sim_rates), len(sim_values)))
    for i, rate in enumerate(sim_rates):
        for j, val in enumerate(sim_values):
            receita = rate * val * (1 - base_perda)
            custo = df['custo_protocolo'].mean() + base_custo_dia * 365 * (1 - rate) + df['custo_nutricional'].mean()
            heatmap_data[i, j] = receita - custo

    fig5 = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f'R${v:,.0f}' for v in sim_values],
        y=[f'{r:.0%}' for r in sim_rates],
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(heatmap_data, 0).astype(int),
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorbar=dict(title='R$/vaca', tickfont=dict(color=COLORS['muted']), titlefont=dict(color=COLORS['muted'])),
    ))
    fig5.update_layout(
        title=dict(text='Heatmap de Margem — Taxa de Prenhez × Valor do Bezerro', font=dict(size=16, color=COLORS['text'])),
        plot_bgcolor=COLORS['card'], paper_bgcolor=COLORS['card'],
        xaxis=dict(title='Valor do Bezerro', color=COLORS['muted']),
        yaxis=dict(title='Taxa de Prenhez', color=COLORS['muted']),
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
        font=dict(family='Inter', color=COLORS['text']),
    )

    # === CHART 6: Distribuição de margem por fazenda ===
    fig6 = go.Figure()
    for fazenda in sorted(df['fazenda'].unique()):
        data = df[df['fazenda'] == fazenda]['margem_bruta']
        fig6.add_trace(go.Box(y=data, name=fazenda.replace('Fazenda ', ''),
                              marker_color=COLORS['primary'], line_color=COLORS['primary']))
    fig6.update_layout(
        title=dict(text='Distribuição de Margem Bruta por Fazenda', font=dict(size=16, color=COLORS['text'])),
        plot_bgcolor=COLORS['card'], paper_bgcolor=COLORS['card'],
        yaxis=dict(title='Margem Bruta (R$)', gridcolor=COLORS['border'], color=COLORS['muted']),
        xaxis=dict(color=COLORS['muted']),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        font=dict(family='Inter', color=COLORS['text']),
        showlegend=False,
    )

    # === BUILD HTML ===
    gain_1pp = calc_margin(base_rate + 0.01) - calc_margin(base_rate)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Executivo — Eficiência Reprodutiva</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', sans-serif; background: {COLORS['bg']}; color: {COLORS['text']}; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        .header {{ text-align: center; padding: 40px 0 32px; }}
        .header h1 {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; }}
        .header p {{ color: {COLORS['muted']}; font-size: 14px; }}
        .tabs {{ display: flex; gap: 4px; justify-content: center; margin-bottom: 32px; }}
        .tab {{ padding: 10px 24px; border-radius: 8px; background: {COLORS['card']};
                border: 1px solid {COLORS['border']}; color: {COLORS['muted']};
                cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.2s; }}
        .tab:hover {{ border-color: {COLORS['primary']}; color: {COLORS['text']}; }}
        .tab.active {{ background: {COLORS['primary']}; border-color: {COLORS['primary']}; color: white; }}
        .page {{ display: none; }}
        .page.active {{ display: block; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }}
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 32px; }}
        .chart-card {{ background: {COLORS['card']}; border: 1px solid {COLORS['border']};
                      border-radius: 12px; padding: 16px; overflow: hidden; }}
        .chart-full {{ grid-column: 1 / -1; }}
        .insight-box {{ background: {COLORS['card']}; border: 1px solid {COLORS['border']};
                       border-radius: 12px; padding: 20px; margin-bottom: 20px;
                       border-left: 4px solid {COLORS['primary']}; }}
        .insight-box h3 {{ font-size: 14px; color: {COLORS['primary']}; margin-bottom: 8px; }}
        .insight-box p {{ font-size: 13px; color: {COLORS['muted']}; line-height: 1.6; }}
        .simulator {{ background: {COLORS['card']}; border: 1px solid {COLORS['primary']};
                     border-radius: 12px; padding: 24px; text-align: center; }}
        .simulator h3 {{ font-size: 18px; margin-bottom: 16px; }}
        .sim-result {{ font-size: 36px; font-weight: 800; color: {COLORS['primary']}; }}
        .sim-detail {{ color: {COLORS['muted']}; font-size: 13px; margin-top: 8px; }}
        @media (max-width: 768px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🐄 Dashboard Executivo</h1>
        <p>Eficiência Reprodutiva e Impacto Econômico — Rebanhos de Corte</p>
    </div>

    <div class="tabs">
        <div class="tab active" onclick="showPage('page1', this)">📊 Visão Geral</div>
        <div class="tab" onclick="showPage('page2', this)">🧬 Inteligência Reprodutiva</div>
        <div class="tab" onclick="showPage('page3', this)">💰 Inteligência Econômica</div>
    </div>

    <!-- PAGE 1: Visão Geral -->
    <div id="page1" class="page active">
        <div class="kpi-grid">
            {create_kpi_card("Taxa de Prenhez", f"{taxa_prenhez:.1%}", "Média geral do rebanho", COLORS['primary'])}
            {create_kpi_card("Margem Bruta", f"R$ {margem:,.0f}", "Por vaca exposta", COLORS['secondary'])}
            {create_kpi_card("ROI Reprodutivo", f"{roi:.0f}%", "Retorno sobre investimento", COLORS['warning'])}
            {create_kpi_card("Dias Abertos", f"{dias_abertos:.0f}", "Média do rebanho", COLORS['danger'])}
        </div>
        <div class="charts-grid">
            <div class="chart-card"><div id="chart1"></div></div>
            <div class="chart-card"><div id="chart6"></div></div>
        </div>
        <div class="insight-box">
            <h3>💡 Insight Principal</h3>
            <p>Com taxa de prenhez de {taxa_prenhez:.1%}, o rebanho apresenta oportunidade de melhoria.
            Cada ponto percentual adicional gera R$ {gain_1pp:,.2f}/vaca — para 1.000 vacas,
            isso representa R$ {gain_1pp*1000:,.0f}/ano. A fazenda com melhor performance supera em
            {(df.groupby('fazenda')['resultado_prenhez'].mean().max() - df.groupby('fazenda')['resultado_prenhez'].mean().min())*100:.1f}pp
            a de pior resultado.</p>
        </div>
    </div>

    <!-- PAGE 2: Inteligência Reprodutiva -->
    <div id="page2" class="page">
        <div class="kpi-grid">
            {create_kpi_card("ECC Médio", f"{df['ecc'].mean():.1f}", "Escore Condição Corporal", COLORS['primary'])}
            {create_kpi_card("Perda Gestacional", f"{df[df['resultado_prenhez']==1]['perda_gestacional'].mean():.1%}", "Das prenhas confirmadas", COLORS['danger'])}
            {create_kpi_card("Melhor Touro", f"{df.groupby('touro')['resultado_prenhez'].mean().idxmax()}", f"{df.groupby('touro')['resultado_prenhez'].mean().max():.0%} de prenhez", COLORS['primary'])}
            {create_kpi_card("N. Animais", f"{len(df):,}", "Total no rebanho", COLORS['secondary'])}
        </div>
        <div class="charts-grid">
            <div class="chart-card"><div id="chart2"></div></div>
            <div class="chart-card"><div id="chart3"></div></div>
        </div>
        <div class="insight-box">
            <h3>🧬 Insight Reprodutivo</h3>
            <p>O ECC é o fator de maior impacto na taxa de prenhez. Animais com ECC ≥ 6.0 apresentam
            taxas significativamente superiores. A performance dos touros varia até
            {(df.groupby('touro')['resultado_prenhez'].mean().max() - df.groupby('touro')['resultado_prenhez'].mean().min())*100:.0f}pp,
            indicando necessidade de avaliação genética rigorosa.</p>
        </div>
    </div>

    <!-- PAGE 3: Inteligência Econômica -->
    <div id="page3" class="page">
        <div class="kpi-grid">
            {create_kpi_card("Receita Total", f"R$ {df['receita_esperada'].sum():,.0f}", "Rebanho completo", COLORS['primary'])}
            {create_kpi_card("Custo Total", f"R$ {df['custo_total'].sum():,.0f}", "Rebanho completo", COLORS['danger'])}
            {create_kpi_card("Ganho +1pp", f"R$ {gain_1pp:,.0f}", "Por vaca/ano", COLORS['primary'])}
            {create_kpi_card("Ganho 1000 Vacas", f"R$ {gain_1pp*1000:,.0f}", "+1pp prenhez/ano", COLORS['warning'])}
        </div>
        <div class="charts-grid">
            <div class="chart-card"><div id="chart4"></div></div>
            <div class="chart-card"><div id="chart5"></div></div>
        </div>
        <div class="simulator">
            <h3>🎯 Simulador Estratégico</h3>
            <p style="color:{COLORS['muted']}; margin-bottom: 16px;">
                "Qual o impacto financeiro ao aumentar a taxa de prenhez?"
            </p>
            <div style="display:grid; grid-template-columns: repeat(3,1fr); gap:16px; text-align:center;">
                <div>
                    <div style="font-size:12px; color:{COLORS['muted']};">+1pp → 1.000 cabeças</div>
                    <div class="sim-result" style="font-size:24px;">R$ {gain_1pp*1000:,.0f}</div>
                </div>
                <div>
                    <div style="font-size:12px; color:{COLORS['muted']};">+5pp → 1.000 cabeças</div>
                    <div class="sim-result" style="font-size:24px;">R$ {(calc_margin(base_rate+0.05)-calc_margin(base_rate))*1000:,.0f}</div>
                </div>
                <div>
                    <div style="font-size:12px; color:{COLORS['muted']};">+10pp → 1.000 cabeças</div>
                    <div class="sim-result" style="font-size:24px;">R$ {(calc_margin(base_rate+0.10)-calc_margin(base_rate))*1000:,.0f}</div>
                </div>
            </div>
            <div class="sim-detail">Projeção baseada nos custos e receitas médias do rebanho atual</div>
        </div>
    </div>
</div>

<script>
    function showPage(id, el) {{
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.getElementById(id).classList.add('active');
        el.classList.add('active');
    }}

    Plotly.newPlot('chart1', {fig1.to_json()}.data, {fig1.to_json()}.layout, {{responsive: true}});
    Plotly.newPlot('chart2', {fig2.to_json()}.data, {fig2.to_json()}.layout, {{responsive: true}});
    Plotly.newPlot('chart3', {fig3.to_json()}.data, {fig3.to_json()}.layout, {{responsive: true}});
    Plotly.newPlot('chart4', {fig4.to_json()}.data, {fig4.to_json()}.layout, {{responsive: true}});
    Plotly.newPlot('chart5', {fig5.to_json()}.data, {fig5.to_json()}.layout, {{responsive: true}});
    Plotly.newPlot('chart6', {fig6.to_json()}.data, {fig6.to_json()}.layout, {{responsive: true}});
</script>
</body>
</html>"""

    output = DASHBOARD_PATH / "index.html"
    with open(output, 'w') as f:
        f.write(html)
    print(f"   ✅ Dashboard salvo em: {output}")
    return output


def main():
    print("\n📊 GERANDO DASHBOARD EXECUTIVO")
    print("=" * 50)
    df, kpis, farms = load_data()
    output = build_dashboard(df, kpis, farms)
    print(f"\n   Abrir: file://{output.resolve()}")


if __name__ == "__main__":
    main()
