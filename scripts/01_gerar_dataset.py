"""
================================================================
GERADOR DE DATASET SIMULADO — REPRODUÇÃO BOVINA
================================================================
Gera 5.000+ animais com coerência biológica e zootécnica.
Variáveis produtivas, reprodutivas e econômicas.
================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# === CONFIG ===
N_ANIMALS = 5200
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "rebanho_simulado.csv"

# Fazendas
FAZENDAS = [
    "Fazenda Santa Helena", "Fazenda Boa Vista", "Fazenda São José",
    "Fazenda Esperança", "Fazenda Nova Aliança"
]
FAZENDA_WEIGHTS = [0.25, 0.20, 0.22, 0.18, 0.15]

# Raças
RACAS = ["Nelore", "Angus", "Brangus", "Hereford", "Senepol"]
RACA_WEIGHTS = [0.40, 0.20, 0.18, 0.12, 0.10]

# Categorias
CATEGORIAS = ["Novilha", "Primípara", "Multípara"]
CAT_WEIGHTS = [0.25, 0.30, 0.45]

# Protocolos reprodutivos
PROTOCOLOS = ["IATF", "Monta Natural", "IATF + Repasse", "TE"]
PROTO_WEIGHTS = [0.40, 0.20, 0.30, 0.10]
PROTO_CUSTOS = {"IATF": 45, "Monta Natural": 15, "IATF + Repasse": 60, "TE": 180}

# Touros
TOUROS = [f"Touro_{chr(65+i)}" for i in range(12)]  # Touro_A ... Touro_L

# === GERAR DADOS ===
def generate_dataset():
    data = {}

    # Identificadores
    data["id_animal"] = [f"BOV-{str(i+1).zfill(5)}" for i in range(N_ANIMALS)]
    data["fazenda"] = np.random.choice(FAZENDAS, N_ANIMALS, p=FAZENDA_WEIGHTS)
    data["raca"] = np.random.choice(RACAS, N_ANIMALS, p=RACA_WEIGHTS)
    data["categoria"] = np.random.choice(CATEGORIAS, N_ANIMALS, p=CAT_WEIGHTS)

    # Idade (coerente com categoria)
    idade = []
    for cat in data["categoria"]:
        if cat == "Novilha":
            idade.append(np.random.uniform(1.5, 3.0))
        elif cat == "Primípara":
            idade.append(np.random.uniform(2.5, 4.5))
        else:
            idade.append(np.random.uniform(4.0, 12.0))
    data["idade_anos"] = np.round(idade, 1)

    # ECC (Escore de Condição Corporal 1-9, coerente com categoria)
    ecc = []
    for cat in data["categoria"]:
        if cat == "Novilha":
            ecc.append(np.clip(np.random.normal(6.0, 0.8), 3, 9))
        elif cat == "Primípara":
            ecc.append(np.clip(np.random.normal(5.0, 1.0), 2, 9))
        else:
            ecc.append(np.clip(np.random.normal(5.5, 1.2), 2, 9))
    data["ecc"] = np.round(ecc, 1)

    # Dias Pós-Parto (DPP) — novilhas não têm, outras sim
    dpp = []
    for cat in data["categoria"]:
        if cat == "Novilha":
            dpp.append(0)
        elif cat == "Primípara":
            dpp.append(max(30, int(np.random.normal(70, 20))))
        else:
            dpp.append(max(30, int(np.random.normal(60, 25))))
    data["dpp"] = dpp

    # Peso (kg, coerente com raça e categoria)
    peso = []
    base_peso = {"Nelore": 420, "Angus": 480, "Brangus": 460, "Hereford": 470, "Senepol": 440}
    for i in range(N_ANIMALS):
        raca = data["raca"][i]
        cat = data["categoria"][i]
        base = base_peso[raca]
        if cat == "Novilha":
            base -= 80
        elif cat == "Primípara":
            base -= 30
        peso.append(np.clip(np.random.normal(base, 40), 250, 650))
    data["peso_kg"] = np.round(peso, 0).astype(int)

    # Touro e Protocolo
    data["touro"] = np.random.choice(TOUROS, N_ANIMALS)
    data["protocolo"] = np.random.choice(PROTOCOLOS, N_ANIMALS, p=PROTO_WEIGHTS)

    # THI (Temperature-Humidity Index) — 60-85
    data["thi"] = np.round(np.random.normal(72, 6, N_ANIMALS).clip(55, 88), 1)

    # === PRENHEZ (modelo biologicamente coerente) ===
    prob_prenhez = np.zeros(N_ANIMALS)
    for i in range(N_ANIMALS):
        # Base probability
        p = 0.55

        # ECC: principal fator — forte efeito positivo acima de 5
        ecc_val = data["ecc"][i]
        if ecc_val >= 6.0:
            p += 0.15
        elif ecc_val >= 5.0:
            p += 0.05
        elif ecc_val < 4.0:
            p -= 0.20
        elif ecc_val < 3.5:
            p -= 0.30

        # DPP: voluntário waiting period, otimo entre 50-80
        dpp_val = data["dpp"][i]
        if data["categoria"][i] != "Novilha":
            if 50 <= dpp_val <= 80:
                p += 0.10
            elif dpp_val > 120:
                p -= 0.08
            elif dpp_val < 45:
                p -= 0.15

        # Categoria
        cat = data["categoria"][i]
        if cat == "Novilha":
            p -= 0.05  # menos experiência reprodutiva
        elif cat == "Multípara":
            p += 0.08  # mais experiência

        # THI: estresse calórico
        thi_val = data["thi"][i]
        if thi_val > 78:
            p -= 0.10
        elif thi_val > 82:
            p -= 0.18
        elif thi_val < 68:
            p += 0.05

        # Protocolo
        proto = data["protocolo"][i]
        if proto == "TE":
            p += 0.12
        elif proto == "IATF + Repasse":
            p += 0.08
        elif proto == "Monta Natural":
            p -= 0.05

        # Raça
        raca = data["raca"][i]
        if raca in ["Angus", "Brangus"]:
            p += 0.03
        elif raca == "Nelore":
            p -= 0.02  # menor fertilidade natural

        # Efeito touro (variação genética)
        touro_effect = {
            "Touro_A": 0.08, "Touro_B": 0.05, "Touro_C": -0.03,
            "Touro_D": 0.10, "Touro_E": -0.08, "Touro_F": 0.02,
            "Touro_G": -0.05, "Touro_H": 0.06, "Touro_I": -0.02,
            "Touro_J": 0.04, "Touro_K": -0.06, "Touro_L": 0.00
        }
        p += touro_effect.get(data["touro"][i], 0)

        # Noise biológico
        p += np.random.normal(0, 0.05)

        prob_prenhez[i] = np.clip(p, 0.05, 0.95)

    # Resultado diagnóstico (binário)
    data["resultado_prenhez"] = (np.random.random(N_ANIMALS) < prob_prenhez).astype(int)
    data["prob_prenhez_real"] = np.round(prob_prenhez, 3)

    # Perda gestacional (apenas para prenhas, ~5-8%)
    perda = np.zeros(N_ANIMALS, dtype=int)
    for i in range(N_ANIMALS):
        if data["resultado_prenhez"][i] == 1:
            taxa_perda = 0.06
            if data["ecc"][i] < 4.0:
                taxa_perda += 0.04
            if data["thi"][i] > 80:
                taxa_perda += 0.03
            perda[i] = 1 if np.random.random() < taxa_perda else 0
    data["perda_gestacional"] = perda

    # === CUSTOS E ECONOMIA ===
    # Custo protocolo
    data["custo_protocolo"] = [PROTO_CUSTOS[p] for p in data["protocolo"]]

    # Custo manutenção diária (R$ 5-12/dia, varia por fazenda)
    fazenda_custo_base = {
        "Fazenda Santa Helena": 8.5, "Fazenda Boa Vista": 7.0,
        "Fazenda São José": 9.0, "Fazenda Esperança": 7.5,
        "Fazenda Nova Aliança": 8.0
    }
    data["custo_manutencao_dia"] = np.round([
        np.random.normal(fazenda_custo_base[f], 1.0) for f in data["fazenda"]
    ], 2).clip(4.0, 15.0)

    # Custo nutricional (R$/animal, correlacionado com peso)
    data["custo_nutricional"] = np.round(
        np.array(data["peso_kg"]) * np.random.uniform(0.8, 1.5, N_ANIMALS),
        2
    )

    # Valor médio do bezerro (R$ 1.800-3.200, varia por raça)
    valor_bezerro_base = {
        "Nelore": 2200, "Angus": 2800, "Brangus": 2600,
        "Hereford": 2700, "Senepol": 2500
    }
    data["valor_bezerro"] = np.round([
        np.random.normal(valor_bezerro_base[r], 300) for r in data["raca"]
    ], 0).clip(1500, 4000).astype(int)

    # Dias abertos (para não-prenhas: 365; prenhas: DPP ou ~85 para novilhas)
    dias_abertos = []
    for i in range(N_ANIMALS):
        if data["resultado_prenhez"][i] == 1:
            if data["categoria"][i] == "Novilha":
                dias_abertos.append(0)
            else:
                dias_abertos.append(data["dpp"][i])
        else:
            dias_abertos.append(365)  # custo total do ano sem prenhez
    data["dias_abertos"] = dias_abertos

    # Build DataFrame
    df = pd.DataFrame(data)

    # Drop internal probability column for clean dataset
    df_export = df.drop(columns=["prob_prenhez_real"])

    return df_export, df


def main():
    print("🐄 Gerando dataset simulado de reprodução bovina...")
    df_export, df_full = generate_dataset()

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_csv(OUTPUT_PATH, index=False)
    df_full.to_csv(OUTPUT_PATH.parent / "rebanho_completo_debug.csv", index=False)

    print(f"   ✅ {len(df_export)} animais gerados")
    print(f"   📊 Taxa de prenhez: {df_export['resultado_prenhez'].mean():.1%}")
    print(f"   📊 Perda gestacional: {df_export[df_export['resultado_prenhez']==1]['perda_gestacional'].mean():.1%}")
    print(f"   💾 Salvo em: {OUTPUT_PATH}")

    # Quick summary
    print("\n📋 Resumo por fazenda:")
    summary = df_export.groupby("fazenda").agg(
        n_animais=("id_animal", "count"),
        taxa_prenhez=("resultado_prenhez", "mean"),
        ecc_medio=("ecc", "mean"),
        peso_medio=("peso_kg", "mean"),
    ).round(3)
    print(summary.to_string())

    print("\n📋 Resumo por categoria:")
    summary2 = df_export.groupby("categoria").agg(
        n_animais=("id_animal", "count"),
        taxa_prenhez=("resultado_prenhez", "mean"),
        ecc_medio=("ecc", "mean"),
    ).round(3)
    print(summary2.to_string())

    return df_export


if __name__ == "__main__":
    main()
