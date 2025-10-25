-- ================================================================
-- SCHEMA SQL — REPRODUÇÃO BOVINA
-- Database para sistema de gestão reprodutiva
-- ================================================================

CREATE TABLE IF NOT EXISTS fazendas (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    estado VARCHAR(2) NOT NULL,
    cidade VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS animais (
    id VARCHAR(10) PRIMARY KEY,
    fazenda_id INTEGER REFERENCES fazendas(id),
    raca VARCHAR(30) NOT NULL,
    categoria VARCHAR(20) NOT NULL CHECK (categoria IN ('Novilha', 'Primípara', 'Multípara')),
    idade_anos DECIMAL(4,1),
    peso_kg INTEGER,
    ecc DECIMAL(3,1) CHECK (ecc BETWEEN 1 AND 9),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS touros (
    id VARCHAR(20) PRIMARY KEY,
    nome VARCHAR(100),
    raca VARCHAR(30),
    dep_prenhez DECIMAL(5,2),
    status VARCHAR(20) DEFAULT 'ativo'
);

CREATE TABLE IF NOT EXISTS protocolos_reprodutivos (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(50) NOT NULL,
    custo_medio DECIMAL(10,2),
    descricao TEXT
);

CREATE TABLE IF NOT EXISTS estacoes_monta (
    id SERIAL PRIMARY KEY,
    animal_id VARCHAR(10) REFERENCES animais(id),
    touro_id VARCHAR(20) REFERENCES touros(id),
    protocolo_id INTEGER REFERENCES protocolos_reprodutivos(id),
    fazenda_id INTEGER REFERENCES fazendas(id),
    data_inicio DATE,
    data_diagnostico DATE,
    dpp INTEGER,
    thi DECIMAL(4,1),
    resultado_prenhez BOOLEAN,
    perda_gestacional BOOLEAN DEFAULT FALSE,
    custo_protocolo DECIMAL(10,2),
    custo_manutencao_dia DECIMAL(8,2),
    custo_nutricional DECIMAL(10,2),
    valor_bezerro DECIMAL(10,2),
    dias_abertos INTEGER,
    observacoes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Views para análise
CREATE VIEW vw_resumo_fazenda AS
SELECT
    f.nome AS fazenda,
    COUNT(em.id) AS total_animais,
    AVG(CASE WHEN em.resultado_prenhez THEN 1.0 ELSE 0.0 END) AS taxa_prenhez,
    AVG(a.ecc) AS ecc_medio,
    AVG(em.dias_abertos) AS dias_abertos_medio,
    SUM(em.custo_protocolo + em.custo_nutricional) AS custo_total
FROM estacoes_monta em
JOIN animais a ON em.animal_id = a.id
JOIN fazendas f ON em.fazenda_id = f.id
GROUP BY f.nome;

CREATE VIEW vw_performance_touro AS
SELECT
    t.id AS touro,
    COUNT(em.id) AS total_coberturas,
    AVG(CASE WHEN em.resultado_prenhez THEN 1.0 ELSE 0.0 END) AS taxa_prenhez,
    AVG(CASE WHEN em.perda_gestacional THEN 1.0 ELSE 0.0 END) AS taxa_perda
FROM estacoes_monta em
JOIN touros t ON em.touro_id = t.id
GROUP BY t.id
ORDER BY taxa_prenhez DESC;

-- Índices
CREATE INDEX idx_estacoes_animal ON estacoes_monta(animal_id);
CREATE INDEX idx_estacoes_fazenda ON estacoes_monta(fazenda_id);
CREATE INDEX idx_estacoes_touro ON estacoes_monta(touro_id);
CREATE INDEX idx_animais_fazenda ON animais(fazenda_id);
