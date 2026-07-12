# Classificação do Animal Causador da Picada — Triagem de Emergência

Projeto da 3ª unidade da disciplina de IA. O objetivo é usar Machine Learning
para **prever qual animal causou a picada/mordida** de um paciente com base
apenas nos sintomas clínicos observáveis, simulando um cenário de triagem
onde o paciente não consegue informar verbalmente o que aconteceu.

## 🎯 Objetivo

Dado um conjunto de sintomas e sinais vitais registrados na triagem
(ex: inchaço, necrose, sangramento, alterações neurológicas, tempo desde a
picada, sinais vitais etc.), treinar um modelo de classificação multiclasse
que prevê a espécie/categoria do animal responsável (ex: cobra, aranha,
escorpião, abelha/vespa etc. — confirmar categorias exatas após inspecionar
o dataset).

## 📊 Dataset

- Fonte: [Emergency Triage: Venomous Bites Dataset](https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset)
- ~1.000.000 de registros sintéticos de pacientes não-verbais picados/mordidos
- Dados tabulares, tarefa de classificação

> ⚠️ O dataset não é versionado no Git (arquivo grande, ~1M linhas).
> Baixe o CSV do Kaggle e coloque em `data/raw/`. Veja `data/raw/README.md`.

## 🗂️ Estrutura do repositório

```
triage-project/
├── data/
│   ├── raw/            # dataset original baixado do Kaggle (não versionado)
│   └── processed/      # dados limpos/tratados, prontos pro modelo
├── notebooks/          # notebooks de exploração e experimentação
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/                # código reutilizável (funções, pipeline)
│   ├── data_processing.py
│   ├── train.py
│   └── evaluate.py
├── models/             # modelos treinados salvos (.pkl)
├── reports/
│   └── figures/        # gráficos gerados para o relatório final
├── requirements.txt
└── README.md
```

## 🚀 Como rodar

```bash
# 1. Clonar o repositório
git clone <url-do-seu-repo>
cd triage-project

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar o dataset do Kaggle e colocar em data/raw/

# 5. Rodar os notebooks em ordem (01 -> 02 -> 03)
jupyter notebook
```

## 🧠 Plano de trabalho (sugestão)

1. **EDA (Análise Exploratória)** — entender as colunas, distribuição das
   classes de animais, valores faltantes, correlações entre sintomas.
2. **Pré-processamento** — tratar dados faltantes, codificar variáveis
   categóricas, normalizar/padronizar, lidar com desbalanceamento de classes
   (comum quando algumas espécies têm poucos casos).
3. **Modelagem** — treinar e comparar pelo menos 2-3 modelos:
   - Baseline: Regressão Logística / Árvore de Decisão
   - Random Forest
   - XGBoost / LightGBM
4. **Avaliação** — acurácia, F1-score por classe (macro), matriz de
   confusão. Em contexto médico, prestar atenção especial a classes que
   são frequentemente confundidas (ex: picadas com sintomas parecidos mas
   tratamento diferente).
5. **Interpretabilidade (opcional, ganha pontos)** — feature importance ou
   SHAP para explicar quais sintomas mais pesam na decisão do modelo.
6. **Relatório final** — reunir resultados, gráficos e conclusões.

## 👥 Equipe

- Nome 1
- Nome 2
- Nome 3

## 📄 Licença

Projeto acadêmico — uso educacional.
