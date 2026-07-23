# Triagem de Picadas e Mordidas Peçonhentas: Classificação da Origem com Aprendizado de Máquina

## Objetivo

Desenvolver um modelo de aprendizado de máquina capaz de classificar a origem de uma picada/mordida potencialmente peçonhenta (inseto inofensivo, víbora, escorpião ou aranha viúva-negra), com base em dados clínicos do paciente (sinais vitais e sintomas). Projeto avaliativo da disciplina de Fundamentos de Inteligência Artificial.

## Integrantes

* Edno Bezerra Nascimento Junior
* Nicolle Rillary Santana Silva
* Wagner Kauê Martins dos Santos

## Fonte dos dados

[Emergency Triage: Venomous Bites Dataset](https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset) (Kaggle), aproximadamente 1 milhão de registros simulados de atendimentos médicos de pacientes picados/mordidos por animais possivelmente peçonhentos.

## Tipo da tarefa

Classificação: o atributo-alvo (`Bite_Source_Target`) é categórico, com 4 classes: `Harmless_Insect`, `Viper_Snake`, `Scorpion`, `Black_Widow_Spider`.

## Organização dos arquivos

* `ProjetoFundamentosDeIA.ipynb`: notebook principal, com todo o desenvolvimento do projeto (compreensão dos dados, análise exploratória, pré-processamento, modelagem e avaliação).
* `dataset/silent_sting_triage_data.parquet`: cópia local do dataset. Caso não esteja presente, o notebook baixa automaticamente o dataset original do Kaggle (via `kagglehub`).

## Como abrir no Colab

Acesse o link:
[colab.research.google.com/github/imnicolle/triage-venomous-bites/blob/main/ProjetoFundamentosDeIA.ipynb](https://colab.research.google.com/github/imnicolle/triage-venomous-bites/blob/main/ProjetoFundamentosDeIA.ipynb)

A primeira célula de código instala automaticamente todas as dependências necessárias (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `kagglehub`, `pyarrow`).

As células que treinam os modelos podem levar alguns minutos para terminar a execução.

## Modelos utilizados

* **Baseline**: `DummyClassifier` (estratégia `most_frequent`): ~40% de acurácia.
* **SGDClassifier** (`loss='hinge'`, `penalty='l2'`).
* **RandomForestClassifier** (`n_estimators=200`): modelo final escolhido.

## Principais resultados

Comparação por validação cruzada (5 folds estratificados):

- SGDClassifier 99,69% de acurácia / F1-macro 99,64%
- RandomForestClassifier 99,72% de acurácia / F1-macro 99,68% (modelo final escolhido).

Avaliação no conjunto de teste (20%, nunca visto pelo modelo): 99,71% de acurácia, F1-macro 99,67%. `Black_Widow_Spider` e `Viper_Snake` com precisão e revocação de 100%. Todo o erro do modelo concentrado na distinção entre `Harmless_Insect` e `Scorpion`.

## Divisão das contribuições

Nicolle Rillary Santana Silva:
- Identificação do problema;
- Compreensão dos dados.

Edno Bezerra Nascimento Junior:
- Análise exploratória dos dados;
- Pré-processamento.

Wagner Kauê Martins dos Santos:
- Modelagem dos agentes;
- Avaliação e discussão.


## Vídeo

[a adicionar]

## Uso de ferramentas de Inteligência Artificial

Uso 1:

- Ferramenta utilizada: **Claude**.
- Finalidade:
    - Consulta de informações sobre o funcionamento de funções e parâmetros das bibliotecas como sklearn e matplotlib.
    - Revisão de conformidade com os critérios do professor para as seções 6 e 7.
- Verificação: O funcionamento foi conferido a partir da execução real do notebook e de checagens adicionais feitas sobre o dataset.

Uso 2:

- Ferramenta utilizada: **Claude**.
- Finalidade:
    - Auxilio na criação de tabelas e graficos pertinemtes ao tema na análise exploratoria.
    - Planejamento das etapas de pré-processamento dos dados.
    - Revisão de conformidade com os critérios do professor para as seções 3 e 5.
- Verificação: O funcionamento foi conferido a partir da execução real do notebook e de checagens adicionais feitas sobre o dataset.
