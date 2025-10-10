# 🎓 Sistema de Recomendação de Conteúdo Educacional

## 🌟 Visão Geral do Projeto

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recomendador-educacional.streamlit.app/)

Este projeto, desenvolvido como requisito de Estágio Supervisionado III, implementa um sistema de recomendação baseado em **Filtros Colaborativos** utilizando o algoritmo **SVD (Singular Value Decomposition)**.

O modelo foi treinado em uma grande base de dados da série **MovieLens** (contendo milhões de interações) e o objetivo é prever a afinidade de um aluno por um tópico, sugerindo o conteúdo mais relevante através de um protótipo interativo construído com Streamlit.

## 📊 Fonte dos Dados

O dataset utilizado neste projeto é derivado da série MovieLens, mantida pelo GroupLens (Universidade de Minnesota).

* **Onde encontrar:** [GroupLens Website](https://grouplens.org/datasets/movielens/)
* **Referência:** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.

## ⚙️ Tecnologias Utilizadas

* **Linguagem:** Python 3.12
* **Análise e Treinamento:** Jupyter Notebook, Pandas, NumPy
* **Modelagem de IA:** Scikit-Surprise (SVD)
* **Protótipo Interativo:** Streamlit
* **Versionamento:** Git e GitHub

## 📁 Estrutura do Projeto

A organização dos arquivos e diretórios do projeto é a seguinte:

```text
RECOMENDADOR-EDUCACIONAL/
├── 📂 dados/
│   └── 📂 ml-32m/               # Local para os arquivos .csv do dataset
├── 📂 notebook/
│   └── 📜 analise_dados.ipynb   # Onde o modelo é treinado e exportado
├── 📂 scripts/
│   ├── __init__.py             # Torna a pasta 'scripts' um módulo Python
│   ├── 🐍 app.py                # Ponto de entrada da UI (Streamlit)
│   ├── 🐍 data_loader.py        # Carrega dados e baixa o modelo .pkl
│   └── 🐍 recommender.py        # Contém a lógica de recomendação
├── .gitattributes              # Configuração do Git para normalização de linhas
├── .gitignore                  # Ignora arquivos grandes (dados, modelo .pkl)
├── LICENSE                     # Licença do projeto
├── README.md                   # Documentação principal
├── requirements.txt            # Lista de dependências Python
└── runtime.txt                 # Define a versão do Python para o deploy
```

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e rodar a aplicação localmente.

### Passo 1: Pré-requisitos

Antes de começar, garanta que você tenha **Python 3.12** e **Git** instalados.

### Passo 2: Configuração do Projeto

1. **Clone o Repositório:**

   ```bash
    git clone [https://github.com/JulianoMata/recomendador-educacional.git](https://github.com/JulianoMata/recomendador-educacional.git)
    cd recomendador-educacional
    ```

2. **Crie e Ative o Ambiente Virtual:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3. **Instale as Dependências:**

    ```bash
    pip install -r requirements.txt
    ```

### Passo 3: Download dos Dados Manuais

1. Acesse o site do [GroupLens](https://grouplens.org/datasets/movielens/) e baixe o dataset que você utilizou (ex: "ml-25m").
2. Descompacte o arquivo `.zip` baixado.
3. Copie os arquivos `ratings.csv` e `movies.csv` para a pasta `dados/ml-32m/` dentro do projeto.

### Passo 4: Execução do Protótipo

1. Certifique-se de que o terminal está na pasta raiz do projeto e que o ambiente virtual está ativado.
2. Execute o comando:

    ```bash
    streamlit run scripts/app.py
    ```

3. **Importante:** Na primeira vez que você rodar, o aplicativo irá **baixar automaticamente** o modelo pré-treinado (`.pkl`) de 1.9 GB. Uma barra de progresso será exibida no navegador. Este processo é demorado, mas só acontecerá uma vez.

### Passo 5: Retreinamento do Modelo (Opcional)

Se você desejar recriar o arquivo do modelo (`svd_model_data.pkl`), execute todas as células do notebook localizado em `notebook/analise_dados.ipynb`.
**Nota:** Este processo requer uma máquina com pelo menos 16 GB de RAM e pode levar várias horas para ser concluído.
