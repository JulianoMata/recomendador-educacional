# 🎓 Sistema de Recomendação de Conteúdo Educacional

## 🌟 Visão Geral do Projeto

Este projeto, desenvolvido como requisito de Estágio Supervisionado III (Inteligência Artificial) do curso de Ciência de Dados e IA da UniDomBosco, implementa um sistema de recomendação baseado em **Filtros Colaborativos**.

Utilizando o algoritmo **SVD (Singular Value Decomposition)** sobre a base de dados **MovieLens 32M**, o objetivo é prever a afinidade de um aluno por um tópico e sugerir o conteúdo mais relevante através de um protótipo interativo.

## 📊 Fonte dos Dados

O dataset utilizado neste projeto é o **MovieLens 32M**, mantido pelo GroupLens (Universidade de Minnesota).

* **Referência:** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
* **Onde encontrar:** [GroupLens Website](https://grouplens.org/datasets/movielens/)

## ⚙️ Tecnologias Utilizadas

* **Linguagem:** Python 3.12
* **Análise e Treinamento:** Jupyter Notebook, Pandas, NumPy
* **Modelagem de IA:** Scikit-Surprise (SVD)
* **Serialização do Modelo:** Pickle
* **Protótipo Interativo:** Streamlit
* **Versionamento:** Git, GitHub, e Git LFS (para arquivos grandes)
* **Ambiente Virtual:** venv

## 📁 Estrutura do Projeto

A organização dos arquivos e diretórios do projeto é a seguinte:

```text
RECOMENDADOR-EDUCACIONAL/
├── 📂 dados/
│   ├── 📂 ml-32m/               # Contém os arquivos .csv do dataset
│   └── 📦 ml-32m.zip            # Arquivo compactado original do dataset
├── 📂 notebook/
│   └── 📜 analise_dados.ipynb   # Onde o modelo é treinado e exportado
├── 📂 scripts/
│   ├── __init__.py             # Torna a pasta 'scripts' um módulo Python
│   ├── 🐍 app.py                # Ponto de entrada da UI (Streamlit)
│   ├── 🐍 data_loader.py        # Carrega dados e o modelo .pkl
│   └── 🐍 recommender.py        # Contém a lógica de recomendação
├── 📂 venv/                    # Pasta do ambiente virtual (isolamento)
├── .gitattributes              # Configuração do Git LFS para arquivos grandes
├── .gitignore                  # Arquivos a serem ignorados pelo Git
├── LICENSE                     # Licença do projeto
├── README.md                   # Documentação principal
├── requirements.txt            # Lista de dependências Python
├── runtime.txt                 # Define a versão do Python para o deploy no Streamlit
└── 📦 svd_model_data.pkl       # O MODELO TREINADO (artefato final do notebook)
```

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e rodar a aplicação localmente.

### Passo 1: Pré-requisitos

Antes de começar, garanta que você tenha os seguintes softwares instalados:

* Python 3.12
* Git
* Git LFS (instale do site [git-lfs.github.com](https://git-lfs.github.com) e depois rode `git lfs install` no terminal uma vez).

### Passo 2: Configuração do Projeto

1. **Clone o Repositório:**

    ```bash
    git clone https://github.com/JulianoMata/recomendador-educacional.git
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

### Passo 3: Download dos Dados

1. Acesse o site do [GroupLens](https://grouplens.org/datasets/movielens/) e baixe o dataset que você utilizou (ex: "ml-latest" ou "ml-25m").
2. Descompacte o arquivo `.zip` que foi baixado.
3. Copie os arquivos `ratings.csv` e `movies.csv` para a pasta `dados/ml-32m/` dentro do projeto.

### Passo 4: Execução do Protótipo

Com o modelo pré-treinado (`.pkl`) já incluído no repositório via Git LFS, você pode iniciar o aplicativo diretamente.

1. Certifique-se de que o terminal está na pasta raiz do projeto e que o ambiente virtual está ativado.
2. Execute o comando:

    ```bash
    streamlit run scripts/app.py
    ```

### Passo 5: Retreinamento do Modelo (**Opcional**)

Se você desejar recriar o arquivo do modelo (`svd_model_data.pkl`) do zero, execute todas as células do notebook localizado em `notebook/analise_dados.ipynb`. Este processo é demorado.
