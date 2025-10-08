# 🎓 Sistema de Recomendação de Conteúdo Educacional

## 🌟 Visão Geral do Projeto

Este projeto, desenvolvido como requisito de Estágio Supervisionado III (Inteligência Artificial) do curso de Ciência de Dados e IA da UniDomBosco, implementa um sistema de recomendação baseado em **Filtros Colaborativos**.

Utilizando o algoritmo **SVD (Singular Value Decomposition)** sobre a base de dados **MovieLens 32M** (com 32 milhões de interações), o objetivo é prever a afinidade de um aluno por um determinado tópico/módulo de estudo e, assim, sugerir o Top-N conteúdo mais relevante através de um protótipo interativo.

## 📊 Fonte dos Dados

O dataset utilizado neste projeto é o **MovieLens 32M**, uma base de dados pública e amplamente utilizada para pesquisa em sistemas de recomendação. Foi coletado e é mantido pelo GroupLens, um laboratório de pesquisa da Universidade de Minnesota.

* **Referência:** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
* **Onde encontrar:** [GroupLens Website](https://grouplens.org/datasets/movielens/)

## ⚙️ Tecnologias Utilizadas

* **Linguagem:** Python 3.12+
* **Análise e Treinamento:** Jupyter Notebook, Pandas, NumPy
* **Modelagem de IA:** Scikit-Surprise (SVD)
* **Serialização do Modelo:** Pickle
* **Protótipo Interativo:** Streamlit
* **Controle de Versão:** Git e GitHub
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
└── 📦 svd_model_data.pkl       # O MODELO TREINADO (artefato final do notebook)
```

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e rodar a aplicação localmente.

### Passo 1: Configuração do Ambiente

1. Clone este repositório para a sua máquina local.
2. A partir da pasta raiz do projeto, crie um ambiente virtual:

    ```bash
    python -m venv venv
    ```

3. Ative o ambiente virtual:
    * **No Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
    * **No Windows (CMD):** `venv\Scripts\activate`
    * **No macOS/Linux:** `source venv/bin/activate`

4. Instale todas as dependências necessárias:

    ```bash
    pip install -r requirements.txt
    ```

### Passo 2: Treinamento e Geração do Modelo

1. Abra e execute todas as células do notebook localizado em `notebook/analise_dados.ipynb`.
2. **Importante:** Este processo irá treinar o modelo SVD e gerar o arquivo `svd_model_data.pkl` na **pasta raiz** do projeto. Este passo é demorado e precisa ser executado apenas uma vez.

### Passo 3: Execução do Protótipo

1. Certifique-se de que o terminal está na **pasta raiz** do projeto e que o ambiente virtual está ativado.
2. Execute o comando para iniciar a aplicação Streamlit:

    ```bash
    streamlit run scripts/app.py
    ```

3. Abra o seu navegador no endereço local fornecido (geralmente `http://localhost:8501`).
4. Interaja com a interface para obter recomendações de conteúdo educacional.
