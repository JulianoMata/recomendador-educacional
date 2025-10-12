# scripts/data_loader.py
import os
import pickle
import time
from typing import Any, Dict, Set, Tuple

import gdown
import pandas as pd
import streamlit as st

# --- CONFIGURAÇÕES GLOBAIS ---
DATA_PATH = "dados/ml-32m/ratings.csv"
# Para o gdown, usamos o link de compartilhamento normal do Google Drive
MODEL_URL = "https://drive.google.com/file/d/1gN4VJVysJwFH2moIPwc5TwC8m0cySncN/"
MODEL_PATH = "svd_model_data.pkl"  # Caminho relativo na raiz do projeto

# ==========================================================
#   FUNÇÃO: DOWNLOAD DO MODELO
# ==========================================================
# @st.cache_data garante que o download só aconteça uma vez por sessão.
@st.cache_data(show_spinner=False)
def download_model_if_missing(url: str, output_path: str) -> None:
    """
    Verifica se o arquivo do modelo existe. Se não, faz o download do Google Drive
    usando a biblioteca gdown, que é robusta para arquivos grandes e páginas de confirmação.
    """
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        st.info(f"🟢 Modelo encontrado localmente ({size_mb:.1f} MB).")
        return

    st.warning("📦 Modelo não encontrado. Iniciando download (~1.9 GB)...")
    with st.spinner("Baixando... esta operação pode levar vários minutos."):
        try:
            # gdown lida com os links de compartilhamento do Google Drive automaticamente
            gdown.download(url, output_path, quiet=False)
            
            # Verificação final para garantir que o arquivo foi criado
            if not os.path.exists(output_path):
                raise FileNotFoundError("gdown finalizou o download, mas o arquivo não foi encontrado no disco.")

        except Exception as e:
            st.error(f"🔴 Erro fatal ao baixar o modelo: {e}")
            # Em caso de erro, remove o arquivo parcial para evitar corrupção
            if os.path.exists(output_path):
                os.remove(output_path)
            st.stop() # Interrompe a execução do app se o download falhar

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    st.success(f"🟢 Download concluído com sucesso! ({size_mb:.1f} MB)")


# ==========================================================
#   FUNÇÃO: CARREGAMENTO PRINCIPAL
# ==========================================================
# @st.cache_resource executa esta função inteira apenas UMA VEZ por sessão do app.
@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Tuple, Tuple, int]:
    """
    Orquestra todo o processo de carga: download do modelo, leitura dos dados
    e preparação das estruturas de dados otimizadas para a recomendação.
    """

    # --- 1. Garante que o modelo está disponível localmente ---
    download_model_if_missing(MODEL_URL, MODEL_PATH)

    # --- 2. Carrega o modelo do arquivo .pkl para a memória ---
    with st.spinner("🧠 Carregando modelo de recomendação..."):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
        except Exception as e:
            st.error(f"🔴 Falha ao ler o arquivo do modelo: {e}")
            st.stop()

    model_svd = model_data.get("model")
    id_to_titulo = model_data.get("id_to_titulo", {})
    id_to_generos = model_data.get("id_to_generos", {})

    # --- 3. Leitura e processamento da base de avaliações ---
    with st.spinner(f"📊 Lendo e processando a base de avaliações..."):
        ratings_df = pd.read_csv(DATA_PATH)
        ratings_df.columns = ["user_id", "item_id", "rating", "timestamp"]
        ratings_df["user_id"] = ratings_df["user_id"].astype(str)
        ratings_df["item_id"] = ratings_df["item_id"].astype(str)
        max_user_id = int(ratings_df["user_id"].astype(int).max())

    # --- 4. Construção de estruturas otimizadas para performance ---
    with st.spinner("🔧 Preparando o motor de recomendação..."):
        user_rated_items: Dict[str, Set[str]] = {}
        for row in ratings_df[['user_id', 'item_id']].itertuples(index=False):
            user_id = str(row.user_id)
            item_id = str(row.item_id)
            user_rated_items.setdefault(user_id, set()).add(item_id)
        
        all_item_ids = set(ratings_df['item_id'].unique())

    st.success("🚀🙌 Modelo e dados prontos para uso!")
    
    # Agrupa os objetos de retorno
    opt_data = (user_rated_items, all_item_ids)
    maps = (id_to_titulo, id_to_generos)
    
    return model_svd, opt_data, maps, max_user_id
# ==========================================================
