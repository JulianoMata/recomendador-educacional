# scripts/data_loader.py

import streamlit as st
import pandas as pd
import pickle
import gdown  # Usaremos gdown para um download mais robusto do Google Drive
import os
from typing import Dict, Any, Tuple, Set

# --- CONFIGURAÇÕES GLOBAIS ---
DATA_PATH = 'dados/ml-32m/ratings.csv'
# Para o gdown, usamos o link de compartilhamento normal (não o convertido)
MODEL_URL = 'https://drive.google.com/file/d/1RJJcByH0cx83orS334n-5uSEHN8LGmeS/view?usp=sharing'
MODEL_PATH = 'svd_model_data.pkl'

# ==========================================================
#   FUNÇÃO: DOWNLOAD DO MODELO
# ==========================================================
# @st.cache_data garante que o download só aconteça uma vez.
@st.cache_data
def download_model(url: str, output_path: str) -> None:
    """
    Verifica se o arquivo do modelo existe. Se não, faz o download do Google Drive
    usando a biblioteca gdown, que é mais robusta para arquivos grandes.
    """
    if not os.path.exists(output_path):
        st.warning(f"📦 Modelo não encontrado localmente. Iniciando download (~1.9 GB)...")
        with st.spinner("Baixando... esta operação pode levar vários minutos."):
            try:
                gdown.download(url, output_path, quiet=False)
                st.success("✅ Download concluído!")
            except Exception as e:
                st.error(f"❌ Erro fatal ao baixar o modelo: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path) # Remove arquivo parcial
                st.stop()
    else:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        st.info(f"✅ Modelo encontrado localmente ({file_size_mb:.1f} MB).")

# ==========================================================
#   FUNÇÃO: CARREGAMENTO PRINCIPAL
# ==========================================================
# @st.cache_resource executa esta função inteira apenas UMA VEZ por sessão.
@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Tuple[Dict[str, Set[str]], Set[str]], Tuple[Dict[str, str], Dict[str, str]], int]:
    """
    Orquestra todo o processo de carga: download do modelo, leitura dos dados
    e preparação das estruturas de dados otimizadas para a recomendação.
    """

    # --- 1. Garante que o modelo está disponível localmente ---
    download_model(MODEL_URL, MODEL_PATH)

    # --- 2. Carrega o modelo do arquivo .pkl para a memória ---
    with st.spinner("🧠 Carregando modelo de recomendação..."):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)

    model_svd = model_data['model']
    id_to_titulo = model_data['id_to_titulo']
    id_to_generos = model_data['id_to_generos']

    # --- 3. Leitura e processamento da base de avaliações ---
    with st.spinner(f"📊 Lendo e processando a base de avaliações..."):
        ratings_df = pd.read_csv(DATA_PATH)
        ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df['user_id'] = ratings_df['user_id'].astype(str)
        ratings_df['item_id'] = ratings_df['item_id'].astype(str)
        max_user_id = int(ratings_df['user_id'].astype(int).max())

    # --- 4. Construção de estruturas otimizadas para performance ---
    with st.spinner("🔧 Preparando o motor de recomendação..."):
        user_rated_items = {}
        for row in ratings_df[['user_id', 'item_id']].itertuples(index=False):
            user_rated_items.setdefault(row.user_id, set()).add(row.item_id)
        
        all_item_ids = set(ratings_df['item_id'].unique())

    st.success("🚀 Modelo e dados prontos para uso!")
    
    # Agrupa os objetos de retorno
    opt_data = (user_rated_items, all_item_ids)
    maps = (id_to_titulo, id_to_generos)
    
    return model_svd, opt_data, maps, max_user_id
# ==========================================================
