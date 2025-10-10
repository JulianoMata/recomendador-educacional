# scripts/data_loader.py
import streamlit as st
import pandas as pd
import pickle
import gdown
import os
from typing import Dict, Any, Tuple, Set

# --- CONFIGURAÇÕES GLOBAIS ---
DATA_PATH = 'dados/ml-32m/ratings.csv'
# Link de compartilhamento normal do Google Drive
MODEL_URL = 'https://drive.google.com/file/d/1RJJcByH0cx83orS334n-5uSEHN8LGmeS/view?usp=sharing'
MODEL_PATH = 'svd_model_data.pkl'

@st.cache_data
def download_model(url: str, output_path: str) -> None:
    """Verifica e baixa o modelo do Google Drive usando gdown."""
    if not os.path.exists(output_path):
        st.warning(f"📦 Modelo não encontrado. Iniciando download (~1.9 GB)...")
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

@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Tuple, Tuple, int]:
    download_model(MODEL_URL, MODEL_PATH)

    with st.spinner("🧠 Carregando modelo..."):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
    
    model_svd = model_data['model']
    id_to_titulo = model_data['id_to_titulo']
    id_to_generos = model_data['id_to_generos']
    
    with st.spinner(f"📊 Lendo e processando a base de avaliações..."):
        ratings_df = pd.read_csv(DATA_PATH)
        ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df['user_id'] = ratings_df['user_id'].astype(str)
        ratings_df['item_id'] = ratings_df['item_id'].astype(str)
        max_user_id = int(ratings_df['user_id'].astype(int).max())
    
    with st.spinner("🔧 Preparando o motor de recomendação..."):
        user_rated_items = {}
        for row in ratings_df[['user_id', 'item_id']].itertuples(index=False):
            user_rated_items.setdefault(row.user_id, set()).add(row.item_id)
        all_item_ids = set(ratings_df['item_id'].unique())
    
    st.success("🚀 Modelo e dados prontos!")
    
    opt_data = (user_rated_items, all_item_ids)
    maps = (id_to_titulo, id_to_generos)
    
    return model_svd, opt_data, maps, max_user_id
