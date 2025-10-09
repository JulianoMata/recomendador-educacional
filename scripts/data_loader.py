# scripts/data_loader.py

import streamlit as st
import pandas as pd
import pickle
import requests
import os
from typing import Dict, Any, Tuple, Set

# --- CONFIGURAÇÃO ---
DATA_PATH = 'dados/ml-32m/ratings.csv'
# 👇👇👇 COLE O LINK CORRETO AQUI
MODEL_URL = 'https://drive.google.com/uc?export=download&id=18S57U_3Pic74oFef-2QjwkbuvjUFEYXe' 
MODEL_PATH = 'svd_model_data.pkl'

@st.cache_data
def download_model(url, filepath):
    """Baixa o modelo de uma URL se ele não existir localmente, com barra de progresso."""
    if not os.path.exists(filepath):
        st.info(f"Modelo não encontrado. Iniciando download (aprox. 1.1 GB)...")
        progress_text = "Baixando modelo... Isso pode levar alguns minutos, dependendo da conexão."
        progress_bar = st.progress(0, text=progress_text)
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size_in_bytes > 0:
                        # Atualiza a barra de progresso
                        progress = min(1.0, bytes_downloaded / total_size_in_bytes)
                        progress_bar.progress(progress, text=f"Baixando modelo... ({bytes_downloaded/1024/1024:.1f} / {total_size_in_bytes/1024/1024:.1f} MB)")
        
        progress_bar.progress(1.0, text="Download concluído!")
        progress_bar.empty()

@st.cache_data
def download_model(url, filepath):
    """Baixa o modelo de uma URL se ele não existir localmente, com barra de progresso."""
    if not os.path.exists(filepath):
        st.info(f"Modelo não encontrado localmente. Iniciando download...")
        progress_text = "Baixando modelo (aprox. 1.1 GB)... Isso pode levar alguns minutos."
        progress_bar = st.progress(0, text=progress_text)
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size_in_bytes > 0:
                        progress = min(1.0, bytes_downloaded / total_size_in_bytes)
                        progress_bar.progress(progress, text=f"Baixando modelo... ({bytes_downloaded/1024/1024:.1f} / {total_size_in_bytes/1024/1024:.1f} MB)")
        
        progress_bar.progress(1.0, text="Download concluído!")
        progress_bar.empty()

@st.cache_resource
def load_model_and_data() -> Tuple[Any, Tuple[Dict[str, Set[str]], Set[str]], Tuple[Dict[str, str], Dict[str, str]], int]:
    
    download_model(MODEL_URL, MODEL_PATH)

    st.info(f"Carregando modelo do arquivo '{MODEL_PATH}'...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    model_svd = model_data['model']
    id_to_titulo = model_data['id_to_titulo']
    id_to_generos = model_data['id_to_generos']

    ratings_df = pd.read_csv(DATA_PATH)
    ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['item_id'] = ratings_df['item_id'].astype(str)

    max_user_id = int(ratings_df['user_id'].astype(int).max())

    st.info("Construindo estruturas de dados para o motor de recomendação...")

    user_rated_items = {} 
    
    for row in ratings_df[['user_id', 'item_id']].itertuples(index=False):
        user_rated_items.setdefault(row.user_id, set()).add(row.item_id)
    
    all_item_ids = set(ratings_df['item_id'].unique())
    
    st.success("Carga e preparação concluídas!")
    
    opt_data = (user_rated_items, all_item_ids)
    maps = (id_to_titulo, id_to_generos)
    
    return model_svd, opt_data, maps, max_user_id
