# scripts/data_loader.py

import streamlit as st
import pandas as pd
import pickle
from typing import Dict, Any, Tuple, Set

# --- CONFIGURAÇÃO ---
DATA_PATH = 'dados/ml-32m/ratings.csv'
# 👇👇👇 COLE O LINK CORRETO AQUI
MODEL_URL = 'https://drive.google.com/uc?export=download&id=18S57U_3Pic74oFef-2QjwkbuvjUFEYXe' 
MODEL_PATH = 'svd_model_data.pkl'

@st.cache_resource
def load_model_and_data() -> Tuple[Any, Tuple[Dict[str, Set[str]], Set[str]], Tuple[Dict[str, str], Dict[str, str]], int]:
    st.info("Carregando modelo pré-treinado e dados...")
    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    model_svd = model_data['model']
    id_to_titulo = model_data['id_to_titulo']
    id_to_generos = model_data['id_to_generos']

    ratings_df = pd.read_csv(DATA_PATH)
    ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['item_id'] = ratings_df['item_id'].astype(str)

    # Calcula o max_user_id aqui dentro, onde temos o DataFrame
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
