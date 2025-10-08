# scripts/data_loader.py

import streamlit as st
import pandas as pd
import pickle
from typing import Dict, Any, Tuple, Set

DATA_PATH = 'dados/ml-32m/ratings.csv'
MODEL_PATH = 'svd_model_data.pkl'

@st.cache_resource
def load_model_and_data() -> Tuple[Any, Tuple[Dict[str, Set[str]], Set[str]], Tuple[Dict[str, str], Dict[str, str]], pd.DataFrame]:
    """
    Carrega o modelo SVD pré-treinado e os dados necessários para as recomendações.
    """
    st.info("Carregando modelo pré-treinado e dados...")
    
    # Carrega o DataFrame de ratings
    ratings_df = pd.read_csv(DATA_PATH)
    ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['item_id'] = ratings_df['item_id'].astype(str)

    # Carrega o modelo e os mapas do arquivo .pkl
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    model_svd = model_data['model']
    id_to_titulo = model_data['id_to_titulo']
    id_to_generos = model_data['id_to_generos']

    # --- INÍCIO DA OTIMIZAÇÃO DE MEMÓRIA ---
    st.info("Construindo mapa de itens avaliados (otimizado para memória)...")

    # 1. Crie um dicionário vazio.
    user_rated_items = {}
    
    # 2. Use .itertuples(), que é muito mais eficiente em memória que o groupby().apply().
    #    Selecionamos apenas as colunas necessárias para diminuir ainda mais o uso de memória.
    for row in ratings_df[['user_id', 'item_id']].itertuples(index=False):
        # 3. Para cada linha, adicione o item_id ao conjunto do usuário.
        #    O .setdefault() cria o conjunto (set) na primeira vez que um usuário é encontrado
        #    e o reutiliza nas vezes seguintes.
        user_rated_items.setdefault(row.user_id, set()).add(row.item_id)

    # --- FIM DA OTIMIZAÇÃO DE MEMÓRIA ---

    all_item_ids = set(ratings_df['item_id'].unique())
    
    st.success(f"Modelo e dados ({len(ratings_df):,} interações) carregados!")
    
    return model_svd, (user_rated_items, all_item_ids), (id_to_titulo, id_to_generos), ratings_df