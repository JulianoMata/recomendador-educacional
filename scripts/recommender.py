# scripts/recommender.py
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple
from scipy.sparse import csr_matrix

def get_top_n_recommendations(
    model: Any,
    user_item_matrix: csr_matrix,
    user_map: Any,
    item_map: Any,
    maps: Tuple,
    user_id: str,
    n: int = 10
) -> pd.DataFrame:
    """ Gera as Top-N Recomendações para um usuário usando um modelo da biblioteca 'implicit'. """
    
    id_to_titulo, id_to_generos = maps
    
    try:
        # Converte o ID de string do usuário para o ID numérico interno
        user_id_interno = user_map.categories.get_loc(user_id)
    except KeyError:
        st.warning(f"Usuário com ID {user_id} não encontrado no modelo.")
        return pd.DataFrame()

    # --- CORREÇÃO AQUI ---
    # A função .recommend() espera o ID do usuário e a LINHA da matriz correspondente a esse usuário.
    # O fatiamento `user_item_matrix[user_id_interno]` retorna a linha no formato CSR correto.
    recommended_items = model.recommend(
        user_id_interno,
        user_item_matrix[user_id_interno], # <-- Passamos a linha correta da matriz
        N=n,
        filter_already_liked_items=True
    )
    # --- FIM DA CORREÇÃO ---

    item_ids_internos, scores = recommended_items

    result_list = []
    for item_id_interno, score in zip(item_ids_internos, scores):
        original_item_id = item_map.categories[item_id_interno]
        
        result_list.append({
            'ID do Tópico': original_item_id,
            'Título do Tópico/Módulo': id_to_titulo.get(original_item_id, "N/A"),
            'Gêneros': id_to_generos.get(original_item_id, "N/A"),
            'Score de Afinidade': f"{score:.4f}"
        })

    return pd.DataFrame(result_list)
# ==========================================================
