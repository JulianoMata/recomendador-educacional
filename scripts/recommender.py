# scripts/recommender.py
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple
from scipy.sparse import csr_matrix

# 👇👇👇 ALTERAÇÃO AQUI: user_id agora é do tipo 'str'
def get_top_n_recommendations(
    model: Any,
    user_item_matrix: csr_matrix,
    user_map: Any,
    item_map: Any,
    maps: Tuple,
    user_id: str, # <-- MUDANÇA DE 'int' PARA 'str'
    n: int = 10
) -> pd.DataFrame:
    """ Gera as Top-N Recomendações... """
    
    id_to_titulo, id_to_generos = maps
    # A conversão para string não é mais necessária aqui
    
    try:
        # Usa a string 'user_id' diretamente
        user_id_interno = user_map.categories.get_loc(user_id)
    except KeyError:
        st.warning(f"Usuário com ID {user_id} não encontrado no modelo.")
        return pd.DataFrame()

    # --- 2. Geração Otimizada das Recomendações ---
    # A função .recommend() já filtra os itens que o usuário interagiu.
    recommended_items = model.recommend(
        user_id_interno,
        user_item_matrix[user_id_interno],
        N=n,
        filter_already_liked_items=True
    )

    item_ids_internos, scores = recommended_items

    # --- 3. Formatação do Resultado Final ---
    result_list = []
    for item_id_interno, score in zip(item_ids_internos, scores):
        # Converte o ID interno do item de volta para o ID original (string)
        original_item_id = item_map.categories[item_id_interno]
        
        result_list.append({
            'ID do Tópico': original_item_id,
            'Título do Tópico/Módulo': id_to_titulo.get(original_item_id, "N/A"),
            'Gêneros': id_to_generos.get(original_item_id, "N/A"),
            'Score de Afinidade': f"{score:.4f}"
        })

    return pd.DataFrame(result_list)
