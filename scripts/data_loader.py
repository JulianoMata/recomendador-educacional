# scripts/data_loader.py
import os
import pickle
from typing import Any, Tuple

import gdown
import pandas as pd
import streamlit as st

# --- CONFIGURAÇÕES GLOBAIS ---
MODEL_URL = "https://drive.google.com/file/d/1nVwaFt5e2nmaNIOvE-MOh96RNm_cZa8q"
MODEL_PATH = "implicit_als_model.pkl"

@st.cache_data(show_spinner=False)
def download_model_if_missing(url: str, output_path: str) -> None:
    """Garante que o arquivo do modelo existe, fazendo download se necessário."""
    if os.path.exists(output_path):
        st.info("✅ Modelo encontrado localmente.")
        return

    st.warning("📦 Modelo não encontrado. Iniciando download...")
    with st.spinner("Baixando o modelo..."):
        try:
            gdown.download(url, output_path, quiet=False)
        except Exception as e:
            st.error(f"🔴 Erro fatal ao baixar o modelo: {e}")
            st.stop()
    st.success("✅ Download concluído!")

@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Any, Any, Any, Tuple, int]:
    """Carrega (ou baixa) o modelo ALS e todos os objetos necessários."""
    
    download_model_if_missing(MODEL_URL, MODEL_PATH)

    with st.spinner("🧠 Carregando modelo ALS do disco..."):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

    # Extrai todos os componentes salvos no .pkl
    model_als = model_data.get("model")
    user_map = model_data.get("user_map")
    item_map = model_data.get("item_map")
    user_item_matrix = model_data.get("user_item_matrix")
    id_to_titulo = model_data.get("id_to_titulo", {})
    id_to_generos = model_data.get("id_to_generos", {})
    
    # O número máximo de usuários agora vem do mapa que salvamos
    max_user_id = len(user_map.categories)

    st.success("🚀🙌 Modelo e dados carregados com sucesso!")

    maps = (id_to_titulo, id_to_generos)

    return model_als, user_item_matrix, user_map, item_map, maps, max_user_id
