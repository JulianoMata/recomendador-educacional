# scripts/data_loader.py
import os
import pickle
import time 
from typing import Any, Tuple, Dict, Set

import pandas as pd
import requests  # Usamos requests para o download do Hugging Face
import streamlit as st

# --- CONFIGURAÇÕES GLOBAIS ---
DATA_PATH = "dados/ml-32m/ratings.csv"
# 👇DOWNLOAD DIRETO DO HUGGING FACE
MODEL_URL = "https://huggingface.co/JulianoMata/recomendador-educacional-als/resolve/main/implicit_als_model.pkl"
MODEL_PATH = "implicit_als_model.pkl"  # Caminho relativo na raiz do projeto

# ==========================================================
#   FUNÇÃO: DOWNLOAD DO MODELO
# ==========================================================
@st.cache_resource(show_spinner=False)
def download_model_if_missing(url: str, output_path: str) -> None:
    """
    Garante que o arquivo do modelo existe localmente.
    Faz o download do Hugging Face com uma barra de progresso.
    """
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        st.info(f"🟢 Modelo encontrado localmente ({size_mb:.1f} MB).")
        return

    st.warning("📦 Modelo não encontrado. Iniciando download (~1.9 GB)...")
    progress_bar = st.progress(0)
    progress_text = st.empty()

    try:
        with requests.get(url, stream=True, timeout=(10, 300)) as r: # Timeout (conexão, leitura)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size > 0:
                        progress = bytes_downloaded / total_size
                        progress_bar.progress(min(1.0, progress))
                        progress_text.text(f"Baixando... {bytes_downloaded/1024/1024:.1f} / {total_size/1024/1024:.1f} MB")
        
        progress_bar.empty()
        progress_text.empty()
        st.success("🟢 Download concluído com sucesso!")

    except Exception as e:
        st.error(f"🔴 Erro fatal ao baixar o modelo: {e}")
        if os.path.exists(output_path):
            os.remove(output_path) # Remove arquivo parcial
        st.stop()

# ==========================================================
#   FUNÇÃO: CARREGAMENTO PRINCIPAL
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Any, Any, Any, Tuple, int]:
    """
    Orquestra todo o processo de carga: download, leitura do modelo e processamento dos dados.
    """
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
# ==========================================================
