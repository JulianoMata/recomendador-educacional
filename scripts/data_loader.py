# scripts/data_loader.py
import os
import pickle
import time
from typing import Any, Dict, Set, Tuple

import pandas as pd
import requests  # Trocamos gdown por requests
import streamlit as st

# --- CONFIGURAÇÕES GLOBAIS ---
DATA_PATH = "dados/ml-32m/ratings.csv"
# 👇👇👇 URL ATUALIZADA PARA O HUGGING FACE
MODEL_URL = "https://huggingface.co/JulianoMata/recomendador-educacional-svd/resolve/main/svd_model_data.pkl?download=true"
MODEL_PATH = "svd_model_data.pkl"  # Caminho relativo na raiz do app

@st.cache_resource(show_spinner=False)
def download_model_if_missing(url: str, output_path: str) -> None:
    """
    Garante que o arquivo do modelo existe localmente.
    Faz o download do Hugging Face com uma barra de progresso detalhada.
    """
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        st.info(f"🟢 Modelo encontrado localmente ({size_mb:.1f} MB).")
        return

    st.warning("📦 Modelo não encontrado. Iniciando download (~1.9 GB)...")
    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_time = time.time()

    try:
        with requests.get(url, stream=True, timeout=(10, 300)) as r: # Timeout de conexão e leitura
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size > 0:
                        progress = bytes_downloaded / total_size
                        elapsed = time.time() - start_time
                        speed = bytes_downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                        
                        progress_bar.progress(min(1.0, progress))
                        progress_text.markdown(f"⬇️ Baixando... {bytes_downloaded/1024/1024:.1f} / {total_size/1024/1024:.1f} MB (`{speed:.2f} MB/s`)")

        progress_bar.empty()
        progress_text.empty()
        st.success("🟢 Download concluído com sucesso!")

    except Exception as e:
        st.error(f"🔴 Erro fatal ao baixar o modelo: {e}")
        if os.path.exists(output_path):
            os.remove(output_path) # Remove arquivo parcial
        st.stop()

@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Tuple, Tuple, int]:
    """
    Orquestra todo o processo de carga: download, leitura do modelo e processamento dos dados.
    """
    # Garantir que o modelo está presente (faz download se necessário)
    download_model_if_missing(MODEL_URL, MODEL_PATH)

    # O resto da sua função continua exatamente igual, pois a lógica de carregar
    # o pickle e processar o pandas não muda.
    with st.spinner("🧠 Carregando modelo SVD do disco..."):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

    model_svd = model_data.get("model")
    id_to_titulo = model_data.get("id_to_titulo", {})
    id_to_generos = model_data.get("id_to_generos", {})

    with st.spinner("📊 Lendo e processando a base de avaliações..."):
        ratings_df = pd.read_csv(DATA_PATH)
        ratings_df.columns = ["user_id", "item_id", "rating", "timestamp"]
        ratings_df["user_id"] = ratings_df["user_id"].astype(str)
        ratings_df["item_id"] = ratings_df["item_id"].astype(str)
        max_user_id = int(ratings_df["user_id"].astype(int).max())

    with st.spinner("🔧 Preparando estruturas auxiliares..."):
        user_rated_items: Dict[str, Set[str]] = {}
        for u, it in ratings_df[["user_id", "item_id"]].itertuples(index=False):
            user_rated_items.setdefault(u, set()).add(it)
        all_item_ids = set(ratings_df["item_id"].unique())

    st.success("🚀🙌 Modelo e dados carregados com sucesso!")

    opt_data = (user_rated_items, all_item_ids)
    maps = (id_to_titulo, id_to_generos)

    return model_svd, opt_data, maps, max_user_id
    
