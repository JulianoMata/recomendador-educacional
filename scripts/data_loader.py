# scripts/data_loader.py
import streamlit as st
import pandas as pd
import pickle
import gdown
import os
from typing import Any, Tuple

# --- CONFIGURAÇÕES GLOBAIS ---
DATA_PATH = 'dados/ml-32m/ratings.csv'
MODEL_URL = 'https://drive.google.com/uc?id=1iJ3ttB73XXBC-GxWDH-mHJmF26n4e6_R'

# Caminho absoluto e seguro dentro do contêiner Streamlit
MODEL_PATH = os.path.join(os.getcwd(), 'svd_model_data.pkl')


@st.cache_data
def download_model(url: str, output_path: str) -> None:
    """Baixa o modelo do Google Drive se não existir localmente."""
    if not os.path.exists(output_path):
        st.warning(f"📦 Modelo não encontrado. Iniciando download (~1.9 GB)...")
        with st.spinner("Baixando... esta operação pode levar vários minutos."):
            try:
                gdown.download(url, output_path, quiet=False)

                # 🔍 Verifica se o arquivo realmente foi salvo
                if os.path.exists(output_path):
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    st.success(f"✅ Download concluído! ({file_size_mb:.1f} MB)")
                    st.info(f"📁 Arquivo salvo em: {output_path}")
                else:
                    st.error(f"❌ Download aparente concluído, mas o arquivo não foi encontrado em: {output_path}")
                    st.stop()

            except Exception as e:
                st.error(f"❌ Erro ao baixar o modelo: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                st.stop()
    else:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        st.info(f"✅ Modelo encontrado localmente ({file_size_mb:.1f} MB).")
        st.info(f"📁 Caminho: {output_path}")


@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Tuple, Tuple, int]:
    """Carrega modelo SVD e dados auxiliares."""
    download_model(MODEL_URL, MODEL_PATH)

    # 🔒 Garante que o arquivo realmente existe antes de abrir
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Arquivo de modelo não encontrado em: {MODEL_PATH}")
        st.stop()

    with st.spinner("🧠 Carregando modelo..."):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)

    model_svd = model_data['model']
    id_to_titulo = model_data['id_to_titulo']
    id_to_generos = model_data['id_to_generos']

    with st.spinner("📊 Lendo e processando a base de avaliações..."):
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

    return (
        model_svd,                    # Modelo SVD treinado
        (user_rated_items, all_item_ids),  # Dados otimizados
        (id_to_titulo, id_to_generos),     # Mapas auxiliares
        max_user_id                   # Maior ID de usuário
    )
