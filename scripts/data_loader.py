# scripts/data_loader.py
import os
import pickle
from typing import Any, Dict, Set, Tuple

import gdown
import pandas as pd
import streamlit as st

# --- CONFIGURAÇÕES GLOBAIS ---
DATA_PATH = "dados/ml-32m/ratings.csv"
MODEL_URL = "https://drive.google.com/uc?id=1iJ3ttB73XXBC-GxWDH-mHJmF26n4e6_R"
MODEL_PATH = "svd_model_data.pkl"  # caminho relativo na raiz do app


def _ensure_parent_dir(path: str) -> None:
    """Cria o diretório pai caso não exista (se houver)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@st.cache_resource(show_spinner=False)
def download_model_if_missing(url: str, output_path: str) -> str:
    """
    Garante que o arquivo do modelo existe localmente.
    Faz o download via gdown se necessário e retorna o caminho final.
    Usa cache_resource para evitar re-downloads entre runs.
    """
    _ensure_parent_dir(output_path)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        st.info(f"🟢 Modelo encontrado localmente ({size_mb:.1f} MB).")
        return output_path

    st.warning("📦 Modelo não encontrado localmente. Iniciando download (~1.9 GB)...")
    with st.spinner("Baixando o modelo — isso pode levar vários minutos..."):
        try:
            # fuzzy=True ajuda a aceitar links de share do Drive
            gdown.download(url, output_path, quiet=False, fuzzy=True)
        except Exception as e:
            # tentativa simples de fallback via comando shell (pode ou não estar disponível)
            st.error(f"🔴 Erro no gdown: {e}. Tentando fallback via comando 'gdown' do sistema...")
            try:
                ret = os.system(f"gdown {url} -O {output_path}")
                if ret != 0:
                    raise RuntimeError(f"fallback gdown retornou {ret}")
            except Exception as e2:
                st.error(f"🔴 Fallback falhou: {e2}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                st.stop()

    # após tentativa de download, verificar existência
    if not os.path.exists(output_path):
        # às vezes gdown salva com nome diferente; checar possíveis alternativas
        possible = os.path.basename(url)
        if os.path.exists(possible):
            os.rename(possible, output_path)

    if not os.path.exists(output_path):
        st.error(f"🔴 Download concluído, mas '{output_path}' não foi criado.")
        st.stop()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    st.success(f"🟢 Download concluído com sucesso! ({size_mb:.1f} MB)")
    st.info(f"📁 Arquivo salvo em: {output_path}")
    return output_path


@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Tuple[Dict[str, Set[str]], Set[str]], Tuple[Dict[str, str], Dict[str, str]], int]:
    """
    Carrega (ou baixa e carrega) o modelo SVD e processa os dados de ratings.
    Retorna:
      - model_svd: o objeto do modelo (carregado do pickle)
      - opt_data: (user_rated_items, all_item_ids)
      - maps: (id_to_titulo, id_to_generos)
      - max_user_id: inteiro
    """
    # Garantir que o modelo está presente (faz download se necessário)
    model_file = download_model_if_missing(MODEL_URL, MODEL_PATH)

    # Segurança: checagem final
    if not os.path.exists(model_file):
        st.error(f"🔴 Arquivo de modelo não encontrado em: {model_file}")
        st.stop()

    # Carrega o pickle
    with st.spinner("🧠 Carregando modelo SVD do disco..."):
        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)
        except Exception as e:
            st.error(f"🔴 Falha ao ler o modelo: {e}")
            st.stop()

    # Extrai componentes do dicionário salvo
    model_svd = model_data.get("model")
    id_to_titulo = model_data.get("id_to_titulo", {})
    id_to_generos = model_data.get("id_to_generos", {})

    # Carrega e processa ratings
    with st.spinner("📊 Lendo e processando a base de avaliações..."):
        try:
            ratings_df = pd.read_csv(DATA_PATH)
        except Exception as e:
            st.error(f"🔴 Falha ao ler {DATA_PATH}: {e}")
            st.stop()

        # Ajuste de colunas e tipos
        if len(ratings_df.columns) >= 4:
            ratings_df = ratings_df.iloc[:, :4]  # seleciona as primeiras 4 colunas se houverem extras
        ratings_df.columns = ["user_id", "item_id", "rating", "timestamp"]
        ratings_df["user_id"] = ratings_df["user_id"].astype(str)
        ratings_df["item_id"] = ratings_df["item_id"].astype(str)

        try:
            max_user_id = int(ratings_df["user_id"].astype(int).max())
        except Exception:
            # fallback: se user_id não for int-conversível
            max_user_id = ratings_df["user_id"].nunique()

    # Preparar estruturas auxiliares
    with st.spinner("🔧 Preparando estruturas auxiliares para recomendações..."):
        user_rated_items: Dict[str, Set[str]] = {}
        for u, it in ratings_df[["user_id", "item_id"]].itertuples(index=False):
            user_rated_items.setdefault(u, set()).add(it)
        all_item_ids = set(ratings_df["item_id"].unique())

    st.success("🚀🙌 Modelo e dados carregados com sucesso!")

    opt_data = (user_rated_items, all_item_ids)
    maps = (id_to_titulo, id_to_generos)

    return model_svd, opt_data, maps, max_user_id
