# scripts/data_loader.py

# --- 1. IMPORTAÇÕES ---
import streamlit as st
import pandas as pd
import pickle
import requests
import os
import time
from typing import Dict, Any, Tuple, Set

# --- 2. CONFIGURAÇÕES GLOBAIS ---
DATA_PATH = 'dados/ml-32m/ratings.csv'
# URL de download direto para o modelo pré-treinado (hospedado no Google Drive)
MODEL_URL = 'https://drive.google.com/uc?export=download&id=18S57U_3Pic74oFef-2QjwkbuvjUFEYXe'
# Nome do arquivo do modelo como será salvo localmente na raiz do projeto
MODEL_PATH = 'svd_model_data.pkl'


# ==========================================================
#   FUNÇÃO: DOWNLOAD DO MODELO
# ==========================================================
# @st.cache_data garante que o download só aconteça uma vez.
# Se o arquivo já foi baixado, o Streamlit não executa a função novamente.
@st.cache_data
def download_model(url: str, filepath: str) -> None:
    """
    Verifica se o arquivo do modelo existe localmente. Se não, faz o download
    de uma URL, exibindo uma barra de progresso detalhada com velocidade e ETA.
    """

    # Verifica se o arquivo já existe para evitar downloads desnecessários
    if os.path.exists(filepath):
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        st.info(f"✅ Modelo encontrado localmente ({file_size_mb:.1f} MB).")
        return # Encerra a função se o arquivo já existe

    st.warning(f"📦 Modelo '{filepath}' não encontrado. Iniciando download (~1.9 GB)...")

    try:
        # Inicia o download usando requests com stream=True, essencial para arquivos grandes,
        # pois não carrega todo o conteúdo na memória de uma vez.
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status() # Garante que a resposta foi bem-sucedida (código 200)

            # Pega o tamanho total do arquivo do cabeçalho da resposta para a barra de progresso
            total_size = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            block_size = 1024 * 1024  # Define pedaços de 1 MB para o download

            # Prepara os elementos da UI do Streamlit que serão atualizados dinamicamente
            progress_bar = st.progress(0)
            progress_text = st.empty()
            start_time = time.time()

            # Abre o arquivo local em modo de "escrita binária" ('wb') para salvar os dados
            with open(filepath, 'wb') as f:
                # Itera sobre o download em pedaços (chunks)
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk: # Filtra chunks de "keep-alive" que podem vir vazios
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

                        # Apenas calcula e exibe as métricas se o download for grande o suficiente
                        if total_size and (time.time() - start_time) > 0.5:
                            progress = bytes_downloaded / total_size
                            elapsed = time.time() - start_time
                            speed = bytes_downloaded / (1024 * 1024 * elapsed)
                            remaining_time = (total_size - bytes_downloaded) / (speed * 1024 * 1024) if speed > 0 else 0

                            # Atualiza a barra de progresso e o texto com as métricas
                            progress_text.markdown(
                                f"⬇️ **Baixando modelo...** "
                                f"{bytes_downloaded/1024/1024:.1f} / {total_size/1024/1024:.1f} MB "
                                f"(`{speed:.2f} MB/s`, tempo restante: {remaining_time:.0f}s)"
                            )
                            progress_bar.progress(min(1.0, progress))

        # Limpa os elementos da UI após o download ser concluído
        progress_bar.empty()
        progress_text.empty()
        st.success("✅ Download concluído!")

    except Exception as e:
        # Em caso de erro durante o download, remove o arquivo parcial para evitar corrupção
        if os.path.exists(filepath):
            os.remove(filepath)
        st.error(f"❌ Erro fatal ao baixar o modelo: {e}")
        st.stop() # Interrompe a execução do app se o modelo não puder ser baixado


# ==========================================================
#   FUNÇÃO: CARREGAMENTO PRINCIPAL
# ==========================================================
# @st.cache_resource executa esta função inteira apenas UMA VEZ.
# show_spinner=False desativa o spinner genérico, pois temos os nossos customizados.
@st.cache_resource(show_spinner=False)
def load_model_and_data() -> Tuple[Any, Tuple[Dict[str, Set[str]], Set[str]], Tuple[Dict[str, str], Dict[str, str]], int]:
    """
    Orquestra todo o processo de carga: download do modelo, leitura dos dados
    e preparação das estruturas de dados otimizadas para a recomendação.
    """

    # --- 1. Garante que o modelo está disponível localmente ---
    download_model(MODEL_URL, MODEL_PATH)

    # --- 2. Carrega o modelo do arquivo .pkl para a memória ---
    with st.spinner("🧠 Carregando modelo de recomendação..."):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)

    # Desempacota os objetos do dicionário salvo
    model_svd = model_data['model']
    id_to_titulo = model_data['id_to_titulo']
    id_to_generos = model_data['id_to_generos']

    # --- 3. Leitura e processamento da base de avaliações ---
    with st.spinner(f"📊 Lendo e processando a base de avaliações..."):
        ratings_df = pd.read_csv(DATA_PATH)
        ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df['user_id'] = ratings_df['user_id'].astype(str)
        ratings_df['item_id'] = ratings_df['item_id'].astype(str)
        max_user_id = int(ratings_df['user_id'].astype(int).max())

    # --- 4. Construção de estruturas otimizadas para performance ---
    with st.spinner("🔧 Preparando o motor de recomendação..."):
        # Cria um dicionário: {user_id: {item_id_1, item_id_2, ...}}
        # Este método (itertuples) é otimizado para baixo consumo de memória.
        user_rated_items = {}
        for row in ratings_df[['user_id', 'item_id']].itertuples(index=False):
            user_rated_items.setdefault(row.user_id, set()).add(row.item_id)
        
        # Cria um conjunto com todos os IDs de itens únicos para a lógica de previsão
        all_item_ids = set(ratings_df['item_id'].unique())

    st.success("🚀 Modelo e dados prontos para uso!")

    # Agrupa os objetos de retorno para uma saída mais limpa da função
    opt_data = (user_rated_items, all_item_ids)
    maps = (id_to_titulo, id_to_generos)
    
    return model_svd, opt_data, maps, max_user_id
