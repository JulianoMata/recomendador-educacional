# scripts/app.py

import streamlit as st
import pandas as pd

from data_loader import load_model_and_data
from recommender import get_top_n_recommendations

# --- Inicialização ---
# Carrega o modelo, dados otimizados, mapas e o max_user_id de uma só vez.
model_svd, opt_data, maps, max_user_id = load_model_and_data()

# Inicializa o estado da sessão para guardar as recomendações
if 'recommendations_df' not in st.session_state:
    st.session_state.recommendations_df = pd.DataFrame()

# --- Interface do Usuário (UI) ---
st.title("🎓 Sistema de Recomendação de Conteúdo Educacional")
st.markdown("Protótipo baseado em **Filtros Colaborativos (SVD)**. *(Base 32M)*")

st.sidebar.header("Configuração de Recomendações")
user_input = st.sidebar.number_input(
    'ID do Aluno (Usuário):', 
    min_value=1, 
    max_value=max_user_id,
    value=1250
)
n_recommendations = st.sidebar.slider(
    'Número de Recomendações (Top-N):', 
    min_value=5, 
    max_value=25, 
    value=10
)

# --- Lógica de Execução ---
if st.sidebar.button('Gerar Recomendações', type="primary"):
    with st.spinner(f'Gerando as Top-{n_recommendations} recomendações para o Aluno {user_input}...'):
        recs_df = get_top_n_recommendations(
            model_svd, opt_data, maps, user_input, n_recommendations
        )
        st.session_state.recommendations_df = recs_df
        st.session_state.current_user = user_input
        st.session_state.current_n = n_recommendations

# --- Bloco de Exibição dos Resultados ---
if not st.session_state.recommendations_df.empty:
    current_user = st.session_state.get('current_user', 'N/A')
    current_n = st.session_state.get('current_n', 10)
    
    st.subheader(f"✨ Top {current_n} Tópicos Sugeridos para o Aluno {current_user}")
    
    st.dataframe(
        st.session_state.recommendations_df, 
        width='stretch',
        hide_index=True
    )
    
    st.caption("A 'Nota Prevista' é a estimativa do modelo sobre o quanto o aluno irá gostar do tópico.")
else:
    # Mensagem inicial quando não há recomendações
    st.info("👈 Use o painel à esquerda para selecionar um ID de Aluno e gerar recomendações.")

    # --- Construção da Página Inicial em um único bloco ---
    welcome_page_markdown = f"""
        ---
        ### 📖 Sobre este Protótipo

        * **Tecnologia:** O motor de recomendação utiliza um modelo de **Filtros Colaborativos (SVD)**.
        * **Dados:** O modelo foi treinado em um subconjunto do dataset [MovieLens](https://grouplens.org/datasets/movielens/), contendo **32 milhões de interações**.
        * **Como Usar:** Insira um `ID de Aluno` válido no painel lateral e clique em "Gerar Recomendações".

        <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; margin-top: 25px; margin-bottom: 25px;">
            <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbndiaGZjOHlxejE3Zzl1bGJhNGRtZjJuZXVtOWpieTBqYzlzdnI4YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/KX5nwoDX97AtPvKBF6/giphy.gif" width="300">
            <p style="font-size: 0.9em; color: gray; margin-top: 5px;">Processando Dados para Gerar Recomendações</p>
        </div>

        ---

        <div style='font-size: 14px; text-align: center; color: #808080;'>
            <p>
                Desenvolvido por <strong>Juliano Mata</strong> - © 2025 | Versão 1.0.0 | Licença MIT<br>
                <a href="https://github.com/julianomata/recomendador-educacional" target="_blank">Código-Fonte no GitHub</a>
            </p>
            <p>
                Créditos: Dados por <a href="https://grouplens.org/datasets/movielens/" target="_blank">GroupLens</a> | 
                Framework por <a href="https://streamlit.io" target="_blank">Streamlit</a> | 
                Imagem por <a href="https://giphy.com/" target="_blank">Giphy</a>
            </p>
            <p style='font-size: 12px; font-style: italic;'>
                Atenção: Este é um protótipo para fins educacionais e não deve ser usado em produção.
            </p>
            <p>Streamlit v{st.__version__}</p>
        </div>
    """
    
    # Renderiza todo o bloco de uma só vez
    st.markdown(welcome_page_markdown, unsafe_allow_html=True)
    