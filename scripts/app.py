# scripts/app.py

import streamlit as st
import pandas as pd

# Importa as funções dos nossos módulos
from data_loader import load_model_and_data
from recommender import get_top_n_recommendations

# ====================================================================
# INTERFACE STREAMLIT
# ====================================================================

# Inicializa o estado da sessão
if 'recommendations_df' not in st.session_state:
    st.session_state.recommendations_df = pd.DataFrame()

# Carregar modelo e dados (APENAS UMA VEZ)
model_svd, opt_data, maps, df_ratings = load_model_and_data()

st.title("🎓 Sistema de Recomendação de Conteúdo Educacional")
st.markdown("Protótipo baseado em **Filtros Colaborativos (SVD)**. *(Base 32M)*")

# Sidebar para input
st.sidebar.header("Configuração de Recomendações")
max_user_id = int(df_ratings['user_id'].astype(int).max())
user_input = st.sidebar.number_input(
    'ID do Aluno (Usuário):', min_value=1, max_value=max_user_id, value=1250
)
n_recommendations = st.sidebar.slider(
    'Número de Recomendações (Top-N):', min_value=5, max_value=25, value=10
)

# Botão de Execução
if st.sidebar.button('Gerar Recomendações', type="primary"):
    with st.spinner(f'Gerando as Top-{n_recommendations} recomendações para o Aluno {user_input}...'):
        recs_df = get_top_n_recommendations(
            model_svd, opt_data, maps, user_input, n_recommendations
        )
        st.session_state.recommendations_df = recs_df
        st.session_state.current_user = user_input
        st.session_state.current_n = n_recommendations

# Bloco de Exibição dos Resultados
if not st.session_state.recommendations_df.empty:
    current_user = st.session_state.get('current_user', 'N/A')
    current_n = st.session_state.get('current_n', 10)
    
    st.subheader(f"✨ Top {current_n} Tópicos Sugeridos para o Aluno {current_user}")
    
    # LINHA CORRIGIDA AQUI:
    st.dataframe(
        st.session_state.recommendations_df, 
        width='stretch',
        hide_index=True
    )
    
    st.caption("A 'Nota Prevista' é a estimativa do modelo sobre o quanto o aluno irá gostar do tópico.")