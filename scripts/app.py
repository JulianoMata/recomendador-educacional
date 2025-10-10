# scripts/app.py

import streamlit as st
import pandas as pd
from data_loader import load_model_and_data
from recommender import get_top_n_recommendations

# --- INICIALIZAÇÃO ---
# Carrega o modelo, dados otimizados, mapas e o max_user_id de uma só vez.
model_svd, opt_data, maps, max_user_id = load_model_and_data()

# Inicializa o estado da sessão para guardar as recomendações
if 'recommendations_df' not in st.session_state:
    st.session_state.recommendations_df = pd.DataFrame()

# --- INTERFACE DO USUÁRIO (UI) ---
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

# --- LÓGICA DE EXECUÇÃO ---
if st.sidebar.button('Gerar Recomendações', type="primary"):
    with st.spinner(f'Gerando as Top-{n_recommendations} recomendações para o Aluno {user_input}...'):
        recs_df = get_top_n_recommendations(
            model_svd, opt_data, maps, user_input, n_recommendations
        )
        st.session_state.recommendations_df = recs_df
        st.session_state.current_user = user_input
        st.session_state.current_n = n_recommendations

# --- BLOCO DE EXIBIÇÃO DOS RESULTADOS ---
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

    st.markdown("---")
    st.markdown("### 📖 Sobre este Protótipo")
    st.markdown("""
    * **Tecnologia:** O motor de recomendação utiliza um modelo de **Filtros Colaborativos (SVD)**.
    * **Dados:** O modelo foi treinado em um subconjunto do dataset [MovieLens](https://grouplens.org/datasets/movielens/), contendo **32 milhões de interações**.
    * **Como Usar:** Insira um `ID de Aluno` válido e clique em "Gerar Recomendações".
    """)
    
    st.image(
        "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbndiaGZjOHlxejE3Zzl1bGJhNGRtZjJuZXVtOWpieTBqYzlzdnI4YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/KX5nwoDX97AtPvKBF6/giphy.gif",
        caption="Processando Dados para Gerar Recomendações",
        width=300
    )
    
    st.caption("Desenvolvido por Juliano Mata - 2025 | Versão 1.0.0")
    