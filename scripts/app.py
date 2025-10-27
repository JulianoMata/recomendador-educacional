# scripts/app.py
import streamlit as st
import pandas as pd
from data_loader import load_model_and_data
from recommender import get_top_n_recommendations

# --- INICIALIZAÇÃO ---
model_als, user_item_matrix, user_map, item_map, maps, max_user_id = load_model_and_data()

if 'recommendations_df' not in st.session_state:
    st.session_state.recommendations_df = pd.DataFrame()

# --- INTERFACE DO USUÁRIO (UI) ---
st.title("🎓 Sistema de Recomendação de Conteúdo Educacional")
st.markdown("Protótipo baseado em **Filtros Colaborativos (ALS / Implicit)**.")

# >>> CORREÇÃO 1: Ordem e remoção de duplicatas na sidebar
st.sidebar.header("Configuração de Recomendações")
user_input = st.sidebar.text_input('ID do Aluno (Usuário):', value='1250')
n_recommendations = st.sidebar.slider('Número de Recomendações (Top-N):', min_value=5, max_value=25, value=10)

# --- LÓGICA DE EXECUÇÃO ---
if st.sidebar.button('Gerar Recomendações', type="primary"):
    # >>> CORREÇÃO 2: Validação simplificada
    if not user_input.strip(): # Verifica se o campo não está vazio
        st.warning('Por favor, informe um ID de Aluno.')
    else:
        with st.spinner(f'Gerando as Top-{n_recommendations} recomendações para o Aluno {user_input}...'):
            recs_df = get_top_n_recommendations(
                model_als,
                user_item_matrix,
                user_map,
                item_map,
                maps,
                user_input, # Passa a string diretamente
                n_recommendations
            )
            st.session_state.recommendations_df = recs_df
            st.session_state.current_user = user_input
            st.session_state.current_n = n_recommendations

# --- BLOCO DE EXIBIÇÃO ---
if not st.session_state.recommendations_df.empty:
    current_user = st.session_state.get('current_user', 'N/A')
    current_n = st.session_state.get('current_n', 10)
    st.subheader(f"✨ Top {current_n} Tópicos Sugeridos para o Aluno {current_user}")
    st.dataframe(st.session_state.recommendations_df, width='stretch', hide_index=True)
    st.caption("O 'Score de Afinidade' indica a força da recomendação do modelo.")
else:
    st.info("👈 Use o painel à esquerda para inserir um ID de Aluno e gerar recomendações.")
    st.markdown("---")
    st.image(
        "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbndiaGZjOHlxejE3Zzl1bGJhNGRtZjJuZXVtOWpieTBqYzlzdnI4YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/KX5nwoDX97AtPvKBF6/giphy.gif",
        caption="Processando Dados para Gerar Recomendações",
        width=300
    )
    st.caption("Desenvolvido por Juliano Mata - 2025 | Versão 2.0.0")
    st.markdown("---")
    