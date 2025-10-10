# scripts/recommender.py

import pandas as pd
from typing import Dict, Any, Tuple, Set

def get_top_n_recommendations(
    algo: Any, 
    opt_data: Tuple[Dict[str, Set[str]], Set[str]], 
    maps: Tuple[Dict[str, str], Dict[str, str]], 
    user_id: int, 
    n: int = 10
) -> pd.DataFrame:
    """
    Gera as Top-N Recomendações para um usuário específico, excluindo itens que ele já avaliou.

    Args:
        algo: O modelo de recomendação treinado (da biblioteca Surprise).
        opt_data: Tupla com dados otimizados (dicionário de itens por usuário, conjunto de todos os itens).
        maps: Tupla com os dicionários de metadados (mapa de ID para título, mapa de ID para gênero).
        user_id: O ID do usuário para o qual gerar as recomendações.
        n: O número de recomendações a serem retornadas.

    Returns:
        Um DataFrame do pandas com as N melhores recomendações formatadas.
    """
    # --- 1. Desempacotamento dos Dados ---
    # Extrai as estruturas de dados pré-processadas que foram carregadas pelo data_loader
    user_rated_items, all_item_ids = opt_data
    id_to_titulo, id_to_generos = maps
    user_id_str = str(user_id)

    # --- 2. Filtro de Itens ---
    # Pega o conjunto de IDs de itens que o usuário já avaliou.
    # Usa .get() com um conjunto vazio como padrão para o caso de um usuário novo.
    items_already_rated = user_rated_items.get(user_id_str, set())

    # Cria a lista de itens candidatos para a previsão, removendo os que já foram vistos.
    # A operação de diferença de conjuntos (set) é extremamente rápida.
    items_to_predict = all_item_ids - items_already_rated
    
    # --- 3. Geração das Previsões ---
    # Itera sobre cada item candidato e usa o método .predict() do modelo para estimar a nota.
    # Esta é a etapa mais intensiva em processamento.
    predictions = [(item_id, algo.predict(user_id_str, item_id).est) for item_id in items_to_predict]
    
    # --- 4. Ordenação e Seleção do Top-N ---
    # Ordena a lista de previsões em ordem decrescente com base na nota estimada.
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Seleciona os N melhores itens do topo da lista.
    top_n = predictions[:n]
    
    # --- 5. Formatação do Resultado Final ---
    # Cria uma lista de dicionários com os resultados, adicionando os metadados
    # (título e gênero) para cada recomendação, prontos para virar uma tabela.
    result_list = [
        {
            'ID do Tópico': item_id,
            'Título do Tópico/Módulo': id_to_titulo.get(item_id, "N/A"),
            'Gêneros': id_to_generos.get(item_id, "N/A"),
            'Nota Prevista (1-5)': f"{est_rating:.2f}"
        }
        for item_id, est_rating in top_n
    ]

    # Converte a lista final em um DataFrame do pandas para ser exibido na interface.
    return pd.DataFrame(result_list)
