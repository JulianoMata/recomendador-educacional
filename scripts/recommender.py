# scripts/recommender.py
import pandas as pd
from typing import Dict, Any, Tuple, Set

def get_top_n_recommendations(algo: Any, opt_data: Tuple, maps: Tuple, user_id: int, n: int = 10) -> pd.DataFrame:
    user_rated_items, all_item_ids = opt_data
    id_to_titulo, id_to_generos = maps
    user_id_str = str(user_id)

    items_already_rated = user_rated_items.get(user_id_str, set())
    items_to_predict = all_item_ids - items_already_rated
    
    predictions = [(item_id, algo.predict(user_id_str, item_id).est) for item_id in items_to_predict]
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    
    result_list = [
        {
            'ID do Tópico': item_id,
            'Título do Tópico/Módulo': id_to_titulo.get(item_id, "N/A"),
            'Gêneros': id_to_generos.get(item_id, "N/A"),
            'Nota Prevista (1-5)': f"{est_rating:.2f}"
        }
        for item_id, est_rating in top_n
    ]

    return pd.DataFrame(result_list)
