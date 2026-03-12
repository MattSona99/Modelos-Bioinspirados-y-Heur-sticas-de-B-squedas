from utils import distancia_manhattan_km

# 1. Algoritmo Greedy (Vecino Más Cercano)

def greedy_algorithm(estaciones_base, coordenadas):
    """
    Construye una ruta utilizando una heurística Greedy (Vecino Más Cercano).
    Partiendo de la Estación 0, busca siempre la estación no visitada más cercana.

    Args:
        estaciones_base (lista): Lista de IDs de estaciones que necesitan ser visitadas.
        coordenadas (lista): Diccionario o lista con las coordenadas de todas las estaciones.
    Returns:
        ruta (lista): Lista de IDs de estaciones en el orden en que se visitan.
    """
    ruta_greedy = []
    # Usamos un set para poder ir eliminado fácilmente las estaciones visitadas
    estaciones_pendientes = set(estaciones_base)
    
    # El camión empieza en la estación 0
    estacion_actual = 0
    
    while estaciones_pendientes:
        estacion_mas_cercana = None
        min_distancia = float('inf')
        
        lat_actual = coordenadas[estacion_actual]['lat']
        lon_actual = coordenadas[estacion_actual]['lon']
        
        # Se busca la estación más cercana entre las pendientes
        for candidata in estaciones_pendientes:
            lat_cand = coordenadas[candidata]['lat']
            lon_cand = coordenadas[candidata]['lon']
            
            dist = distancia_manhattan_km(lat_actual, lon_actual, lat_cand, lon_cand)
            
            if dist < min_distancia:
                min_distancia = dist
                estacion_mas_cercana = candidata
                
        # Se agrega la mejor candidata a la ruta y se elimina de pendientes
        ruta_greedy.append(estacion_mas_cercana)
        estaciones_pendientes.remove(estacion_mas_cercana)
        
        # El camión se "mueve" a esa estación para la siguiente iteración
        estacion_actual = estacion_mas_cercana
        
    return ruta_greedy