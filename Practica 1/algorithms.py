from utils import distancia_manhattan_km, generar_solucion_inicial_aleatoria, generar_vecino_swap
import random
import math

# ----------------------------------------
# 1. Algoritmo Greedy (Vecino Más Cercano)
# ----------------------------------------

def greedy_algorithm(
    funcion_objetivo,
    estaciones_base,
    coordenadas,
    caso_bicis,
    caso_capacidad,
    evaluar_ruta,
    semilla=None,
    **kwargs
    ):
    """
    Construye una ruta utilizando una heurística Greedy (Vecino Más Cercano).
    Partiendo de la Estación 0, busca siempre la estación no visitada más cercana.
    """
    ruta_greedy = []
    estaciones_pendientes = set(estaciones_base)
    estacion_actual = 0

    # La estación 0 no se visita (es el punto de partida y llegada),
    # pero sí se procesa su intercambio inicial
    while estaciones_pendientes:
        estacion_mas_cercana = None
        min_distancia = float('inf')
        lat_actual, lon_actual = coordenadas[estacion_actual]['lat'], coordenadas[estacion_actual]['lon']

        # Encontrar la estación más cercana entre las pendientes
        for candidata in estaciones_pendientes:
            lat_cand, lon_cand = coordenadas[candidata]['lat'], coordenadas[candidata]['lon']
            dist = distancia_manhattan_km(lat_actual, lon_actual, lat_cand, lon_cand)
            if dist < min_distancia:
                min_distancia, estacion_mas_cercana = dist, candidata

        ruta_greedy.append(estacion_mas_cercana)
        estaciones_pendientes.remove(estacion_mas_cercana)
        estacion_actual = estacion_mas_cercana

    # Evaluación estandarizada
    res = evaluar_ruta(ruta_greedy, caso_bicis, caso_capacidad, coordenadas)
    kms, entropia = res['kms_recorridos'], res['entropia']
    fobj = funcion_objetivo(kms=kms, entropia=entropia, **kwargs)

    # El Greedy tiene un historial plano de 1 solo punto (solo se evalúa 1 vez)
    return {
        'ruta': ruta_greedy, 'fobj': fobj, 'kms': kms, 'entropia': entropia,
        'historial': {'iteracion': [0], 'fobj': [fobj], 'kms': [kms], 'entropia': [entropia]},
        'evaluaciones': 1, 'semilla': "N/A"
    }

# ----------------------------------
# 2. Algoritmo de Búsqueda Aleatoria
# ----------------------------------

def busqueda_aleatoria(
    funcion_objetivo,
    estaciones_base,
    coordenadas,
    caso_bicis,
    caso_capacidad,
    evaluar_ruta,
    semilla,
    max_iter=100,
    **kwargs
    ):
    """
    Algoritmo de Búsqueda Aleatoria.
    Genera max_iter soluciones aleatorias y se queda con la mejor.
    """
    random.seed(semilla)
    mejor_ruta, mejor_fobj, mejor_kms, mejor_entropia = None, float('inf'), 0, 0
    historial = {'iteracion': [], 'fobj': [], 'kms': [], 'entropia': []}
    evaluaciones = 0
    
    # Bucle principal de generación y evaluación de soluciones aleatorias
    for i in range(max_iter):
        ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
        res = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
        evaluaciones += 1 # Se conta la llamada
        
        kms, entropia = res['kms_recorridos'], res['entropia']
        fobj = funcion_objetivo(kms=kms, entropia=entropia, **kwargs)
        
        if fobj < mejor_fobj:
            mejor_fobj, mejor_ruta, mejor_kms, mejor_entropia = fobj, ruta_actual, kms, entropia
            historial['iteracion'].append(i)
            historial['fobj'].append(fobj)
            historial['kms'].append(kms)
            historial['entropia'].append(entropia)
            
    return {
        'ruta': mejor_ruta, 'fobj': mejor_fobj, 'kms': mejor_kms, 'entropia': mejor_entropia,
        'historial': historial, 'evaluaciones': evaluaciones, 'semilla': semilla
    }
    
# ---------------------------------------------
# 3. Algoritmo de Búsqueda Local (Mejor Vecino)
# ---------------------------------------------

def busqueda_local_mejor_vecino(
    funcion_objetivo,
    estaciones_base,
    coordenadas,
    caso_bicis,
    caso_capacidad,
    evaluar_ruta,
    semilla,
    max_evals=3000,
    **kwargs
    ):
    """
    Algoritmo de Búsqueda Local (Mejor Vecino).
    Explora todo el entorno intercambiando posiciones (dejando la 1ª fija) y se mueve al mejor.
    Se detiene en un óptimo local o a las 3000 evaluaciones.
    """
    random.seed(semilla)
    evaluaciones = 0
    
    # Generar solución inicial aleatoria
    ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
    res = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
    evaluaciones += 1
    
    kms_actual = res['kms_recorridos']
    entropia_actual = res['entropia']
    fobj_actual = funcion_objetivo(kms=kms_actual, entropia=entropia_actual, **kwargs)
    
    # Historial
    n_iteracion = 0
    historial = {
        'iteracion': [n_iteracion],
        'fobj': [fobj_actual],
        'kms': [kms_actual],
        'entropia': [entropia_actual]
    }
    
    n = len(ruta_actual)
    
    while evaluaciones < max_evals:
        mejor_vecino_ruta = None
        mejor_vecino_fobj = float('inf')
        mejor_vecino_kms = 0
        mejor_vecino_entropia = 0
        
        # Explorar todo el entorno
        # Bucle i va desde 1 hasta n-2
        # Bucle j va desde i+1 hasta n-1
        for i in range(1, n-1):
            for j in range(i+1, n):
                if evaluaciones >= max_evals:
                    break # Salida de emergencia por evaluaciones
                
                # Generar y evaluar el vecino
                vecino = generar_vecino_swap(ruta_actual, i, j)
                res_vecino = evaluar_ruta(vecino, caso_bicis, caso_capacidad, coordenadas)
                evaluaciones += 1
                
                kms_vec = res_vecino['kms_recorridos']
                ent_vec = res_vecino['entropia']
                fobj_vec = kms_vec / ent_vec if ent_vec > 0 else float('inf')
                
                # Quedarse con el mejor vecino del entorno
                if fobj_vec < mejor_vecino_fobj:
                    mejor_vecino_ruta = vecino
                    mejor_vecino_fobj = fobj_vec
                    mejor_vecino_kms = kms_vec
                    mejor_vecino_entropia = ent_vec

        if evaluaciones >= max_evals:
            break # Salida de emergencia por evaluaciones
        
        # Decisión de movimiento
        # Si el mejor vecino es mejor que la solución actual, moverse a él
        if mejor_vecino_fobj < fobj_actual:
            ruta_actual = mejor_vecino_ruta
            fobj_actual = mejor_vecino_fobj
            kms_actual = mejor_vecino_kms
            entropia_actual = mejor_vecino_entropia
            n_iteracion += 1
            historial['iteracion'].append(n_iteracion)
            historial['fobj'].append(fobj_actual)
            historial['kms'].append(kms_actual)
            historial['entropia'].append(entropia_actual)
        else:
            break # No se mejora, se ha alcanzado un óptimo local
    
    return {
        'ruta': ruta_actual, 'fobj': fobj_actual, 'kms': kms_actual, 'entropia': entropia_actual,
        'historial': historial, 'evaluaciones': evaluaciones, 'semilla': semilla
    }
             
# ---------------------------------------------
# 4. Algoritmo de Búsqueda Local (Primer Mejor)
# ---------------------------------------------

def busqueda_local_primer_mejor(
    funcion_objetivo,
    estaciones_base,
    coordenadas,
    caso_bicis,
    caso_capacidad,
    evaluar_ruta,
    semilla,
    max_evals=3000,
    **kwargs
    ):
    """
    Algoritmo de Búsqueda Local (Primer Mejor - 3 mejoras).
    Explora el entorno en orden aleatorio y acepta el primer vecino que mejora
    (después de haber encontrado hasta 3 soluciones mejores en el entorno actual).
    """
    random.seed(semilla)
    evaluaciones = 0
    
    # Generar solución inicial aleatoria
    ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
    res = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
    evaluaciones += 1
    
    kms_actual = res['kms_recorridos']
    entropia_actual = res['entropia']
    fobj_actual = funcion_objetivo(kms=kms_actual, entropia=entropia_actual, **kwargs)
    
    n_iteracion = 0
    historial = {
        'iteracion': [n_iteracion],
        'fobj': [fobj_actual],
        'kms': [kms_actual],
        'entropia': [entropia_actual]
    }
    
    n = len(ruta_actual)
    
    while evaluaciones < max_evals:
        # Generar todas las combinaciones posibles de (i, j) para el entorno,
        # dejando fijo el índice 0
        movimientos_posibles = [(i, j) for i in range(1, n - 1) for j in range(i + 1, n)]
        # Reordenación aleatoria del entorno antes de explorar
        random.shuffle(movimientos_posibles)
        
        mejores_vecinos_encontrados = []
        
        # Exploración del entorno
        for i, j in movimientos_posibles:
            if evaluaciones >= max_evals:
                break
                
            vecino = generar_vecino_swap(ruta_actual, i, j)
            res_vecino = evaluar_ruta(vecino, caso_bicis, caso_capacidad, coordenadas)
            evaluaciones += 1
            
            kms_vec = res_vecino['kms_recorridos']
            ent_vec = res_vecino['entropia']
            fobj_vec = funcion_objetivo(kms=kms_vec, entropia=ent_vec, **kwargs)
            
            # Si mejora la solución actual, se guarda como candidato a movimiento (hasta 3)
            if fobj_vec < fobj_actual:
                mejores_vecinos_encontrados.append({
                    'ruta': vecino, 'fobj': fobj_vec, 'kms': kms_vec, 'entropia': ent_vec
                })
                
                # Condición de salida: se han encontrado 3 vecinos mejores
                if len(mejores_vecinos_encontrados) >= 3:
                    break
                    
        # 3. Decisión de movimiento
        if mejores_vecinos_encontrados:
            # Moverse al Mejor de los (hasta) 3 vecinos que se han encontrado
            mejor_de_los_encontrados = min(mejores_vecinos_encontrados, key=lambda x: x['fobj'])
            
            ruta_actual = mejor_de_los_encontrados['ruta']
            fobj_actual = mejor_de_los_encontrados['fobj']
            kms_actual = mejor_de_los_encontrados['kms']
            entropia_actual = mejor_de_los_encontrados['entropia']
            
            n_iteracion += 1
            historial['iteracion'].append(n_iteracion)
            historial['fobj'].append(fobj_actual)
            historial['kms'].append(kms_actual)
            historial['entropia'].append(entropia_actual)
        else:
            break # No se mejora, se ha alcanzado un óptimo local
            
    return {
        'ruta': ruta_actual, 'fobj': fobj_actual, 'kms': kms_actual, 'entropia': entropia_actual,
        'historial': historial, 'evaluaciones': evaluaciones, 'semilla': semilla
    }
    
# -------------------------------------
# 5. Algoritmo de Enfriamiento Simulado
# -------------------------------------

def busqueda_enfriamiento_simulado(
    funcion_objetivo,
    estaciones_base,
    coordenadas,
    caso_bicis,
    caso_capacidad,
    evaluar_ruta,
    semilla,
    max_iteraciones=80,
    max_vecinos=20,
    mu=0.15,
    phi=0.20,
    **kwargs
    ):
    """
    Algoritmo de Enfriamiento Simulado (Simulated Annealing).
    Utiliza un esquema de enfriamiento de Cauchy y permite movimientos a peores
    soluciones probabilísticamente para escapar de óptimos locales.
    """
    random.seed(semilla)
    evaluaciones = 0
    
    # Generar solución inicial aleatoria
    ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
    res = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
    evaluaciones += 1
    
    kms_actual = res['kms_recorridos']
    entropia_actual = res['entropia']
    fobj_actual = funcion_objetivo(kms=kms_actual, entropia=entropia_actual, **kwargs)
    
    # Calcular Temperatura inicial (T0)
    # T0 = (mu / -log(phi)) * C(Si)
    # Se usa abs() para evitar problemas con log(0) o log de números negativos
    T0 = (mu / -math.log(phi)) * abs(fobj_actual) if fobj_actual != float('inf') else 100.0
    T = T0
    
    # Variables para rastrear la mejor solución encontrada
    mejor_ruta = list(ruta_actual)
    mejor_fobj = fobj_actual
    mejor_kms = kms_actual
    mejor_entropia = entropia_actual
    
    # Historial
    n_cambios = 0
    historial = {
        'iteracion': [n_cambios],
        'fobj': [fobj_actual],
        'kms': [kms_actual],
        'entropia': [entropia_actual]
    }
    
    n = len(ruta_actual)
    
    # Bucle Externo: Condición de parada (Enfriamientos)
    for k in range(max_iteraciones):
        
        # Bucle Interno: Condición L(T) (Generar y evaluar vecinos)
        for v in range(max_vecinos):
            i = random.randint(1, n - 2)
            j = random.randint(i + 1, n - 1)
            
            vecino = generar_vecino_swap(ruta_actual, i, j)
            res_vecino = evaluar_ruta(vecino, caso_bicis, caso_capacidad, coordenadas)
            evaluaciones += 1
            
            kms_vec = res_vecino['kms_recorridos']
            ent_vec = res_vecino['entropia']
            fobj_vec = funcion_objetivo(kms=kms_vec, entropia=ent_vec, **kwargs)
            
            # Criterio de Aceptación
            delta = fobj_vec - fobj_actual
            aceptado = False
            
            if delta < 0:
                aceptado = True # Mejor solución, se acepta siempre
            else:
                # Empeora: se acepta con una probabilidad de exp(-delta / T)
                if T > 0:
                    prob_aceptacion = math.exp(-delta / T)
                    if random.random() < prob_aceptacion:
                        aceptado = True
                        
            # Si el movimento es aceptado, se actualiza la solución actual
            if aceptado:
                ruta_actual = vecino
                fobj_actual = fobj_vec
                kms_actual = kms_vec
                entropia_actual = ent_vec
                
                # Guardar en el historial la ruta actual
                n_cambios += 1
                historial['iteracion'].append(n_cambios)
                historial['fobj'].append(fobj_actual)
                historial['kms'].append(kms_actual)
                historial['entropia'].append(entropia_actual)
                
                # Actualizar la mejor solución encontrada
                if fobj_actual < mejor_fobj:
                    mejor_fobj = fobj_actual
                    mejor_ruta = list(ruta_actual)
                    mejor_kms = kms_actual
                    mejor_entropia = entropia_actual
                
        # Enfriamiento de la temperatura (Esquema de Cauchy)
        T = T0 / (1 + (k + 1)) # Evitar división por cero en la primera iteración
        
    return {
        'ruta': ruta_actual, 'fobj': fobj_actual, 'kms': kms_actual, 'entropia': entropia_actual,
        'historial': historial, 'evaluaciones': evaluaciones, 'semilla': semilla
    }