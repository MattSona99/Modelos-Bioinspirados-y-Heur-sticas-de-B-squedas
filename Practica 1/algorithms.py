from utils import calibrar_mu_phi, distancia_manhattan_km, fobj_ratio, generar_solucion_inicial_aleatoria, generar_vecino_swap, generar_greedy_probabilistico
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
                fobj_vec = funcion_objetivo(kms=kms_vec, entropia=ent_vec, **kwargs)
                
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

CACHE_CALIBRACION = {}

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
    **kwargs
    ):
    """
    Algoritmo de Enfriamiento Simulado.
    Utiliza un esquema de enfriamiento de Cauchy y permite movimientos a peores
    soluciones probabilísticamente para escapar de óptimos locales.
    """
    # Ejecutar la calibración de mu y phi óptimos
    res_greedy = kwargs.get('res_greedy')
    casos = kwargs.get('casos')
    nombre_caso = 'Caso 1'
    
    if nombre_caso not in CACHE_CALIBRACION:
        mejor_mu, mejor_phi = calibrar_mu_phi(
            ruta_base=res_greedy[nombre_caso]['ruta'],
            fobj_base=res_greedy[nombre_caso]['fobj'],
            coordenadas=coordenadas,
            caso_bicis=casos[nombre_caso]['bicis'],
            caso_capacidad=casos[nombre_caso]['capacidad'],
            evaluar_ruta=evaluar_ruta,
            funcion_objetivo=fobj_ratio,
            num_vecinos=100
        )
        CACHE_CALIBRACION[nombre_caso] = (mejor_mu, mejor_phi)
    else:
        mejor_mu, mejor_phi = CACHE_CALIBRACION[nombre_caso]
    
    # Entorno dinámico basado en max_vecinos
    entorno = [max_vecinos -5, max_vecinos, max_vecinos + 5]
    
    mejor_absoluto_ruta = None
    mejor_absoluto_fobj = float('inf')
    mejor_absoluto_kms = 0
    mejor_absoluto_entropia = 0
    mejor_absoluto_historial = None
    mejor_absoluto_entorno = max_vecinos
    evaluaciones_totales = 0
    
    # Bucle principal de iteraciones para L(T)
    for vec in entorno:
        random.seed(semilla)
        evaluaciones_run = 0
        
        # Generar solución inicial aleatoria
        ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
        res = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
        evaluaciones_run += 1
        
        kms_actual = res['kms_recorridos']
        entropia_actual = res['entropia']
        fobj_actual = funcion_objetivo(kms=kms_actual, entropia=entropia_actual, **kwargs)
        
        T0 = (mejor_mu / -math.log(mejor_phi)) * fobj_actual if fobj_actual != float('inf') else 100.0
        T = T0
        
        # Variables para el historial de esta ejecución
        mejor_run_ruta = list(ruta_actual)
        mejor_run_fobj = fobj_actual
        mejor_run_kms = kms_actual
        mejor_run_entropia = entropia_actual
        
        n_cambios = 0
        historial = {
            'iteracion': [n_cambios], 'fobj': [fobj_actual],
            'kms': [kms_actual], 'entropia': [entropia_actual]
        }
        
        n = len(ruta_actual)
        
        # Bucle Externo: Condición de parada
        for k in range(max_iteraciones):
            
            # Bucle Interno: Condición L(T) dinámico
            for v in range(vec):
                i = random.randint(1, n - 2)
                j = random.randint(i + 1, n - 1)
                
                vecino = generar_vecino_swap(ruta_actual, i, j)
                res_vecino = evaluar_ruta(vecino, caso_bicis, caso_capacidad, coordenadas)
                evaluaciones_run += 1
                
                kms_vec = res_vecino['kms_recorridos']
                ent_vec = res_vecino['entropia']
                fobj_vec = funcion_objetivo(kms=kms_vec, entropia=ent_vec, **kwargs)
                
                delta = fobj_vec - fobj_actual
                aceptado = False
                
                if delta < 0:
                    aceptado = True
                else:
                    if T > 0:
                        prob_aceptacion = math.exp(-delta / T)
                        if random.random() < prob_aceptacion:
                            aceptado = True
                            
                if aceptado:
                    ruta_actual = vecino
                    fobj_actual = fobj_vec
                    kms_actual = kms_vec
                    entropia_actual = ent_vec
                    
                    n_cambios += 1
                    historial['iteracion'].append(n_cambios)
                    historial['fobj'].append(fobj_actual)
                    historial['kms'].append(kms_actual)
                    historial['entropia'].append(entropia_actual)
                    
                    if fobj_actual < mejor_run_fobj:
                        mejor_run_fobj = fobj_actual
                        mejor_run_ruta = list(ruta_actual)
                        mejor_run_kms = kms_actual
                        mejor_run_entropia = entropia_actual
            
            T = T0 / (1 + (k + 1))
            
        evaluaciones_totales += evaluaciones_run
        
        # Evaluar si esta configuración de L(T) batió el récord absoluto
        if mejor_run_fobj < mejor_absoluto_fobj:
            mejor_absoluto_fobj = mejor_run_fobj
            mejor_absoluto_ruta = mejor_run_ruta
            mejor_absoluto_kms = mejor_run_kms
            mejor_absoluto_entropia = mejor_run_entropia
            mejor_absoluto_historial = historial
            mejor_absoluto_entorno = vec
            
    # Formateo de parámetros personalizados para el log
    info_parametros = f"Mejor L(T) = {mejor_absoluto_entorno} | μ = {mejor_mu} | φ = {mejor_phi}"
            
    return {
        'ruta': mejor_absoluto_ruta, 
        'fobj': mejor_absoluto_fobj, 
        'kms': mejor_absoluto_kms, 
        'entropia': mejor_absoluto_entropia,
        'historial': mejor_absoluto_historial, 
        'evaluaciones': evaluaciones_totales, 
        'semilla': semilla,
        'parametros_extra': info_parametros
    }

# -----------------------------
# 6. Algoritmo de Búsqueda Tabú
# -----------------------------

def busqueda_tabu(
    funcion_objetivo,
    estaciones_base,
    coordenadas,
    caso_bicis,
    caso_capacidad,
    evaluar_ruta,
    semilla,
    max_iteraciones=150,
    vecinos_por_iteracion=20,
    **kwargs
    ):
    """
    Algoritmo de Búsqueda Tabú.
    Incorpora memoria a corto plazo (Lista Tabú) y a largo plazo (Matriz de Frecuencias),
    junto con estrategias de reinicialización probabilística.
    """
    random.seed(semilla)
    evaluaciones = 0
    
    # Memoria a Largo Plazo: Matriz Frecuencia[N][N]
    N = len(coordenadas)
    frecuencia = [[0] * N for _ in range(N)]
    
    # Solución Inicial
    ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
    res = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
    evaluaciones += 1
    
    fobj_actual = funcion_objetivo(kms=res['kms_recorridos'], entropia=res['entropia'], **kwargs)
    
    mejor_absoluto_ruta = list(ruta_actual)
    mejor_absoluto_fobj = fobj_actual
    mejor_absoluto_kms = res['kms_recorridos']
    mejor_absoluto_entropia = res['entropia']
    
    # Memoria a Corto Plazo: Lista Tabú
    lista_tabu = []
    tamano_tabu = 4
    
    n_cambios = 0
    historial = {
        'iteracion': [n_cambios], 'fobj': [fobj_actual],
        'kms': [mejor_absoluto_kms], 'entropia': [mejor_absoluto_entropia]
    }
    
    n_estaciones = len(ruta_actual)
    
    for iteracion in range(1, max_iteraciones +1):
        mejores_candidatos = []
        
        # Estrategia de selección: examinar 20 vecinos
        for _ in range(vecinos_por_iteracion):
            i = random.randint(1, n_estaciones - 2)
            j = random.randint(i + 1, n_estaciones - 1)
            
            vecino = generar_vecino_swap(ruta_actual, i, j)
            res_vecino = evaluar_ruta(vecino, caso_bicis, caso_capacidad, coordenadas)
            evaluaciones += 1
            
            kms_vec = res_vecino['kms_recorridos']
            ent_vec = res_vecino['entropia']
            fobj_vec = funcion_objetivo(kms=kms_vec, entropia=ent_vec, **kwargs)
            
            mejores_candidatos.append({
                'ruta': vecino,
                'fobj': fobj_vec,
                'kms': kms_vec,
                'entropia': ent_vec,
                'movimiento': (i, j)
            })
            
        # Ordenar candidatos de mejor a peor
        mejores_candidatos.sort(key=lambda x: x['fobj'])
        
        movimiento_aceptado = None
        
        # Buscar el primer candidato no tabú o que cumpla el criterio de aspiración
        for candidato in mejores_candidatos:
            mov = candidato['movimiento']
            
            # Es tabú si alguno de los indices intercambiados está bloqueado
            es_tabu = any(mov[0] in tabu_mov or mov[1] in tabu_mov for tabu_mov in lista_tabu)
            
            # Criterio de Aspiración: mejora el récord absoluto
            criterio_aspiracion = candidato['fobj'] < mejor_absoluto_fobj
            
            if not es_tabu or criterio_aspiracion:
                movimiento_aceptado = candidato
                break
        
        # Si no se ha aceptado ninguno de los candidatos, se acepta el primero
        if not movimiento_aceptado:
            movimiento_aceptado = mejores_candidatos[0]
            
        ruta_actual = movimiento_aceptado['ruta']
        fobj_actual = movimiento_aceptado['fobj']
        
        # Matriz de Frecuencias
        ruta_completa = [0] + ruta_actual + [0]
        for idx in range(len(ruta_completa) - 1):
            origen = ruta_completa[idx]
            destino = ruta_completa[idx + 1]
            frecuencia[origen][destino] += 1
            
        # Memoria a Corto Plazo: Lista Tabú
        lista_tabu.append(movimiento_aceptado['movimiento'])
        if len(lista_tabu) > tamano_tabu:
            lista_tabu.pop(0) # FIFO
            
        # Récord Global
        if fobj_actual < mejor_absoluto_fobj:
            mejor_absoluto_fobj = fobj_actual
            mejor_absoluto_ruta = list(ruta_actual)
            mejor_absoluto_kms = movimiento_aceptado['kms']
            mejor_absoluto_entropia = movimiento_aceptado['entropia']
            
        n_cambios += 1
        historial['iteracion'].append(n_cambios)
        historial['fobj'].append(fobj_actual) # En tabú se suele graficar la actual, que sube y baja
        historial['kms'].append(movimiento_aceptado['kms'])
        historial['entropia'].append(movimiento_aceptado['entropia'])
        
        # Reinicializzaciones: cada max_iteraciones / 4
        if iteracion % (max_iteraciones // 4) == 0 and iteracion < max_iteraciones:
            prob = random.random()
            
            if prob < 0.25:
                # 25% Aleatoria
                ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
            elif prob < 0.75:
                # 50% Greedy Probabilistico basado en Memoria a Largo Plazo
                ruta_actual = generar_greedy_probabilistico(frecuencia, estaciones_base)
            else:
                # 25% Explotación (Mejor Solución Obtenida)
                ruta_actual = list(mejor_absoluto_ruta)
                
            # Evaluar la nueva ruta base
            res_reinicializacion = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
            evaluaciones += 1
            fobj_actual = funcion_objetivo(kms=res_reinicializacion['kms_recorridos'], entropia=res_reinicializacion['entropia'], **kwargs)
            
            # Vaciar lista tabú al reiniciar
            lista_tabu = []
            
            # Alterar tamaño de la lista tabú (+50% o -50%)
            if random.random() < 0.5:
                tamano_tabu = max(1, int(tamano_tabu * 1.5))
            else:
                tamano_tabu = max(1, int(tamano_tabu * 0.5))
    
    return {
        'ruta': mejor_absoluto_ruta, 
        'fobj': mejor_absoluto_fobj, 
        'kms': mejor_absoluto_kms, 
        'entropia': mejor_absoluto_entropia,
        'historial': historial, 
        'evaluaciones': evaluaciones, 
        'n_cambios': n_cambios,
        'semilla': semilla
    }