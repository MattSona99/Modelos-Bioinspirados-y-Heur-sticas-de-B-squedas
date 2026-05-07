import sys
import os
import random

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

from Practica_1.utils import generar_vecino_swap, generar_solucion_inicial_aleatoria
from Practica_2.utils import construir_solucion_grasp, aplicar_busqueda_local_primer_mejor, mutacion_fuerte_sublista


# ---------------------------------------------
# 1. Algoritmo GRASP
# ---------------------------------------------
def busqueda_grasp(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad, evaluar_ruta, semilla,
    max_iter_grasp=10, tamano_rcl=3, **kwargs
    ):
    """
    Implementación del algoritmo GRASP (Greedy Randomized Adaptive Search Procedure).
    Genera iterativamente soluciones construidas probabilísticamente y las optimiza
    mediante el motor de Búsqueda Local del Primer Mejor.

    El parámetro tamano_rcl define la longitud de la Lista Restringida de Candidatos.
    """
    random.seed(semilla)
    evaluaciones_totales = 0

    historial = {'evaluaciones': [], 'fobj_actual': [], 'fobj': [], 'kms': [], 'entropia': []}
    record_absoluto = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}

    for iter_grasp in range(1, max_iter_grasp + 1):
        # Fase constructiva probabilística
        ruta_actual = construir_solucion_grasp(estaciones_base, coordenadas, caso_bicis, caso_capacidad,
                                               tamano_rcl=tamano_rcl)
        res_actual = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
        evaluaciones_totales += 1
        
        fobj_actual = funcion_objetivo(kms=res_actual['kms_recorridos'], entropia=res_actual['entropia'], **kwargs)
        
        # Inicialización del registro óptimo global
        if fobj_actual < record_absoluto['fobj']:
            record_absoluto.update({
                'ruta': list(ruta_actual),
                'fobj': fobj_actual,
                'kms': res_actual['kms_recorridos'],
                'entropia': res_actual['entropia']
            })

        # Registro de la trayectoria de exploración (Caja Blanca)
        historial['evaluaciones'].append(evaluaciones_totales)
        historial['fobj_actual'].append(fobj_actual)
        historial['fobj'].append(record_absoluto['fobj'])
        historial['kms'].append(record_absoluto['kms'])
        historial['entropia'].append(record_absoluto['entropia'])
        
        # Fase de explotación mediante Búsqueda Local
        _, _, _, evaluaciones_totales = aplicar_busqueda_local_primer_mejor(
            ruta_actual, fobj_actual, res_actual,
            evaluar_ruta, funcion_objetivo, caso_bicis, caso_capacidad, coordenadas,
            evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
        )
    
    return {
        'ruta': record_absoluto['ruta'], 'fobj': record_absoluto['fobj'],
        'kms': record_absoluto['kms'], 'entropia': record_absoluto['entropia'],
        'historial': historial, 'evaluaciones': evaluaciones_totales, 'semilla': semilla
    }


# ---------------------------------------------
# 2. Algoritmo ILS (Iterated Local Search)
# ---------------------------------------------
def busqueda_ils(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad, evaluar_ruta, semilla,
    max_iter_ils=10, **kwargs
    ):
    """
    Implementación del algoritmo ILS. Inicia con una solución aleatoria optimizada
    y ejecuta ciclos de mutación fuerte (Criterio del Mejor) seguidos de una nueva optimización local.
    """
    random.seed(semilla)
    evaluaciones_totales = 0

    historial = {'evaluaciones': [], 'fobj_actual': [], 'fobj': [], 'kms': [], 'entropia': [],
                 'evals_por_bl': []}
    record_absoluto = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}

    # Inicialización aleatoria y primera evaluación
    ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
    res_actual = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
    evaluaciones_totales += 1

    fobj_actual = funcion_objetivo(kms=res_actual['kms_recorridos'], entropia=res_actual['entropia'], **kwargs)

    record_absoluto.update({
        'ruta': list(ruta_actual),
        'fobj': fobj_actual,
        'kms': res_actual['kms_recorridos'],
        'entropia': res_actual['entropia']
    })

    historial['evaluaciones'].append(evaluaciones_totales)
    historial['fobj_actual'].append(fobj_actual)
    historial['fobj'].append(record_absoluto['fobj'])
    historial['kms'].append(record_absoluto['kms'])
    historial['entropia'].append(record_absoluto['entropia'])

    # Optimización inicial
    evals_pre_bl = evaluaciones_totales
    _, _, _, evaluaciones_totales = aplicar_busqueda_local_primer_mejor(
        ruta_actual, fobj_actual, res_actual,
        evaluar_ruta, funcion_objetivo, caso_bicis, caso_capacidad, coordenadas,
        evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
    )
    historial['evals_por_bl'].append(evaluaciones_totales - evals_pre_bl)

    # Ciclo iterativo de perturbación e intensificación
    for iter_ils in range(max_iter_ils):
        
        # Aplicación de perturbación sobre la mejor solución registrada
        ruta_mutada = mutacion_fuerte_sublista(record_absoluto['ruta'], tamaño=4)

        res_mutada = evaluar_ruta(ruta_mutada, caso_bicis, caso_capacidad, coordenadas)
        evaluaciones_totales += 1
        fobj_mutada = funcion_objetivo(kms=res_mutada['kms_recorridos'], entropia=res_mutada['entropia'], **kwargs)

        historial['evaluaciones'].append(evaluaciones_totales)
        historial['fobj_actual'].append(fobj_mutada)
        historial['fobj'].append(record_absoluto['fobj'])
        historial['kms'].append(record_absoluto['kms'])
        historial['entropia'].append(record_absoluto['entropia'])

        # Búsqueda Local sobre la solución perturbada
        evals_pre_bl = evaluaciones_totales
        _, _, _, evaluaciones_totales = aplicar_busqueda_local_primer_mejor(
            ruta_mutada, fobj_mutada, res_mutada,
            evaluar_ruta, funcion_objetivo, caso_bicis, caso_capacidad, coordenadas,
            evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
        )
        historial['evals_por_bl'].append(evaluaciones_totales - evals_pre_bl)

    return {
        'ruta': record_absoluto['ruta'], 'fobj': record_absoluto['fobj'],
        'kms': record_absoluto['kms'], 'entropia': record_absoluto['entropia'],
        'historial': historial, 'evaluaciones': evaluaciones_totales, 'semilla': semilla
    }


# ---------------------------------------------
# 3. Algoritmo VNS (Variable Neighborhood Search)
# ---------------------------------------------
def busqueda_vns(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad, evaluar_ruta, semilla,
    bl_max=10, k_max=4, **kwargs
    ):
    """
    Implementación del algoritmo VNS. Escala sistemáticamente la estructura 
    de vecindario ante el estancamiento y reinicia el tamaño al obtener mejoras.
    """
    random.seed(semilla)
    evaluaciones_totales = 0

    historial = {'evaluaciones': [], 'fobj_actual': [], 'fobj': [], 'kms': [], 'entropia': [],
                 'mejoras_por_k': {kk: 0 for kk in range(1, k_max + 1)},
                 'intentos_por_k': {kk: 0 for kk in range(1, k_max + 1)}}
    record_absoluto = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}
    
    # Generación y evaluación de solución inicial aleatoria
    ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
    res_actual = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
    evaluaciones_totales += 1
    
    fobj_actual = funcion_objetivo(kms=res_actual['kms_recorridos'], entropia=res_actual['entropia'], **kwargs)
    
    record_absoluto.update({
        'ruta': list(ruta_actual),
        'fobj': fobj_actual,
        'kms': res_actual['kms_recorridos'],
        'entropia': res_actual['entropia']
    })
    
    k = 1
    bl = 0
    n = len(ruta_actual)

    historial['evaluaciones'].append(evaluaciones_totales)
    historial['fobj_actual'].append(fobj_actual)
    historial['fobj'].append(record_absoluto['fobj'])
    historial['kms'].append(record_absoluto['kms'])
    historial['entropia'].append(record_absoluto['entropia'])

    while bl < bl_max:
        
        # Control del límite de vecindarios
        if k > k_max:
            k = 1
            
        # Determinación del tamaño de sublista (s) en función de k
        if k == 1: s = max(2, n // 6)
        elif k == 2: s = max(2, n // 5)
        elif k == 3: s = max(2, n // 4)
        else: s = max(2, n // 3)
        
        # Fase de Shaking (Perturbación cíclica)
        ruta_vecina = mutacion_fuerte_sublista(ruta_actual, tamaño=s)
        res_vecina = evaluar_ruta(ruta_vecina, caso_bicis, caso_capacidad, coordenadas)
        evaluaciones_totales += 1
        fobj_vecina = funcion_objetivo(kms=res_vecina['kms_recorridos'], entropia=res_vecina['entropia'], **kwargs)

        historial['evaluaciones'].append(evaluaciones_totales)
        historial['fobj_actual'].append(fobj_vecina)
        historial['fobj'].append(record_absoluto['fobj'])
        historial['kms'].append(record_absoluto['kms'])
        historial['entropia'].append(record_absoluto['entropia'])

        # Fase de Intensificación
        ruta_optimizada, fobj_optimizado, res_optimizado, evaluaciones_totales = aplicar_busqueda_local_primer_mejor(
            ruta_vecina, fobj_vecina, res_vecina,
            evaluar_ruta, funcion_objetivo, caso_bicis, caso_capacidad, coordenadas,
            evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
        )
        
        bl += 1
        historial['intentos_por_k'][k] += 1

        # Criterio de aceptación y ajuste de vecindario
        if fobj_optimizado < fobj_actual:
            historial['mejoras_por_k'][k] += 1
            ruta_actual = list(ruta_optimizada)
            fobj_actual = fobj_optimizado
            res_actual = res_optimizado.copy()
            k = 1
        else:
            k += 1

    return {
        'ruta': record_absoluto['ruta'], 
        'fobj': record_absoluto['fobj'],
        'kms': record_absoluto['kms'], 
        'entropia': record_absoluto['entropia'],
        'historial': historial, 
        'evaluaciones': evaluaciones_totales, 
        'semilla': semilla
    }