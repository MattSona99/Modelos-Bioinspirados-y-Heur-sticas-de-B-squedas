import sys
import os
import random

import numpy as np

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

from Practica_4.config import (
    M_HORMIGAS, N_ITERACIONES, ALPHA, BETA, E_ELITISTAS, RHO, PHI, Q0,
)
from Practica_4.utils import (
    construir_matriz_distancias, calcular_visibilidad, calcular_tau0,
    inicializar_feromona, construir_tour_hormiga, depositar_feromona,
    arcos_tour, coste_greedy_determinista, evaluar_tour_sin_capacidad,
    greedy_aleatorizado_lista_corta,
)


# =============================================================================
# INFRAESTRUCTURA COMÚN
# =============================================================================

def _preparar(estaciones_base, coordenadas, caso_capacidad):
    """Matriz de distancias, visibilidad, ciudades a visitar y τ0 (vía greedy)."""
    matriz = construir_matriz_distancias(coordenadas)
    visibilidad = calcular_visibilidad(matriz)
    n = len(coordenadas)
    ciudades = [c for c in estaciones_base if c != 0]
    _, coste_greedy = coste_greedy_determinista(matriz, n)
    tau0 = calcular_tau0(n, coste_greedy)
    return matriz, visibilidad, n, ciudades, tau0


def _evaluar(ruta, caso_bicis, caso_capacidad, coordenadas, funcion_objetivo,
             matriz, **kwargs):
    """Evalúa un tour: kms (matriz, barato), entropía (motor) y F. Objetivo."""
    res = evaluar_tour_sin_capacidad(ruta, caso_bicis, caso_capacidad, coordenadas)
    kms, entropia = res['kms_recorridos'], res['entropia']
    fobj = funcion_objetivo(kms=kms, entropia=entropia, **kwargs)
    return fobj, kms, entropia


def _resultado(record, historial, evaluaciones, semilla, parametros_extra=''):
    return {
        'ruta': record['ruta'], 'fobj': record['fobj'],
        'kms': record['kms'], 'entropia': record['entropia'],
        'historial': historial, 'evaluaciones': evaluaciones, 'semilla': semilla,
        'parametros_extra': parametros_extra,
    }


def _snapshot_feromona(feromona, mejor_ruta, etiqueta, tau0):
    return {'feromona': np.array(feromona, copy=True),
            'mejor_ruta': list(mejor_ruta) if mejor_ruta else None,
            'etiqueta': etiqueta, 'tau0': tau0}


# =============================================================================
# 1. SISTEMA DE HORMIGAS (SH)
# =============================================================================

def sistema_hormigas(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad,
    evaluar_ruta, semilla, m=M_HORMIGAS, n_iteraciones=N_ITERACIONES,
    alpha=ALPHA, beta=BETA, rho=RHO, registrar_feromona=False,
    intervalo_feromona=None, **kwargs
):
    """
    Sistema de Hormigas clásico (Dorigo): cada iteración, las m hormigas
    construyen un tour con la regla proporcional; tras evaporar (ρ), TODAS las
    hormigas depositan feromona 1/C(S) en sus arcos. El conteo de evaluaciones es
    m por iteración (una evaluación por tour construido).
    """
    kwargs.pop('casos', None)
    random.seed(semilla)
    rng = random.Random(semilla)

    matriz, visibilidad, n, ciudades, tau0 = _preparar(
        estaciones_base, coordenadas, caso_capacidad)
    feromona = inicializar_feromona(n, tau0)

    historial = {'iteracion': [], 'mejor_coste': [], 'coste_iteracion': [],
                 'snapshots_feromona': []}
    record = {'ruta': None, 'fobj': float('inf'), 'kms': float('inf'), 'entropia': 0}
    evaluaciones = 0

    intervalo = intervalo_feromona or max(1, n_iteraciones // 50)
    for it in range(n_iteraciones):
        tours = []
        for _ in range(m):
            ruta = construir_tour_hormiga(feromona, visibilidad, ciudades, 0,
                                          alpha, beta, rng)
            fobj, kms, entropia = _evaluar(ruta, caso_bicis, caso_capacidad,
                                           coordenadas, funcion_objetivo, matriz,
                                           **kwargs)
            evaluaciones += 1
            tours.append((ruta, fobj, kms, entropia))
            if fobj < record['fobj']:
                record.update({'ruta': list(ruta), 'fobj': fobj, 'kms': kms,
                               'entropia': entropia})

        # Evaporación + depósito de todas las hormigas (proporcional a 1/coste).
        feromona *= (1 - rho)
        for ruta, _, kms, _ in tours:
            depositar_feromona(feromona, ruta, 1.0 / kms if kms > 0 else 0.0)

        historial['iteracion'].append(it)
        historial['mejor_coste'].append(record['kms'])
        historial['coste_iteracion'].append(min(t[2] for t in tours))
        if registrar_feromona and (it % intervalo == 0 or it == n_iteraciones - 1):
            historial['snapshots_feromona'].append(
                _snapshot_feromona(feromona, record['ruta'], f"Iteración {it}",
                                   tau0))

    return _resultado(record, historial, evaluaciones, semilla,
                      parametros_extra=f"m={m}, iter={n_iteraciones}, rho={rho}")


# =============================================================================
# 2. SISTEMA DE HORMIGAS ELITISTA (SHE)
# =============================================================================

def sistema_hormigas_elitista(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad,
    evaluar_ruta, semilla, m=M_HORMIGAS, n_iteraciones=N_ITERACIONES,
    alpha=ALPHA, beta=BETA, rho=RHO, e=E_ELITISTAS,
    registrar_feromona=False, intervalo_feromona=None, **kwargs
):
    """
    Sistema de Hormigas Elitista: como SH, pero además 'e' hormigas elitistas
    refuerzan en cada iteración los arcos del MEJOR tour global con un depósito
    extra e·(1/C*). Intensifica alrededor del mejor encontrado.
    """
    kwargs.pop('casos', None)
    random.seed(semilla)
    rng = random.Random(semilla)

    matriz, visibilidad, n, ciudades, tau0 = _preparar(
        estaciones_base, coordenadas, caso_capacidad)
    feromona = inicializar_feromona(n, tau0)

    historial = {'iteracion': [], 'mejor_coste': [], 'coste_iteracion': [],
                 'snapshots_feromona': []}
    record = {'ruta': None, 'fobj': float('inf'), 'kms': float('inf'), 'entropia': 0}
    evaluaciones = 0

    intervalo = intervalo_feromona or max(1, n_iteraciones // 50)
    for it in range(n_iteraciones):
        tours = []
        for _ in range(m):
            ruta = construir_tour_hormiga(feromona, visibilidad, ciudades, 0,
                                          alpha, beta, rng)
            fobj, kms, entropia = _evaluar(ruta, caso_bicis, caso_capacidad,
                                           coordenadas, funcion_objetivo, matriz,
                                           **kwargs)
            evaluaciones += 1
            tours.append((ruta, fobj, kms, entropia))
            if fobj < record['fobj']:
                record.update({'ruta': list(ruta), 'fobj': fobj, 'kms': kms,
                               'entropia': entropia})

        feromona *= (1 - rho)
        for ruta, _, kms, _ in tours:
            depositar_feromona(feromona, ruta, 1.0 / kms if kms > 0 else 0.0)

        # Refuerzo elitista: e hormigas depositan sobre el mejor tour global.
        if record['ruta'] is not None and record['kms'] > 0:
            depositar_feromona(feromona, record['ruta'], e * (1.0 / record['kms']))

        historial['iteracion'].append(it)
        historial['mejor_coste'].append(record['kms'])
        historial['coste_iteracion'].append(min(t[2] for t in tours))
        if registrar_feromona and (it % intervalo == 0 or it == n_iteraciones - 1):
            historial['snapshots_feromona'].append(
                _snapshot_feromona(feromona, record['ruta'], f"Iteración {it}",
                                   tau0))

    return _resultado(record, historial, evaluaciones, semilla,
                      parametros_extra=f"m={m}, iter={n_iteraciones}, "
                                       f"rho={rho}, e={e}")


# =============================================================================
# 3. SISTEMA DE COLONIAS DE HORMIGAS (SCH / ACS)
# =============================================================================

def sistema_colonias_hormigas(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad,
    evaluar_ruta, semilla, m=M_HORMIGAS, n_iteraciones=N_ITERACIONES,
    alpha=ALPHA, beta=BETA, rho=RHO, phi=PHI, q0=Q0,
    registrar_feromona=False, intervalo_feromona=None, **kwargs
):
    """
    Sistema de Colonias de Hormigas (ACS):
    - Regla pseudo-aleatoria proporcional (con prob. q0 explota el mejor arco).
    - Actualización LOCAL de feromona durante la construcción: cada arco recorrido
      se acerca a τ0 con tasa φ (diversifica, evita estancamiento).
    - Actualización GLOBAL sólo por la MEJOR hormiga (global-best): evaporación ρ
      y depósito ρ·(1/C*) únicamente en los arcos del mejor tour.
    """
    kwargs.pop('casos', None)
    random.seed(semilla)
    rng = random.Random(semilla)

    matriz, visibilidad, n, ciudades, tau0 = _preparar(
        estaciones_base, coordenadas, caso_capacidad)
    feromona = inicializar_feromona(n, tau0)

    historial = {'iteracion': [], 'mejor_coste': [], 'coste_iteracion': [],
                 'snapshots_feromona': []}
    record = {'ruta': None, 'fobj': float('inf'), 'kms': float('inf'), 'entropia': 0}
    evaluaciones = 0

    intervalo = intervalo_feromona or max(1, n_iteraciones // 50)
    for it in range(n_iteraciones):
        coste_it = float('inf')
        for _ in range(m):
            ruta = construir_tour_hormiga(feromona, visibilidad, ciudades, 0,
                                          alpha, beta, rng, q0=q0, phi=phi,
                                          tau0=tau0)
            fobj, kms, entropia = _evaluar(ruta, caso_bicis, caso_capacidad,
                                           coordenadas, funcion_objetivo, matriz,
                                           **kwargs)
            evaluaciones += 1
            coste_it = min(coste_it, kms)
            if fobj < record['fobj']:
                record.update({'ruta': list(ruta), 'fobj': fobj, 'kms': kms,
                               'entropia': entropia})

        # Actualización global: sólo el mejor tour (evaporación + depósito en sus arcos).
        if record['ruta'] is not None and record['kms'] > 0:
            deposito = rho * (1.0 / record['kms'])
            for i, j in arcos_tour(record['ruta'], 0):
                feromona[i][j] = (1 - rho) * feromona[i][j] + deposito
                feromona[j][i] = feromona[i][j]

        historial['iteracion'].append(it)
        historial['mejor_coste'].append(record['kms'])
        historial['coste_iteracion'].append(coste_it)
        if registrar_feromona and (it % intervalo == 0 or it == n_iteraciones - 1):
            historial['snapshots_feromona'].append(
                _snapshot_feromona(feromona, record['ruta'], f"Iteración {it}",
                                   tau0))

    return _resultado(record, historial, evaluaciones, semilla,
                      parametros_extra=f"m={m}, iter={n_iteraciones}, "
                                       f"rho={rho}, phi={phi}, q0={q0}")


# =============================================================================
# 4. COMPARADOR: GREEDY ALEATORIZADO DE LISTA CORTA
# =============================================================================

def greedy_aleatorizado(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad,
    evaluar_ruta, semilla, tam_rcl=3, **kwargs
):
    """
    Greedy aleatorizado de lista corta (tamaño 3): construye un tour eligiendo en
    cada paso al azar entre las 3 estaciones no visitadas más cercanas. Una sola
    construcción/evaluación por ejecución (la consigna lo usa de línea base).
    """
    kwargs.pop('casos', None)
    matriz = construir_matriz_distancias(coordenadas)
    n = len(coordenadas)
    ruta = greedy_aleatorizado_lista_corta(matriz, n, semilla, tam_rcl)
    fobj, kms, entropia = _evaluar(ruta, caso_bicis, caso_capacidad, coordenadas,
                                   funcion_objetivo, matriz, **kwargs)
    historial = {'iteracion': [0], 'mejor_coste': [kms], 'snapshots_feromona': []}
    record = {'ruta': ruta, 'fobj': fobj, 'kms': kms, 'entropia': entropia}
    return _resultado(record, historial, 1, semilla,
                      parametros_extra=f"tam_rcl={tam_rcl}")
