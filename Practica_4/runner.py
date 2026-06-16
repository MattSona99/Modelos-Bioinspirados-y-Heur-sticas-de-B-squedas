import sys
import os

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

import numpy as np
import time
from IPython.display import display

from Practica_1.utils import (
    cargar_coordenadas, evaluar_ruta, fobj_ratio, dibujar_mapa_trayecto,
)
from Practica_4.config import CASOS, SEMILLAS_P4
from Practica_4.algorithms import (
    sistema_hormigas, sistema_hormigas_elitista, sistema_colonias_hormigas,
    greedy_aleatorizado,
)
from Practica_4.utils import (
    generar_tabla_global_aco, generar_tabla_parcial_aco,
    graficar_convergencia_aco, evaluar_tour_sin_capacidad,
)

REGISTRY = {
    'sh': {'nombre_display': 'SH (Sistema de Hormigas)',
           'funcion': sistema_hormigas},
    'she': {'nombre_display': 'SHE (Sistema de Hormigas Elitista)',
            'funcion': sistema_hormigas_elitista},
    'sch': {'nombre_display': 'SCH (Sistema de Colonias de Hormigas)',
            'funcion': sistema_colonias_hormigas},
}


def _cargar_coords():
    return cargar_coordenadas(os.path.join(abspath, 'Practica_1', 'coords.json'))


def _estaciones_tsp(coordenadas):
    """Todas las estaciones no-depósito (TSP puro de la P4)."""
    return list(range(1, len(coordenadas)))


def ejecutar_experimento_aco(id_algoritmo, casos=CASOS, semillas=SEMILLAS_P4,
                             maps=False, **kwargs):
    """
    Ejecuta un algoritmo de hormigas sobre los 3 casos × semillas. Selecciona el
    mejor por score universal (fobj_ratio = kms/entropía), imprime la tabla
    resumen por caso, grafica la convergencia y, si maps=True, dibuja el mejor tour.
    Devuelve los resultados globales para la Tabla 1.1.
    """
    if id_algoritmo not in REGISTRY:
        raise ValueError(f'Algoritmo "{id_algoritmo}" no reconocido.')

    info = REGISTRY[id_algoritmo]
    nombre_algoritmo = info['nombre_display']
    funcion_algoritmo = info['funcion']
    coordenadas = _cargar_coords()
    estaciones = _estaciones_tsp(coordenadas)

    print(f"\n{'='*100}")
    print(f" EXPERIMENTACIÓN: {nombre_algoritmo.upper()}")
    print(f"{'='*100}")

    resultados_globales = {}
    datos_para_graficar = []
    filas_tabla = []

    for nombre_caso, datos in casos.items():
        bicis, capacidad = datos['bicis'], datos['capacidad']

        mejor_res = None
        evaluaciones_lista, tiempos_lista = [], []
        for sem in semillas:
            t0 = time.perf_counter()
            res = funcion_algoritmo(
                funcion_objetivo=fobj_ratio, estaciones_base=estaciones,
                coordenadas=coordenadas, caso_bicis=bicis,
                caso_capacidad=capacidad, evaluar_ruta=evaluar_ruta,
                semilla=sem, **kwargs)
            tiempos_lista.append(time.perf_counter() - t0)
            evaluaciones_lista.append(res['evaluaciones'])
            res['score_universal'] = fobj_ratio(res['kms'], res['entropia'])
            if mejor_res is None or res['score_universal'] < mejor_res['score_universal']:
                mejor_res = res

        mejor_res['ev_media'] = float(np.mean(evaluaciones_lista))
        mejor_res['sigma_ev'] = float(np.std(evaluaciones_lista))
        resultados_globales[nombre_caso] = mejor_res

        filas_tabla.append(
            f"| {nombre_caso:<6} | {mejor_res['score_universal']:>11.4f} "
            f"| {mejor_res['kms']:>8.2f} | {mejor_res['entropia']:>8.4f} "
            f"| {mejor_res['ev_media']:>9.1f} | {mejor_res['evaluaciones']:>9} "
            f"| {np.mean(tiempos_lista):>9.3f} | {str(mejor_res['semilla']):>10} |")

        datos_para_graficar.append({
            'historial': mejor_res['historial'], 'nombre_caso': nombre_caso,
            'semilla': mejor_res['semilla']})

        if maps:
            ev = evaluar_tour_sin_capacidad(mejor_res['ruta'], bicis, capacidad,
                                            coordenadas)
            print(f"\n>> Mejor tour ({nombre_caso}) — Kms={mejor_res['kms']:.2f}")
            display(dibujar_mapa_trayecto(coordenadas, ev['movimientos_mapa']))

    encabezado = ("| Caso   | Score (Ratio)| Kms      | Entropía | Ev. Media "
                  "| Ev. Total | T.Medio(s)|  Semilla   |")
    print(encabezado)
    print("-" * len(encabezado))
    for fila in filas_tabla:
        print(fila)
    print("-" * len(encabezado))
    print(f" Parámetros: {next(iter(resultados_globales.values()))['parametros_extra']}")

    graficar_convergencia_aco(datos_para_graficar, nombre_algoritmo)
    return resultados_globales


def ejecutar_tabla_global_aco(casos=CASOS, semillas=SEMILLAS_P4, **kwargs):
    """
    Construye la Tabla 1.1 (Resultados Globales): ejecuta SH, SHE, SCH y el Greedy
    aleatorizado sobre los 3 casos (mejor de las 'semillas' por fobj_ratio) y
    renderiza la tabla comparativa con #Ev, F OBJ, Kms y Entr por caso.
    """
    coordenadas = _cargar_coords()
    estaciones = _estaciones_tsp(coordenadas)

    algoritmos = [
        ('SH', sistema_hormigas),
        ('SHE', sistema_hormigas_elitista),
        ('SCH', sistema_colonias_hormigas),
        ('Greedy A.', greedy_aleatorizado),
    ]

    diccionario = {}
    for nombre, funcion in algoritmos:
        por_caso = {}
        for nombre_caso, datos in casos.items():
            bicis, capacidad = datos['bicis'], datos['capacidad']
            mejor = None
            for sem in semillas:
                res = funcion(
                    funcion_objetivo=fobj_ratio, estaciones_base=estaciones,
                    coordenadas=coordenadas, caso_bicis=bicis,
                    caso_capacidad=capacidad, evaluar_ruta=evaluar_ruta,
                    semilla=sem, **kwargs)
                score = fobj_ratio(res['kms'], res['entropia'])
                if mejor is None or score < mejor['fobj']:
                    mejor = {'ev': res['evaluaciones'], 'fobj': score,
                             'kms': res['kms'], 'entropia': res['entropia']}
            por_caso[nombre_caso] = mejor
        diccionario[nombre] = por_caso

    generar_tabla_global_aco(diccionario)
    return diccionario


def ejecutar_tabla_parcial_aco(id_algoritmo, casos=CASOS, semillas=SEMILLAS_P4,
                               **kwargs):
    """
    Genera la Tabla 1.1 (Resultados Parciales) de un algoritmo: una fila por
    semilla (ejecución) en cada caso, más Media y Desviación típica, con
    fobj_ratio. Cubre el requisito del PDF de reportar media y desviación.
    """
    if id_algoritmo not in REGISTRY:
        raise ValueError(f'Algoritmo "{id_algoritmo}" no reconocido.')

    info = REGISTRY[id_algoritmo]
    funcion = info['funcion']
    coordenadas = _cargar_coords()
    estaciones = _estaciones_tsp(coordenadas)

    datos_por_caso = {}
    for nombre_caso, datos in casos.items():
        bicis, capacidad = datos['bicis'], datos['capacidad']
        ejecuciones = []
        for sem in semillas:
            res = funcion(
                funcion_objetivo=fobj_ratio, estaciones_base=estaciones,
                coordenadas=coordenadas, caso_bicis=bicis,
                caso_capacidad=capacidad, evaluar_ruta=evaluar_ruta,
                semilla=sem, **kwargs)
            ejecuciones.append({
                'ev': res['evaluaciones'],
                'fobj': fobj_ratio(res['kms'], res['entropia']),
                'kms': res['kms'], 'entropia': res['entropia']})
        datos_por_caso[nombre_caso] = ejecuciones

    generar_tabla_parcial_aco(info['nombre_display'], datos_por_caso)
    return datos_por_caso


if __name__ == '__main__':
    # Smoke test reducido: 1 caso, 2 semillas, pocas iteraciones.
    print(">> SMOKE TEST Práctica 4 (reducido)")
    caso_min = {'Caso 1': CASOS['Caso 1']}
    for algo in ('sh', 'she', 'sch'):
        ejecutar_experimento_aco(algo, casos=caso_min,
                                 semillas=SEMILLAS_P4[:2], n_iteraciones=30)

    print("\n>> SMOKE TEST Tabla 1.1 global (reducida)")
    ejecutar_tabla_global_aco(casos=caso_min, semillas=SEMILLAS_P4[:2],
                              n_iteraciones=30)

    print("\n>> SMOKE TEST Tabla 1.1 parcial (media/desviación, reducida)")
    ejecutar_tabla_parcial_aco('sh', casos=caso_min, semillas=SEMILLAS_P4[:2],
                               n_iteraciones=30)
