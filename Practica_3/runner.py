import sys
import os

directorio_runner = os.path.dirname(__file__)
abspath = os.path.abspath(os.path.join(directorio_runner, '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

import numpy as np
import time
from IPython.display import display

from Practica_1.config import CASOS, SEMILLAS_P2, TOLERANCIA
from Practica_1.utils import (
    obtener_estaciones_a_visitar, cargar_coordenadas, evaluar_ruta,
    fobj_ratio, FUNCIONES_OBJETIVO, graficar_historiales,
    dibujar_mapa_estado, dibujar_mapa_trayecto,
)

from Practica_2.utils import (
    calcular_metricas_caja_negra,
    graficar_boxplot_caja_negra,
    calcular_tasa_overlap_hamming,
)

from Practica_3.algorithms import (
    algoritmo_genetico_basico,
    algoritmo_genetico_chc,
    algoritmo_genetico_multimodal,
)
from Practica_3.utils import (
    generar_tabla_parcial,
    diversificacion_estructural_arcos,
    graficar_convergencia_chc,
    graficar_fitness_chc,
    graficar_diversidad_chc,
    analizar_clearing,
    tabla_estudio_parametrico,
)

# Mejor F. Objetivo conocido (score ratio) de las Prácticas 1/2 — referencia RPD.
MEJOR_FOBJ_CONOCIDO = {'Caso 1': 1.3665, 'Caso 2': 1.4377, 'Caso 3': 2.0116}

REGISTRY = {
    'basico': {
        'nombre_display': 'AG Básico',
        'funcion': algoritmo_genetico_basico,
    },
    'chc': {
        'nombre_display': 'AG CHC',
        'funcion': algoritmo_genetico_chc,
    },
    'multimodal': {
        'nombre_display': 'AG Multimodal (Clearing)',
        'funcion': algoritmo_genetico_multimodal,
    },
}


def _cargar_coords():
    return cargar_coordenadas(os.path.join(abspath, 'Practica_1', 'coords.json'))


def ejecutar_experimento_ga(
    id_algoritmo,
    casos=CASOS, semillas=SEMILLAS_P2, tolerancia=TOLERANCIA,
    maps=False, **kwargs
):
    """
    Ejecuta un algoritmo evolutivo barriendo todas las FUNCIONES_OBJETIVO y
    todas las semillas sobre los 3 casos. Selecciona el mejor por score
    universal (fobj_ratio), imprime la tabla por caso, grafica la evolución
    (Caja Blanca) y devuelve los resultados globales para la Tabla 1.2.
    """
    if id_algoritmo not in REGISTRY:
        raise ValueError(f'Algoritmo "{id_algoritmo}" no reconocido.')

    info = REGISTRY[id_algoritmo]
    nombre_algoritmo = info['nombre_display']
    funcion_algoritmo = info['funcion']
    coordenadas = _cargar_coords()

    resultados_globales = {}
    print(f"\n{'='*108}")
    print(f" EXPERIMENTACIÓN: {nombre_algoritmo.upper()}")
    print(f"{'='*108}")

    datos_para_graficar = []
    filas_tabla = []

    for nombre_caso, datos in casos.items():
        bicis, capacidad = datos['bicis'], datos['capacidad']
        est_base = obtener_estaciones_a_visitar(bicis, capacidad, tolerancia)

        mejor_res_absoluto = None
        evaluaciones_lista = []
        tiempos_lista = []

        for f_obj in FUNCIONES_OBJETIVO:
            mejor_res_fobj = None
            for sem in semillas:
                t0 = time.perf_counter()
                res = funcion_algoritmo(
                    funcion_objetivo=f_obj, estaciones_base=est_base,
                    coordenadas=coordenadas, caso_bicis=bicis,
                    caso_capacidad=capacidad, evaluar_ruta=evaluar_ruta,
                    semilla=sem, casos=casos, **kwargs
                )
                tiempos_lista.append(time.perf_counter() - t0)
                evaluaciones_lista.append(res['evaluaciones'])

                if mejor_res_fobj is None or res['fobj'] < mejor_res_fobj['fobj']:
                    mejor_res_fobj = res

            score_universal = fobj_ratio(mejor_res_fobj['kms'],
                                         mejor_res_fobj['entropia'])
            mejor_res_fobj['score_universal'] = score_universal
            mejor_res_fobj['nombre_fobj'] = f_obj.__name__

            if (mejor_res_absoluto is None or
                    score_universal < mejor_res_absoluto['score_universal']):
                mejor_res_absoluto = mejor_res_fobj

        ev_media = np.mean(evaluaciones_lista)
        sigma_ev = np.std(evaluaciones_lista)
        t_medio = np.mean(tiempos_lista)
        mejor_res_absoluto['ev_media'] = ev_media
        mejor_res_absoluto['sigma_ev'] = sigma_ev
        resultados_globales[nombre_caso] = mejor_res_absoluto

        if maps:
            ev_final = evaluar_ruta(mejor_res_absoluto['ruta'], bicis,
                                    capacidad, coordenadas)
            print(f"\n>> Mapa TRAYECTO ({nombre_caso})")
            display(dibujar_mapa_trayecto(coordenadas,
                                          ev_final['movimientos_mapa']))
            print(f">> Mapa ESTADO ({nombre_caso})")
            display(dibujar_mapa_estado(coordenadas, capacidad,
                                        ev_final['inventario_final']))

        sem_str = str(mejor_res_absoluto['semilla'])
        fobj_corta = mejor_res_absoluto['nombre_fobj'].replace('fobj_', '')[:10]
        filas_tabla.append(
            f"| {nombre_caso:<6} | {mejor_res_absoluto['score_universal']:>11.4f} "
            f"| {mejor_res_absoluto['kms']:>7.2f} "
            f"| {mejor_res_absoluto['entropia']:>8.4f} | {ev_media:>9.1f} "
            f"| {mejor_res_absoluto['evaluaciones']:>9} | {t_medio:>10.4f} "
            f"| {sem_str:>12} | {fobj_corta:<10} |")

        historial_std = mejor_res_absoluto['historial']
        historial_std['fobj'] = [fobj_ratio(k, e) for k, e in
                                 zip(historial_std['kms'],
                                     historial_std['entropia'])]
        datos_para_graficar.append({
            'historial': historial_std,
            'nombre_caso': nombre_caso,
            'semilla': mejor_res_absoluto['semilla'],
        })

    encabezado = ("| Caso   | Score (Ratio)| Kms     | Entropía | Ev. Media "
                  "| Ev. Mejor | T. Medio(s)|   Semilla    | Mejor FObj |")
    print(encabezado)
    print("-" * len(encabezado))
    for fila in filas_tabla:
        print(fila)
    print("-" * len(encabezado))

    parametros = [f" - {c}: {r['parametros_extra']}"
                  for c, r in resultados_globales.items()
                  if r.get('parametros_extra')]
    if parametros:
        print("\n --- Parámetros Específicos ---")
        for p in parametros:
            print(p)

    if datos_para_graficar:
        graficar_historiales(datos_para_graficar, nombre_algoritmo)

    return resultados_globales


def ejecutar_tabla_parcial(id_algoritmo, casos=CASOS, semillas=SEMILLAS_P2,
                           tolerancia=TOLERANCIA, **kwargs):
    """
    Genera la Tabla 1.1 (Resultados Parciales) de un algoritmo: 5 ejecuciones
    (una por semilla) por caso usando la función objetivo estándar fobj_ratio.
    """
    if id_algoritmo not in REGISTRY:
        raise ValueError(f'Algoritmo "{id_algoritmo}" no reconocido.')

    info = REGISTRY[id_algoritmo]
    funcion_algoritmo = info['funcion']
    coordenadas = _cargar_coords()
    datos_por_caso = {}

    for nombre_caso, datos in casos.items():
        bicis, capacidad = datos['bicis'], datos['capacidad']
        est_base = obtener_estaciones_a_visitar(bicis, capacidad, tolerancia)
        ejecuciones = []
        for sem in semillas:
            res = funcion_algoritmo(
                funcion_objetivo=fobj_ratio, estaciones_base=est_base,
                coordenadas=coordenadas, caso_bicis=bicis,
                caso_capacidad=capacidad, evaluar_ruta=evaluar_ruta,
                semilla=sem, casos=casos, **kwargs
            )
            ejecuciones.append({
                'ev': res['evaluaciones'], 'fobj': res['fobj'],
                'kms': res['kms'], 'entropia': res['entropia'],
            })
        datos_por_caso[nombre_caso] = ejecuciones

    generar_tabla_parcial(info['nombre_display'], datos_por_caso)
    return datos_por_caso


def ejecutar_estudio_parametrico(id_algoritmo, parametro, valores,
                                 caso_nombre="Caso 1", semillas=SEMILLAS_P2,
                                 tolerancia=TOLERANCIA, **kwargs_fijos):
    """
    Estudio paramétrico (Notable/Sobresaliente): barre 'parametro' sobre 'valores'
    para un algoritmo evolutivo, ejecutando las 5 semillas con fobj_ratio sobre un
    caso. Reporta Media/Mejor/σ/CV por valor y, según el algoritmo, una métrica
    extra de Caja Blanca (CHC: nº de reinicios; Multimodal: dominantes medios).
    Sirve para justificar la elección del valor por defecto como 'sweet spot'.
    """
    if id_algoritmo not in REGISTRY:
        raise ValueError(f'Algoritmo "{id_algoritmo}" no reconocido.')

    info = REGISTRY[id_algoritmo]
    funcion_algoritmo = info['funcion']
    coordenadas = _cargar_coords()
    datos = CASOS[caso_nombre]
    bicis, capacidad = datos['bicis'], datos['capacidad']
    est_base = obtener_estaciones_a_visitar(bicis, capacidad, tolerancia)

    print(f"\n{'='*80}")
    print(f" ESTUDIO PARAMÉTRICO — {info['nombre_display']} | "
          f"parámetro='{parametro}' | {caso_nombre} (L={len(est_base)})")
    print(f"{'='*80}")

    filas = []
    for val in valores:
        scores, extra_vals = [], []
        for sem in semillas:
            res = funcion_algoritmo(
                funcion_objetivo=fobj_ratio, estaciones_base=est_base,
                coordenadas=coordenadas, caso_bicis=bicis,
                caso_capacidad=capacidad, evaluar_ruta=evaluar_ruta,
                semilla=sem, **{parametro: val}, **kwargs_fijos
            )
            scores.append(res['fobj'])
            if id_algoritmo == 'chc':
                extra_vals.append(len(res['historial'].get('reinicios_gen', [])))
            elif id_algoritmo == 'multimodal':
                dom = res['historial'].get('dominantes_por_generacion', [])
                if dom:
                    extra_vals.append(float(np.mean(dom)))

        media = float(np.mean(scores))
        fila = {'valor': val, 'media': media, 'best': float(np.min(scores)),
                'sigma': float(np.std(scores)),
                'cv': (np.std(scores) / media * 100) if media > 0 else 0.0}
        if extra_vals:
            fila['extra'] = float(np.mean(extra_vals))
        filas.append(fila)
        print(f" - {parametro}={val}: media={fila['media']:.4f} "
              f"best={fila['best']:.4f} CV={fila['cv']:.2f}%")

    columna_extra = ('Reinicios' if id_algoritmo == 'chc'
                     else 'Dominantes' if id_algoritmo == 'multimodal' else None)
    tabla_estudio_parametrico(info['nombre_display'], parametro, filas, columna_extra)
    return filas


def ejecutar_analisis_cajas_p3(caso_nombre="Caso 1", mejor_fobj_conocido=None,
                               semillas=SEMILLAS_P2, tolerancia=TOLERANCIA):
    """
    Análisis de Caja Negra (CV, RPD) y Caja Blanca (convergencia CHC, dinámica
    de Clearing, diversificación estructural por arcos) para los tres AG.
    """
    if mejor_fobj_conocido is None:
        mejor_fobj_conocido = MEJOR_FOBJ_CONOCIDO.get(caso_nombre, 1.0)

    print(f"\n{'='*80}")
    print(f" ANÁLISIS CAJA NEGRA Y BLANCA — ALGORITMOS EVOLUTIVOS ({caso_nombre})")
    print(f"{'='*80}")

    coordenadas = _cargar_coords()
    datos = CASOS[caso_nombre]
    bicis, capacidad = datos['bicis'], datos['capacidad']
    est_base = obtener_estaciones_a_visitar(bicis, capacidad, tolerancia)

    resultados_fobj = {'Básico': [], 'CHC': [], 'Multimodal': []}
    rutas = {'Básico': [], 'CHC': [], 'Multimodal': []}
    historiales_chc = []
    historiales_clearing = []

    print("Ejecutando los 3 AG (5 semillas) con fobj_ratio...\n")
    for sem in semillas:
        rb = algoritmo_genetico_basico(fobj_ratio, est_base, coordenadas, bicis,
                                       capacidad, evaluar_ruta, sem)
        resultados_fobj['Básico'].append(fobj_ratio(rb['kms'], rb['entropia']))
        rutas['Básico'].append(rb['ruta'])

        rc = algoritmo_genetico_chc(fobj_ratio, est_base, coordenadas, bicis,
                                    capacidad, evaluar_ruta, sem,
                                    registrar_diversidad=True)
        resultados_fobj['CHC'].append(fobj_ratio(rc['kms'], rc['entropia']))
        rutas['CHC'].append(rc['ruta'])
        rc['historial']['semilla'] = sem
        historiales_chc.append(rc['historial'])

        rm = algoritmo_genetico_multimodal(fobj_ratio, est_base, coordenadas,
                                           bicis, capacidad, evaluar_ruta, sem)
        resultados_fobj['Multimodal'].append(fobj_ratio(rm['kms'], rm['entropia']))
        rutas['Multimodal'].append(rm['ruta'])
        rm['historial']['semilla'] = sem
        historiales_clearing.append(rm['historial'])

    print("--- 1. MEDIDAS DE CAJA NEGRA (CV, RPD) ---")
    calcular_metricas_caja_negra(resultados_fobj, mejor_fobj_conocido)
    graficar_boxplot_caja_negra(resultados_fobj, caso_nombre)

    print("\n--- 2. MEDIDAS DE CAJA BLANCA ---")
    print("\n[2.1] Diversificación estructural por POSICIÓN (Hamming clásica):")
    for algo in resultados_fobj:
        print(f"  {algo}:")
        calcular_tasa_overlap_hamming(rutas[algo], n_posiciones_prefijo=4)

    print("\n[2.2] Diversificación estructural por ARCOS (Hamming de secuencia):")
    for algo in resultados_fobj:
        diversificacion_estructural_arcos(rutas[algo], etiqueta=algo)

    print("\n[2.3] Convergencia y reinicios de CHC:")
    graficar_convergencia_chc(historiales_chc, caso_nombre)

    print("\n[2.3b] Fitness mejor/media/peor de CHC (ejecución representativa):")
    graficar_fitness_chc(historiales_chc, caso_nombre)

    print("\n[2.3c] Diversidad genética de CHC (Hamming por arcos normalizada):")
    graficar_diversidad_chc(historiales_chc, caso_nombre)

    print("\n[2.4] Dinámica de Clearing del AG Multimodal:")
    analizar_clearing(historiales_clearing, caso_nombre)

    print("\nAnálisis estadístico finalizado.")


def mostrar_soluciones_multimodal(caso_nombre="Caso 1", semilla=None, k=3,
                                  tolerancia=TOLERANCIA, **kwargs):
    """
    Caja Blanca multimodal (opcional de la consegna): ejecuta UNA vez el AG
    Multimodal y muestra las mejores soluciones de hasta 'k' nichos DISTINTOS
    (separados > sigma por arcos), con su F. Objetivo (ratio), kms, entropía y
    su representación en el mapa. Evidencia que el Clearing mantiene varios
    óptimos estructuralmente diferentes en la población final.
    """
    if semilla is None:
        semilla = SEMILLAS_P2[0]
    coordenadas = _cargar_coords()
    datos = CASOS[caso_nombre]
    bicis, capacidad = datos['bicis'], datos['capacidad']
    est_base = obtener_estaciones_a_visitar(bicis, capacidad, tolerancia)

    res = algoritmo_genetico_multimodal(
        fobj_ratio, est_base, coordenadas, bicis, capacidad, evaluar_ruta,
        semilla, **kwargs)
    soluciones = res['historial'].get('soluciones_nicho', [])[:k]

    print(f"\n{'='*80}")
    print(f" SOLUCIONES MULTIMODALES (nichos distintos) — {caso_nombre} "
          f"| semilla {semilla} | {len(soluciones)} nichos")
    print(f"{'='*80}")
    if not soluciones:
        print(">> Sin soluciones de nicho registradas.")
        return res

    for n, sol in enumerate(soluciones, 1):
        score = fobj_ratio(sol['kms'], sol['entropia'])
        print(f"\n>> Nicho {n}: F.Obj(ratio)={score:.4f} | "
              f"Kms={sol['kms']:.2f} | Entropía={sol['entropia']:.4f}")
        ev = evaluar_ruta(sol['ruta'], bicis, capacidad, coordenadas)
        display(dibujar_mapa_trayecto(coordenadas, ev['movimientos_mapa']))
    return res


if __name__ == '__main__':
    # Smoke test reducido: 1 caso, 2 semillas, presupuesto corto.
    print(">> SMOKE TEST Práctica 3 (reducido)")
    caso_min = {'Caso 1': CASOS['Caso 1']}
    for algo in ('basico', 'chc', 'multimodal'):
        ejecutar_experimento_ga(algo, casos=caso_min,
                                 semillas=SEMILLAS_P2[:2],
                                 max_evaluaciones=400)

    print("\n>> SMOKE TEST estudio paramétrico (reducido)")
    ejecutar_estudio_parametrico('multimodal', 'sigma_clearing', [2, 4, 8],
                                 caso_nombre='Caso 1', semillas=SEMILLAS_P2[:2],
                                 max_evaluaciones=400)
    ejecutar_estudio_parametrico('chc', 'umbral_ini', [2, 4, 8],
                                 caso_nombre='Caso 1', semillas=SEMILLAS_P2[:2],
                                 max_evaluaciones=400)
    ejecutar_estudio_parametrico('basico', 'k_torneo', [2, 3, 6],
                                 caso_nombre='Caso 1', semillas=SEMILLAS_P2[:2],
                                 max_evaluaciones=400)
