import numpy as np
from utils import obtener_estaciones_a_visitar, graficar_historiales, FUNCIONES_OBJETIVO, fobj_ratio

def ejecutar_experimento(
    nombre_algoritmo,
    funcion_algoritmo,
    is_deterministic,
    casos, semillas, tolerancia,
    coordenadas, evaluar_ruta, **kwargs
    ):
    """
    Ejecuta un experimento de prueba para un algoritmo de optimización dado.
    Compara los resultados obtenidos con diferentes funciones objetivo y semillas,
    y presenta los resultados en una tabla resumen, además de graficar los
    historiales de evolución para los casos no determinísticos.
    """
    
    resultados_globales = {}
    print(f"\n{'='*93}")
    print(f" EXPERIMENTACIÓN: {nombre_algoritmo.upper()}")
    print(f"{'='*93}")

    datos_para_graficar = []
    filas_tabla = []
    parametros_extra_lista = []

    for nombre_caso, datos in casos.items():
        bicis, capacidad = datos['bicis'], datos['capacidad']
        est_base = obtener_estaciones_a_visitar(bicis, capacidad, tolerancia)


        semillas_a_usar = [None] if is_deterministic else semillas

        # Variables para rastrear al Mejor absoluto de este caso (usando el Arbitro)
        mejor_res_absoluto = None
        evaluaciones_totales_lista = [] # Para la media de evals
        
        # Se itera sobre todas las funciones objetivo
        for f_obj in FUNCIONES_OBJETIVO:
            mejor_res_fobj = None
            evals_por_fobj = []
            
            # Para cada función, se itera sobre todas las semillas
            for sem in semillas_a_usar:
                res = funcion_algoritmo(
                    estaciones_base=est_base, coordenadas=coordenadas, 
                    caso_bicis=bicis, caso_capacidad=capacidad, 
                    evaluar_ruta=evaluar_ruta, semilla=sem, 
                    funcion_objetivo=f_obj,
                    casos=casos,
                    **kwargs
                )
                
                evals_por_fobj.append(res['evaluaciones'])
                
                # Se compara internamente usando el valor bruto que devuelve la función actual
                if mejor_res_fobj is None or res['fobj'] < mejor_res_fobj['fobj']:
                    mejor_res_fobj = res
                    
            evaluaciones_totales_lista.extend(evals_por_fobj)
            
            # Se evalua la mejor ruta de esta F.Obj usando la F.Obj estándar (Ratio) para poder comparar peras con peras
            score_universal = fobj_ratio(mejor_res_fobj['kms'], mejor_res_fobj['entropia'])
            mejor_res_fobj['score_universal'] = score_universal
            mejor_res_fobj['nombre_fobj'] = f_obj.__name__ # Para saber quién ganó
            
            if mejor_res_absoluto is None or score_universal < mejor_res_absoluto['score_universal']:
                mejor_res_absoluto = mejor_res_fobj

        # Estadísticas finales
        ev_media = np.mean(evaluaciones_totales_lista)
        ev_mejor = mejor_res_absoluto['evaluaciones']

        resultados_globales[nombre_caso] = mejor_res_absoluto

        semilla_str = str(mejor_res_absoluto['semilla']) if mejor_res_absoluto['semilla'] is not None else 'N/A'
        nombre_fobj_corta = mejor_res_absoluto['nombre_fobj'].replace('fobj_', '')[:10]
        
        # Fila de la tabla
        fila = f"| {nombre_caso:<6} | {mejor_res_absoluto['score_universal']:>11.4f} | {mejor_res_absoluto['kms']:>7.2f} | {mejor_res_absoluto['entropia']:>8.4f} | {ev_media:>8.1f} | {ev_mejor:>9} | {semilla_str:>7} | {nombre_fobj_corta:<10} |"
        filas_tabla.append(fila)
        
        if 'parametros_extra' in mejor_res_absoluto:
            parametros_extra_lista.append(f" - {nombre_caso}: {mejor_res_absoluto['parametros_extra']}")
                
        if not is_deterministic and 'historial' in mejor_res_absoluto:
            historial_estandarizado = mejor_res_absoluto['historial']
            historial_estandarizado['fobj'] = [fobj_ratio(k, e) for k, e in zip(historial_estandarizado['kms'], historial_estandarizado['entropia'])]
            
            datos_para_graficar.append({
                'historial': historial_estandarizado,
                'nombre_caso': nombre_caso,
                'semilla': mejor_res_absoluto['semilla']
            })
    
    # Imprimir tabla
    encabezado = f"| Caso   | Score (Ratio)| Kms     | Entropía | Ev. Media| Ev. Mejor | Semilla | Mejor FObj |"
    separador = "-" * len(encabezado)
    
    print(encabezado)
    print(separador)
    for fila in filas_tabla:
        print(fila)
    print(separador)
    
    if parametros_extra_lista:
        print("\n --- Parámetros Específicos del Algoritmo ---")
        for info in parametros_extra_lista:
            print(info)

    if not is_deterministic and datos_para_graficar:
        graficar_historiales(datos_para_graficar, nombre_algoritmo)

    return resultados_globales