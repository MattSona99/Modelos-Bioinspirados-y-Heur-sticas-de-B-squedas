import numpy as np
from utils import obtener_estaciones_a_visitar, graficar_historiales

def ejecutar_experimento(
    nombre_algoritmo,
    funcion_algoritmo,
    is_deterministic,
    casos, semillas, tolerancia,
    coordenadas, evaluar_ruta, **kwargs
    ):
    """
    Ejecuta un experimento para un algoritmo dado,
    evaluando su desempeño en varios casos y semillas.
    """
    resultados_globales = {}
    print(f"\n{'='*60}\n EXPERIMENTACIÓN: {nombre_algoritmo.upper()}\n{'='*60}")

    datos_para_graficar = []

    for nombre_caso, datos in casos.items():
        bicis, capacidad = datos['bicis'], datos['capacidad']
        est_base = obtener_estaciones_a_visitar(bicis, capacidad, tolerancia)

        # Si es determinístico (Greedy), se ejecuta 1 sola vez con semilla None.
        # Si no, se ejecutan las 5 semillas.
        semillas_a_usar = [None] if is_deterministic else semillas

        mejores_fobj = []
        evaluaciones_lista = []
        mejor_res = None

        for sem in semillas_a_usar:
            res = funcion_algoritmo(
                estaciones_base=est_base, coordenadas=coordenadas, 
                caso_bicis=bicis, caso_capacidad=capacidad, 
                evaluar_ruta=evaluar_ruta, semilla=sem, **kwargs
            )

            mejores_fobj.append(res['fobj'])
            evaluaciones_lista.append(res['evaluaciones'])
            
            if mejor_res is None or res['fobj'] < mejor_res['fobj']:
                mejor_res = res
                
        # Calcular estadísticas finales
        fobj_media, fobj_std = np.mean(mejores_fobj), np.std(mejores_fobj)

        resultados_globales[nombre_caso] = {
            'Mejor F. Objetivo': mejor_res['fobj'],
            'Mejor Kms': mejor_res['kms'],
            'Mejor Entropía': mejor_res['entropia'],
            'Media F. Obj': fobj_media,
            'Std F. Obj': fobj_std,
            'Ev. Medias': np.mean(evaluaciones_lista),
            'Ev. Mejor': mejor_res['evaluaciones']
        }

        # Imprimir resultados del caso
        print(f"\n--- {nombre_caso} ---")
        print(f" -> Mejor F.Obj: {mejor_res['fobj']:.4f} (Semilla: {mejor_res['semilla']})")
        print(f" -> Media F.Obj: {fobj_media:.4f} ± {fobj_std:.4f}")

        # Si no es determinístico, guardamos el historial para graficar después de todos los casos
        if not is_deterministic and 'historial' in mejor_res:
            datos_para_graficar.append({
                'historial': mejor_res['historial'],
                'nombre_caso': nombre_caso,
                'semilla': mejor_res['semilla']
            })

    # Graficar los históricos de las mejores soluciones para cada caso (solo para algoritmos no determinísticos)
    if not is_deterministic and datos_para_graficar:
        graficar_historiales(datos_para_graficar, nombre_algoritmo)

    return resultados_globales