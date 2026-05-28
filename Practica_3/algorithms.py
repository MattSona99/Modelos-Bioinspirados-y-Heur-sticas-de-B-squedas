import sys
import os
import random

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

from Practica_1.config import TAM_POBLACION, MAX_EVALUACIONES, PROB_CRUCE
from Practica_3.utils import (
    crear_individuo_aleatorio,
    evaluar_individuo,
    seleccion_torneo,
    cruce_ox,
    mutacion_2opt,
    cruce_hux_orden,
    distancia_hamming_arcos,
)


def _historial_vacio():
    return {'evaluaciones': [], 'fobj_actual': [], 'fobj': [],
            'kms': [], 'entropia': []}


def _registrar(historial, evals, fobj_actual, record):
    """Registra un punto de trayectoria (Caja Blanca, compatible con P1/P2)."""
    historial['evaluaciones'].append(evals)
    historial['fobj_actual'].append(fobj_actual)
    historial['fobj'].append(record['fobj'])
    historial['kms'].append(record['kms'])
    historial['entropia'].append(record['entropia'])


def _evaluar_poblacion(poblacion, caso_bicis, caso_capacidad, coordenadas,
                       evaluar_ruta, funcion_objetivo, **kwargs):
    fit, kms, ent = [], [], []
    for ind in poblacion:
        f, k, e = evaluar_individuo(ind, caso_bicis, caso_capacidad, coordenadas,
                                    evaluar_ruta, funcion_objetivo, **kwargs)
        fit.append(f)
        kms.append(k)
        ent.append(e)
    return fit, kms, ent


def _resultado(record, historial, evaluaciones, semilla, extra=None):
    res = {
        'ruta': record['ruta'], 'fobj': record['fobj'],
        'kms': record['kms'], 'entropia': record['entropia'],
        'historial': historial, 'evaluaciones': evaluaciones, 'semilla': semilla,
    }
    if extra:
        res['historial'].update(extra)
        res['parametros_extra'] = extra.get('parametros_extra', '')
    return res


# ---------------------------------------------
# 1. Algoritmo Genético Básico (Generacional + Elitismo)
# ---------------------------------------------
def algoritmo_genetico_basico(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad,
    evaluar_ruta, semilla, tam_poblacion=TAM_POBLACION,
    max_evaluaciones=MAX_EVALUACIONES, prob_cruce=PROB_CRUCE,
    k_torneo=None, modelo='generacional', elitismo=True, **kwargs
):
    """
    AG Básico. Selección por torneo (K=10%), cruce OX al 90% y, cuando no hay
    cruce, mutación 2-opt (5-10%) de cada padre.
    - modelo='generacional' (def.): nueva población completa por generación; con
      'elitismo' el mejor de la generación previa sustituye al peor hijo.
    - modelo='estacionario': steady-state; cada descendiente entra por torneo de
      reemplazo (muere el peor de K candidatos aleatorios), como recomienda el
      guion para el estacionario.
    'k_torneo' permite barrer la presión selectiva (None = 10% de la población).
    """
    kwargs.pop('casos', None)
    random.seed(semilla)
    historial = _historial_vacio()
    record = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}

    estaciones_base = list(estaciones_base)
    if len(estaciones_base) < 2:
        f, k, e = evaluar_individuo(estaciones_base, caso_bicis, caso_capacidad,
                                    coordenadas, evaluar_ruta, funcion_objetivo,
                                    **kwargs)
        record.update({'ruta': list(estaciones_base), 'fobj': f, 'kms': k,
                        'entropia': e})
        _registrar(historial, 1, f, record)
        return _resultado(record, historial, 1, semilla)

    if k_torneo is None:
        k_torneo = max(1, round(0.10 * tam_poblacion))
    poblacion = [crear_individuo_aleatorio(estaciones_base)
                 for _ in range(tam_poblacion)]
    fit, kms, ent = _evaluar_poblacion(poblacion, caso_bicis, caso_capacidad,
                                       coordenadas, evaluar_ruta,
                                       funcion_objetivo, **kwargs)
    evaluaciones = tam_poblacion

    def _actualizar_record():
        i = min(range(len(fit)), key=lambda x: fit[x])
        if fit[i] < record['fobj']:
            record.update({'ruta': list(poblacion[i]), 'fobj': fit[i],
                            'kms': kms[i], 'entropia': ent[i]})
        return i

    mejor_idx = _actualizar_record()
    _registrar(historial, evaluaciones, fit[mejor_idx], record)

    if modelo == 'estacionario':
        # Steady-state: cada descendiente entra por torneo de reemplazo
        # (muere el peor de K candidatos aleatorios). El mejor casi nunca es
        # elegido como víctima, por lo que el elitismo es implícito.
        ultimo = evaluaciones
        while evaluaciones < max_evaluaciones:
            p1 = seleccion_torneo(poblacion, fit, k_torneo)
            p2 = seleccion_torneo(poblacion, fit, k_torneo)
            if random.random() < prob_cruce:
                descendientes = list(cruce_ox(p1, p2))
            else:
                descendientes = [mutacion_2opt(p1), mutacion_2opt(p2)]
            for h in descendientes:
                if evaluaciones >= max_evaluaciones:
                    break
                fh, kh, eh = evaluar_individuo(h, caso_bicis, caso_capacidad,
                                               coordenadas, evaluar_ruta,
                                               funcion_objetivo, **kwargs)
                evaluaciones += 1
                cand = random.sample(range(tam_poblacion), k_torneo)
                victima = max(cand, key=lambda x: fit[x])
                if fh < fit[victima]:
                    poblacion[victima] = h
                    fit[victima], kms[victima], ent[victima] = fh, kh, eh
                if evaluaciones - ultimo >= tam_poblacion:
                    idx = _actualizar_record()
                    _registrar(historial, evaluaciones, fit[idx], record)
                    ultimo = evaluaciones
        idx = _actualizar_record()
        _registrar(historial, evaluaciones, fit[idx], record)
    else:
        while evaluaciones < max_evaluaciones:
            if elitismo:
                elite_idx = min(range(len(fit)), key=lambda x: fit[x])
                elite = (list(poblacion[elite_idx]), fit[elite_idx],
                         kms[elite_idx], ent[elite_idx])

            nueva = []
            while len(nueva) < tam_poblacion:
                p1 = seleccion_torneo(poblacion, fit, k_torneo)
                p2 = seleccion_torneo(poblacion, fit, k_torneo)
                if random.random() < prob_cruce:
                    h1, h2 = cruce_ox(p1, p2)
                else:
                    h1, h2 = mutacion_2opt(p1), mutacion_2opt(p2)
                nueva.append(h1)
                if len(nueva) < tam_poblacion:
                    nueva.append(h2)

            poblacion = nueva
            fit, kms, ent = _evaluar_poblacion(poblacion, caso_bicis,
                                               caso_capacidad, coordenadas,
                                               evaluar_ruta, funcion_objetivo,
                                               **kwargs)
            evaluaciones += tam_poblacion

            if elitismo:
                # El peor hijo es sustituido por la élite anterior.
                peor = max(range(len(fit)), key=lambda x: fit[x])
                if elite[1] < fit[peor]:
                    poblacion[peor] = elite[0]
                    fit[peor], kms[peor], ent[peor] = elite[1], elite[2], elite[3]

            mejor_idx = _actualizar_record()
            _registrar(historial, evaluaciones, fit[mejor_idx], record)

    return _resultado(record, historial, evaluaciones, semilla)


# ---------------------------------------------
# 2. Algoritmo Genético CHC
# ---------------------------------------------
def algoritmo_genetico_chc(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad,
    evaluar_ruta, semilla, tam_poblacion=TAM_POBLACION,
    max_evaluaciones=MAX_EVALUACIONES, umbral_ini=None,
    registrar_diversidad=False, **kwargs
):
    """
    CHC con codificación de orden. Prevención de incesto mediante la distancia
    de Hamming por arcos: sólo se recombinan padres con distancia > umbral
    (umbral inicial = L/4; 'umbral_ini' permite barrerlo). Supervivencia
    elitista (mejores N de padres+hijos),
    sin mutación. Si ningún hijo entra en la población, el umbral baja en 1;
    al llegar a 0 se reinicia la población copiando el mejor y aleatorizando
    el resto (mismo run). Registra reinicios para el estudio de convergencia.
    Con 'registrar_diversidad=True' también guarda la media de la distancia de
    Hamming por arcos normalizada por generación (curva de diversidad genética).
    """
    kwargs.pop('casos', None)
    random.seed(semilla)
    historial = _historial_vacio()
    historial['mejor_por_generacion'] = []
    historial['media_por_generacion'] = []
    historial['peor_por_generacion'] = []
    historial['diversidad_por_generacion'] = []
    historial['reinicios_gen'] = []
    record = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}

    estaciones_base = list(estaciones_base)
    L = len(estaciones_base)
    if L < 2:
        f, k, e = evaluar_individuo(estaciones_base, caso_bicis, caso_capacidad,
                                    coordenadas, evaluar_ruta, funcion_objetivo,
                                    **kwargs)
        record.update({'ruta': list(estaciones_base), 'fobj': f, 'kms': k,
                        'entropia': e})
        _registrar(historial, 1, f, record)
        return _resultado(record, historial, 1, semilla)

    if umbral_ini is None:
        umbral_ini = max(1, L // 4)
    else:
        umbral_ini = max(1, int(umbral_ini))
    umbral = umbral_ini

    poblacion = [crear_individuo_aleatorio(estaciones_base)
                 for _ in range(tam_poblacion)]
    fit, kms, ent = _evaluar_poblacion(poblacion, caso_bicis, caso_capacidad,
                                       coordenadas, evaluar_ruta,
                                       funcion_objetivo, **kwargs)
    evaluaciones = tam_poblacion
    n_reinicios = 0
    generacion = 0

    def _mejor():
        i = min(range(len(fit)), key=lambda x: fit[x])
        if fit[i] < record['fobj']:
            record.update({'ruta': list(poblacion[i]), 'fobj': fit[i],
                            'kms': kms[i], 'entropia': ent[i]})
        return i

    n_arcos = L + 1

    def _diversidad_poblacion():
        """Media de la distancia de Hamming por arcos normalizada entre todos
        los pares de la población (0 = idénticas, 1 = sin arcos comunes)."""
        n = len(poblacion)
        if n < 2:
            return 0.0
        total, pares = 0.0, 0
        for a in range(n):
            for b in range(a + 1, n):
                total += distancia_hamming_arcos(poblacion[a], poblacion[b]) / n_arcos
                pares += 1
        return total / pares if pares else 0.0

    def _registrar_generacion(idx):
        historial['mejor_por_generacion'].append(fit[idx])
        historial['media_por_generacion'].append(sum(fit) / len(fit))
        historial['peor_por_generacion'].append(max(fit))
        if registrar_diversidad:
            historial['diversidad_por_generacion'].append(_diversidad_poblacion())

    i = _mejor()
    _registrar(historial, evaluaciones, fit[i], record)
    _registrar_generacion(i)

    while evaluaciones < max_evaluaciones:
        generacion += 1
        indices = list(range(tam_poblacion))
        random.shuffle(indices)

        hijos, hijos_fit, hijos_kms, hijos_ent = [], [], [], []
        for a in range(0, tam_poblacion - 1, 2):
            p1, p2 = poblacion[indices[a]], poblacion[indices[a + 1]]
            if distancia_hamming_arcos(p1, p2) > umbral:
                c1, c2 = cruce_hux_orden(p1, p2)
                for c in (c1, c2):
                    if evaluaciones >= max_evaluaciones:
                        break
                    f, k, e = evaluar_individuo(c, caso_bicis, caso_capacidad,
                                                coordenadas, evaluar_ruta,
                                                funcion_objetivo, **kwargs)
                    evaluaciones += 1
                    hijos.append(c)
                    hijos_fit.append(f)
                    hijos_kms.append(k)
                    hijos_ent.append(e)

        # Supervivencia elitista (μ+λ): mejores N de padres ∪ hijos.
        combinado = list(zip(poblacion + hijos, fit + hijos_fit,
                             kms + hijos_kms, ent + hijos_ent,
                             [0] * len(poblacion) + [1] * len(hijos)))
        combinado.sort(key=lambda x: x[1])
        supervivientes = combinado[:tam_poblacion]

        algun_hijo_insertado = any(tag == 1 for *_, tag in supervivientes)

        poblacion = [list(s[0]) for s in supervivientes]
        fit = [s[1] for s in supervivientes]
        kms = [s[2] for s in supervivientes]
        ent = [s[3] for s in supervivientes]

        if not algun_hijo_insertado:
            umbral -= 1

        if umbral < 0:
            # Reinicio: el mejor se conserva como patrón, el resto aleatorio.
            n_reinicios += 1
            historial['reinicios_gen'].append(generacion)
            mejor_ruta = list(record['ruta'])
            poblacion = [mejor_ruta] + [crear_individuo_aleatorio(estaciones_base)
                                        for _ in range(tam_poblacion - 1)]
            fit, kms, ent = _evaluar_poblacion(
                poblacion, caso_bicis, caso_capacidad, coordenadas,
                evaluar_ruta, funcion_objetivo, **kwargs)
            evaluaciones += tam_poblacion
            umbral = umbral_ini

        i = _mejor()
        _registrar(historial, evaluaciones, fit[i], record)
        _registrar_generacion(i)

    return _resultado(record, historial, evaluaciones, semilla,
                      extra={'parametros_extra': f"reinicios={n_reinicios}, "
                             f"umbral_ini={umbral_ini}",
                             'mejor_por_generacion': historial['mejor_por_generacion'],
                             'media_por_generacion': historial['media_por_generacion'],
                             'peor_por_generacion': historial['peor_por_generacion'],
                             'diversidad_por_generacion':
                             historial['diversidad_por_generacion'],
                             'reinicios_gen': historial['reinicios_gen']})


# ---------------------------------------------
# 3. Algoritmo Genético Multimodal (Clearing)
# ---------------------------------------------
def algoritmo_genetico_multimodal(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad,
    evaluar_ruta, semilla, tam_poblacion=TAM_POBLACION,
    max_evaluaciones=MAX_EVALUACIONES, prob_cruce=PROB_CRUCE,
    capacidad_nicho=1, sigma_clearing=None, k_torneo=None,
    n_soluciones_nicho=5, **kwargs
):
    """
    AG Multimodal mediante Clearing sobre el AG Básico. Tras evaluar cada
    población, se ordena por fitness y, recorriendo de mejor a peor, cada
    ganador "limpia" (penaliza con fitness infinito) a los individuos dentro
    de su radio sigma —medido con la distancia de Hamming por arcos— salvo los
    'capacidad_nicho' mejores de cada nicho. La selección usa el fitness ya
    limpiado, lo que mantiene varios óptimos (nichos) en la población.
    Devuelve además 'soluciones_nicho': las mejores soluciones de hasta
    'n_soluciones_nicho' nichos DISTINTOS de la población final.
    """
    kwargs.pop('casos', None)
    random.seed(semilla)
    historial = _historial_vacio()
    historial['dominantes_por_generacion'] = []
    record = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}

    estaciones_base = list(estaciones_base)
    L = len(estaciones_base)
    if L < 2:
        f, k, e = evaluar_individuo(estaciones_base, caso_bicis, caso_capacidad,
                                    coordenadas, evaluar_ruta, funcion_objetivo,
                                    **kwargs)
        record.update({'ruta': list(estaciones_base), 'fobj': f, 'kms': k,
                        'entropia': e})
        _registrar(historial, 1, f, record)
        return _resultado(record, historial, 1, semilla)

    sigma = sigma_clearing if sigma_clearing is not None else max(1, L // 4)
    if k_torneo is None:
        k_torneo = max(1, round(0.10 * tam_poblacion))

    poblacion = [crear_individuo_aleatorio(estaciones_base)
                 for _ in range(tam_poblacion)]
    fit, kms, ent = _evaluar_poblacion(poblacion, caso_bicis, caso_capacidad,
                                       coordenadas, evaluar_ruta,
                                       funcion_objetivo, **kwargs)
    evaluaciones = tam_poblacion

    def _clearing():
        """Aplica clearing y devuelve (fitness_limpio, nº de dominantes)."""
        orden = sorted(range(len(fit)), key=lambda x: fit[x])
        fit_limpio = list(fit)
        limpiado = [False] * len(fit)
        dominantes = 0
        for pos, i in enumerate(orden):
            if limpiado[i]:
                continue
            dominantes += 1
            ganadores = 1
            for j in orden[pos + 1:]:
                if limpiado[j]:
                    continue
                if distancia_hamming_arcos(poblacion[i], poblacion[j]) <= sigma:
                    if ganadores < capacidad_nicho:
                        ganadores += 1
                    else:
                        fit_limpio[j] = float('inf')
                        limpiado[j] = True
        return fit_limpio, dominantes

    def _actualizar_record():
        i = min(range(len(fit)), key=lambda x: fit[x])
        if fit[i] < record['fobj']:
            record.update({'ruta': list(poblacion[i]), 'fobj': fit[i],
                            'kms': kms[i], 'entropia': ent[i]})
        return i

    fit_limpio, dominantes = _clearing()
    historial['dominantes_por_generacion'].append(dominantes)
    mejor_idx = _actualizar_record()
    _registrar(historial, evaluaciones, fit[mejor_idx], record)

    while evaluaciones < max_evaluaciones:
        elite_idx = min(range(len(fit)), key=lambda x: fit[x])
        elite = (list(poblacion[elite_idx]), fit[elite_idx],
                 kms[elite_idx], ent[elite_idx])

        nueva = []
        while len(nueva) < tam_poblacion:
            p1 = seleccion_torneo(poblacion, fit_limpio, k_torneo)
            p2 = seleccion_torneo(poblacion, fit_limpio, k_torneo)
            if random.random() < prob_cruce:
                h1, h2 = cruce_ox(p1, p2)
            else:
                h1, h2 = mutacion_2opt(p1), mutacion_2opt(p2)
            nueva.append(h1)
            if len(nueva) < tam_poblacion:
                nueva.append(h2)

        poblacion = nueva
        fit, kms, ent = _evaluar_poblacion(poblacion, caso_bicis, caso_capacidad,
                                           coordenadas, evaluar_ruta,
                                           funcion_objetivo, **kwargs)
        evaluaciones += tam_poblacion

        peor = max(range(len(fit)), key=lambda x: fit[x])
        if elite[1] < fit[peor]:
            poblacion[peor] = elite[0]
            fit[peor], kms[peor], ent[peor] = elite[1], elite[2], elite[3]

        fit_limpio, dominantes = _clearing()
        historial['dominantes_por_generacion'].append(dominantes)
        mejor_idx = _actualizar_record()
        _registrar(historial, evaluaciones, fit[mejor_idx], record)

    def _representantes_nicho(k):
        """Mejores soluciones de nichos DISTINTOS de la población final: la
        mejor y, después, cada siguiente que diste > sigma (por arcos) de todas
        las ya elegidas. Devuelve sus rutas y métricas reales (fobj/kms/entropía)."""
        orden = sorted(range(len(fit)), key=lambda x: fit[x])
        reps = []
        for i in orden:
            if any(distancia_hamming_arcos(poblacion[i], poblacion[r]) <= sigma
                   for r in reps):
                continue
            reps.append(i)
            if len(reps) >= k:
                break
        return [{'ruta': list(poblacion[i]), 'fobj': fit[i],
                 'kms': kms[i], 'entropia': ent[i]} for i in reps]

    soluciones_nicho = _representantes_nicho(n_soluciones_nicho)

    return _resultado(record, historial, evaluaciones, semilla,
                      extra={'parametros_extra': f"sigma={sigma}, "
                             f"capacidad_nicho={capacidad_nicho}",
                             'dominantes_por_generacion':
                             historial['dominantes_por_generacion'],
                             'soluciones_nicho': soluciones_nicho})
