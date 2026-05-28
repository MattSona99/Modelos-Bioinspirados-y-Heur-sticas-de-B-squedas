import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

from Practica_1.config import ESTACION_INICIO

# =============================================================================
# DISTANCIA DE HAMMING ADAPTADA A SECUENCIA (BASADA EN ARCOS)
# =============================================================================

def _arcos_de_ruta(ruta, inicio=ESTACION_INICIO):
    """
    Devuelve el conjunto de arcos dirigidos que describe una solución.

    El camión parte del depósito (estación de inicio), recorre la permutación
    de estaciones y vuelve al depósito. Por tanto, una solución NO es solo la
    permutación, sino la secuencia de arcos: inicio->r0, r0->r1, ..., rn->inicio.
    """
    if not ruta:
        return set()
    secuencia = [inicio] + list(ruta) + [inicio]
    return {(secuencia[i], secuencia[i + 1]) for i in range(len(secuencia) - 1)}


def distancia_hamming_arcos(ruta_a, ruta_b, inicio=ESTACION_INICIO):
    """
    Distancia de Hamming ADAPTADA A SECUENCIA.

    La Hamming clásica (cuántas posiciones difieren) falsea los datos en una
    permutación: dos rutas que recorren exactamente los mismos tramos pero
    desplazadas un puesto tendrían distancia máxima aun siendo casi idénticas
    en coste. Lo relevante en este problema no es la posición absoluta de una
    estación, sino qué estación se visita justo después de cuál (el ARCO), ya
    que de los arcos dependen los kilómetros recorridos.

    Definimos por tanto la distancia como el número de arcos presentes en una
    solución que NO existen en la otra. Ambas rutas tienen el mismo número de
    arcos, así que la medida es simétrica. Máximo = nº de arcos (len(ruta)+1).
    """
    arcos_a = _arcos_de_ruta(ruta_a, inicio)
    arcos_b = _arcos_de_ruta(ruta_b, inicio)
    return len(arcos_a - arcos_b)

# =============================================================================
# OPERADORES GENÉTICOS
# =============================================================================

def crear_individuo_aleatorio(estaciones_base):
    """Genera un cromosoma (permutación de estaciones) aleatorio."""
    individuo = list(estaciones_base)
    random.shuffle(individuo)
    return individuo


def evaluar_individuo(ruta, caso_bicis, caso_capacidad, coordenadas,
                      evaluar_ruta, funcion_objetivo, **kwargs):
    """
    Evalúa un cromosoma con el motor de la Práctica 1.
    Devuelve (fitness, kms, entropia). El conteo de evaluaciones lo gestiona
    el algoritmo que llama, para mantener la trazabilidad de Caja Blanca.
    """
    res = evaluar_ruta(ruta, caso_bicis, caso_capacidad, coordenadas)
    fitness = funcion_objetivo(kms=res['kms_recorridos'],
                               entropia=res['entropia'], **kwargs)
    return fitness, res['kms_recorridos'], res['entropia']


def seleccion_torneo(poblacion, fitnesses, k):
    """
    Selección por Torneo. Se eligen K participantes al azar y gana el de
    menor fitness (problema de minimización). K se recomienda ~10% de la
    población. Devuelve una COPIA del individuo ganador.
    """
    k = max(1, min(k, len(poblacion)))
    aspirantes = random.sample(range(len(poblacion)), k)
    ganador = min(aspirantes, key=lambda idx: fitnesses[idx])
    return list(poblacion[ganador])


def cruce_ox(padre1, padre2):
    """
    Order Crossover (OX). Se conserva un segmento del primer padre y el resto
    se rellena, en orden circular a partir del segundo punto de corte, con los
    genes del segundo padre que aún no están presentes. Devuelve 2 hijos
    (el segundo invirtiendo los roles de los padres).
    """
    n = len(padre1)
    if n < 2:
        return list(padre1), list(padre2)

    def _ox(p1, p2):
        a, b = sorted(random.sample(range(n), 2))
        hijo = [None] * n
        hijo[a:b + 1] = p1[a:b + 1]
        presentes = set(hijo[a:b + 1])

        pos = (b + 1) % n
        for desplazamiento in range(n):
            gen = p2[(b + 1 + desplazamiento) % n]
            if gen not in presentes:
                hijo[pos] = gen
                presentes.add(gen)
                pos = (pos + 1) % n
        return hijo

    return _ox(padre1, padre2), _ox(padre2, padre1)


def mutacion_2opt(ruta, porcentaje=None):
    """
    Mutación basada en el operador 2-opt: invierte un segmento del cromosoma.
    La longitud del segmento es un 5%-10% de la longitud del cromosoma
    (si no se fija 'porcentaje', se elige aleatoriamente en ese rango).
    """
    nueva = list(ruta)
    n = len(nueva)
    if n < 2:
        return nueva

    pct = porcentaje if porcentaje is not None else random.uniform(0.05, 0.10)
    longitud = max(2, int(round(pct * n)))
    longitud = min(longitud, n)

    inicio = random.randint(0, n - longitud)
    nueva[inicio:inicio + longitud] = reversed(nueva[inicio:inicio + longitud])
    return nueva


def cruce_hux_orden(padre1, padre2):
    """
    Recombinación tipo HUX adaptada a codificación de orden (para CHC).

    CHC nació para binario (HUX intercambia la mitad de los bits que difieren).
    Para una permutación, intercambiar valores por posición rompería la validez
    del cromosoma. Por ello, se localizan las posiciones en las que ambos padres
    difieren, se elige la mitad al azar y, para cada una, se transpone dentro
    del hijo el gen objetivo del otro padre con el que ocupa esa posición. Al
    realizarse mediante transposiciones, el resultado siempre es una permutación
    válida. Devuelve 2 hijos.
    """
    n = len(padre1)
    diferentes = [i for i in range(n) if padre1[i] != padre2[i]]
    if not diferentes:
        return list(padre1), list(padre2)

    cuantas = max(1, len(diferentes) // 2)
    seleccion = random.sample(diferentes, cuantas)

    def _recombinar(base, donante):
        hijo = list(base)
        pos = {gen: idx for idx, gen in enumerate(hijo)}
        for i in seleccion:
            objetivo = donante[i]
            j = pos[objetivo]
            if j == i:
                continue
            hijo[i], hijo[j] = hijo[j], hijo[i]
            pos[hijo[j]] = j
            pos[hijo[i]] = i
        return hijo

    return _recombinar(padre1, padre2), _recombinar(padre2, padre1)

# =============================================================================
# MÉTRICA DE DIVERSIFICACIÓN ESTRUCTURAL (BASADA EN ARCOS)
# =============================================================================

def diversificacion_estructural_arcos(rutas, etiqueta=""):
    """
    Tasa de diversificación estructural usando la distancia de Hamming por
    arcos: cuánto se parecen REALMENTE las soluciones obtenidas con distintas
    semillas (qué hace una solución parecida a otra → compartir arcos).
    """
    n = len(rutas)
    if n < 2:
        print(f">> {etiqueta} Diversificación: insuficientes rutas.")
        return 0.0

    distancias = []
    for i in range(n):
        for j in range(i + 1, n):
            n_arcos = len(rutas[i]) + 1
            d = distancia_hamming_arcos(rutas[i], rutas[j])
            distancias.append(d / n_arcos if n_arcos > 0 else 0.0)

    media = np.mean(distancias) * 100
    print(f">> {etiqueta} Diversificación estructural (arcos distintos): {media:.1f}%")
    if media < 30:
        print("   ¡Alerta! Soluciones estructuralmente parecidas: comparten "
              "muchos arcos (baja diversidad real).")
    else:
        print("   Buena diversidad: las soluciones recorren tramos distintos.")
    return media

# =============================================================================
# CAJA BLANCA ESPECÍFICA: CONVERGENCIA CHC
# =============================================================================

def graficar_convergencia_chc(historiales_chc, nombre_caso=""):
    """
    Estudio de convergencia de CHC: representa el valor del mejor individuo de
    la población por generación y marca con líneas verticales los reinicios.
    Sirve para decidir cuántos rearranques conviene ejecutar por problema.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, h in enumerate(historiales_chc):
        mejores = h.get('mejor_por_generacion', [])
        if not mejores:
            continue
        ax.plot(range(len(mejores)), mejores, alpha=0.7,
                label=f"Ejecución {idx + 1} (semilla {h.get('semilla', '?')})")
        for g in h.get('reinicios_gen', []):
            ax.axvline(g, color='red', linestyle='--', alpha=0.35)

    ax.set_title(f'Convergencia CHC y reinicios {nombre_caso}'.strip(), fontsize=14)
    ax.set_xlabel('Generación')
    ax.set_ylabel('Mejor F. Objetivo de la población')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    total_reinicios = [len(h.get('reinicios_gen', [])) for h in historiales_chc]
    if total_reinicios:
        print(f">> Reinicios por ejecución: {total_reinicios} "
              f"(media={np.mean(total_reinicios):.1f}). "
              f"El gráfico muestra cómo cada reinicio reinyecta diversidad "
              f"copiando el mejor y aleatorizando el resto (estrategia de "
              f"diversidad de CHC: sin mutación, divergencia por reinicio).")


def graficar_fitness_chc(historiales_chc, nombre_caso="", idx_ejecucion=0):
    """
    Convergencia del CHC para una ejecución representativa: mejor, media y peor
    fitness de la población por generación (réplica del 1er gráfico de ejemplo de
    la consegna). Las líneas verticales marcan los reinicios.
    """
    if not historiales_chc or idx_ejecucion >= len(historiales_chc):
        print(">> CHC fitness: sin datos.")
        return
    h = historiales_chc[idx_ejecucion]
    mejores = h.get('mejor_por_generacion', [])
    medias = h.get('media_por_generacion', [])
    peores = h.get('peor_por_generacion', [])
    if not mejores:
        print(">> CHC fitness: sin datos de generación.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    gens = range(len(mejores))
    ax.plot(gens, mejores, color='green', label='Mejor (Best)')
    if medias:
        ax.plot(gens, medias, color='blue', linestyle='--', label='Media (Average)')
    if peores:
        ax.plot(gens, peores, color='red', linestyle=':', label='Peor (Worst)')
    for g in h.get('reinicios_gen', []):
        ax.axvline(g, color='grey', linestyle='--', alpha=0.4)

    ax.set_title(f'Convergencia CHC (mejor/media/peor) {nombre_caso} '
                 f'— semilla {h.get("semilla", "?")}'.strip(), fontsize=14)
    ax.set_xlabel('Generación')
    ax.set_ylabel('Fitness (F. Objetivo)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def graficar_diversidad_chc(historiales_chc, nombre_caso="", idx_ejecucion=0):
    """
    Diversidad genética del CHC: media de la distancia de Hamming por arcos
    normalizada por generación (1 = población diversa, 0 = convergida). Réplica
    del 2º gráfico de ejemplo de la consegna. Explica la estrategia de diversidad
    de CHC: descenso por convergencia entre reinicios y salto en cada reinicio.

    Para que el "diente de sierra" sea legible se destaca UNA ejecución en primer
    plano (con sus reinicios) y se dibujan las demás muy tenues de fondo. No se
    promedia entre semillas porque los reinicios caen en generaciones distintas y
    la media borraría justamente el patrón. Requiere 'registrar_diversidad=True'.
    """
    validas = [i for i, h in enumerate(historiales_chc)
               if h.get('diversidad_por_generacion')]
    if not validas:
        print(">> CHC diversidad: no registrada (usar registrar_diversidad=True).")
        return
    if idx_ejecucion not in validas:
        idx_ejecucion = validas[0]

    fig, ax = plt.subplots(figsize=(11, 6))

    # Fondo: las demás ejecuciones, muy tenues.
    for i in validas:
        if i == idx_ejecucion:
            continue
        div = historiales_chc[i]['diversidad_por_generacion']
        ax.plot(range(len(div)), div, color='grey', alpha=0.20, linewidth=1)

    # Primer plano: la ejecución representativa, con sus reinicios.
    h0 = historiales_chc[idx_ejecucion]
    div0 = h0['diversidad_por_generacion']
    ax.plot(range(len(div0)), div0, color='purple', linewidth=2.2,
            label=f"Ejecución {idx_ejecucion + 1} (semilla {h0.get('semilla', '?')})")
    for g in h0.get('reinicios_gen', []):
        ax.axvline(g, color='red', linestyle='--', alpha=0.5)

    # Proxies de leyenda para el fondo y los reinicios.
    ax.plot([], [], color='grey', alpha=0.5, linewidth=1, label='Otras ejecuciones')
    ax.plot([], [], color='red', linestyle='--', alpha=0.5,
            label='Reinicios (ejecución destacada)')

    ax.set_title(f'Diversidad genética CHC {nombre_caso}'.strip(), fontsize=14)
    ax.set_xlabel('Generación')
    ax.set_ylabel('Media Distancia de Hamming normalizada')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    finales = [np.mean(h['diversidad_por_generacion'][-10:])
               for h in historiales_chc if h.get('diversidad_por_generacion')]
    if finales:
        print(f">> Diversidad media en las últimas 10 generaciones: {np.mean(finales):.3f}. "
              f"Los picos coinciden con los reinicios (reinyección de diversidad sin "
              f"mutación); el descenso entre reinicios refleja la convergencia por "
              f"prevención de incesto.")

# =============================================================================
# CAJA BLANCA ESPECÍFICA: DINÁMICA DE CLEARING (MULTIMODAL)
# =============================================================================

def analizar_clearing(historiales_clearing, nombre_caso=""):
    """
    Caja Blanca del AG Multimodal con Clearing: nº de dominantes (nichos) que
    sobreviven al clearing por generación. En lugar de superponer las 5 curvas
    (ruidosas e ilegibles), se resume entre semillas con la MEDIA, su banda de
    variabilidad (mín–máx y ±σ) y una media móvil para ver la tendencia. Un nivel
    alto y sostenido = población repartida en muchos nichos (multimodalidad).
    """
    series = [h.get('dominantes_por_generacion', [])
              for h in historiales_clearing if h.get('dominantes_por_generacion')]
    if not series:
        print(">> Clearing: sin datos de dominantes.")
        return

    long_min = min(len(s) for s in series)
    datos = np.array([s[:long_min] for s in series], dtype=float)
    gens = np.arange(long_min)
    media, desv = datos.mean(axis=0), datos.std(axis=0)
    mn, mx = datos.min(axis=0), datos.max(axis=0)

    # Media móvil (normalizada en los bordes para no hundir los extremos).
    w = max(1, min(7, long_min // 10))
    if w > 1:
        k = np.ones(w)
        media_suave = (np.convolve(media, k, mode='same') /
                       np.convolve(np.ones_like(media), k, mode='same'))
    else:
        media_suave = media

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(gens, mn, mx, color='steelblue', alpha=0.12,
                    label='Rango mín–máx entre semillas')
    ax.fill_between(gens, media - desv, media + desv, color='steelblue', alpha=0.25,
                    label='Media ± σ')
    ax.plot(gens, media, color='steelblue', alpha=0.35, linewidth=1)
    ax.plot(gens, media_suave, color='navy', linewidth=2.2,
            label=f'Media entre semillas (móvil w={w})')
    ax.axhline(media.mean(), color='red', linestyle='--', alpha=0.6,
               label=f'Media global = {media.mean():.1f}')

    ax.set_title(f'Clearing: dominantes (nichos) por generación {nombre_caso}'.strip(),
                 fontsize=14)
    ax.set_xlabel('Generación')
    ax.set_ylabel('Nº de dominantes (nichos) tras el clearing')
    ax.set_ylim(0, mx.max() + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.show()

    print(f">> Clearing aplicado en {long_min} generaciones | dominantes: media "
          f"global={media.mean():.1f} (rango {mn.min():.0f}–{mx.max():.0f}) sobre "
          f"{datos.shape[0]} semillas. Un nivel alto y estable indica que la población "
          f"se reparte de forma sostenida en múltiples nichos estructurales "
          f"(multimodalidad efectiva); no satura en un único óptimo.")

# =============================================================================
# TABLA 1.1 — RESULTADOS PARCIALES (POR ALGORITMO)
# =============================================================================

def generar_tabla_parcial(nombre_algo, datos_por_caso):
    """
    Genera la Tabla 1.1 (Resultados Parciales) de un algoritmo: filas Ej1..Ej5,
    Media y Desviación; columnas (#Ev, F OBJ, Kms, Entr) para cada uno de los
    3 casos. Marca en negrita el mejor resultado de cada columna.

    datos_por_caso: { 'Caso X': [ {'ev','fobj','kms','entropia'}, ... x5 ] }
    """
    casos = ['Caso 1', 'Caso 2', 'Caso 3']
    metricas = ['ev', 'fobj', 'kms', 'entropia']
    # Sentido de "mejor": para entropía, mayor es mejor; resto, menor es mejor.
    mejor_es_max = {'ev': False, 'fobj': False, 'kms': False, 'entropia': True}

    # Índice de la mejor ejecución por (caso, métrica) para resaltar en negrita.
    mejor_idx = {}
    for caso in casos:
        ejec = datos_por_caso.get(caso, [])
        for m in metricas:
            if not ejec:
                mejor_idx[(caso, m)] = None
                continue
            valores = [e[m] for e in ejec]
            mejor_idx[(caso, m)] = (np.argmax(valores) if mejor_es_max[m]
                                    else np.argmin(valores))

    html = """
    <style>
        .tabla-p3 { width:100%; border-collapse:collapse;
            font-family:Arial,sans-serif; margin-top:15px;
            background:#fff; color:#000; font-size:0.88em; }
        .tabla-p3 th { background:#e0e0e0; color:#000; font-weight:bold;
            text-align:center; padding:7px 4px; border:1px solid #444; }
        .tabla-p3 td { padding:7px 4px; text-align:center;
            border:1px solid #bbb; color:#111; }
        .tabla-p3 .borde-grueso { border-right:3px solid #222; }
        .tabla-p3 td.fila-name { font-weight:bold; background:#f5f5f5;
            border-left:3px solid #222; }
        .tabla-p3 tr.resumen td { background:#eef3ff; font-weight:bold; }
    </style>
    """
    html += f'<h3 style="color:inherit;">Tabla 1.1. Resultados Parciales — {nombre_algo}</h3>'
    html += '<table class="tabla-p3"><thead><tr>'
    html += '<th rowspan="2" style="border-left:3px solid #222;">Ejecuciones</th>'
    for caso in casos:
        html += f'<th colspan="4" class="borde-grueso">{caso}</th>'
    html += '</tr><tr>'
    for _ in casos:
        html += '<th>#Ev</th><th>F OBJ</th><th>Kms</th><th class="borde-grueso">Entr</th>'
    html += '</tr></thead><tbody>'

    def _celda(caso, m, idx, valor, ultima):
        clase = ' class="borde-grueso"' if m == 'entropia' else ''
        if m == 'ev':
            txt = f"{valor:.0f}"
        elif m == 'fobj' or m == 'entropia':
            txt = f"{valor:.4f}"
        else:
            txt = f"{valor:.2f}"
        if idx is not None and mejor_idx.get((caso, m)) == idx:
            txt = f"<b>{txt}</b>"
        return f"<td{clase}>{txt}</td>"

    n_ej = max((len(datos_por_caso.get(c, [])) for c in casos), default=0)
    for e in range(n_ej):
        html += f'<tr><td class="fila-name">Ej{e + 1}</td>'
        for caso in casos:
            ejec = datos_por_caso.get(caso, [])
            if e < len(ejec):
                for m in metricas:
                    html += _celda(caso, m, e, ejec[e][m], m == 'entropia')
            else:
                html += '<td>-</td><td>-</td><td>-</td><td class="borde-grueso">-</td>'
        html += '</tr>'

    for resumen, fn in (('Media', np.mean), ('Desviación', np.std)):
        html += f'<tr class="resumen"><td class="fila-name">{resumen}</td>'
        for caso in casos:
            ejec = datos_por_caso.get(caso, [])
            if ejec:
                for m in metricas:
                    val = fn([x[m] for x in ejec])
                    clase = ' class="borde-grueso"' if m == 'entropia' else ''
                    fmt = (f"{val:.0f}" if m == 'ev'
                           else f"{val:.4f}" if m in ('fobj', 'entropia')
                           else f"{val:.2f}")
                    html += f"<td{clase}>{fmt}</td>"
            else:
                html += '<td>-</td><td>-</td><td>-</td><td class="borde-grueso">-</td>'
        html += '</tr>'

    html += '</tbody></table>'
    display(HTML(html))

# =============================================================================
# TABLA 1.2 — RESULTADOS GLOBALES
# =============================================================================

def generar_tabla_global_p3(diccionario_resultados):
    """
    Genera la Tabla 1.2 (Resultados Globales): mejor resultado de cada
    algoritmo y caso. Filas: Greedy, P.Mejor, Básico, CHC, Multimodal.
    """
    html = """
    <style>
        .tabla-g3 { width:100%; border-collapse:collapse;
            font-family:Arial,sans-serif; margin-top:15px;
            background:#fff; color:#000; font-size:0.9em; }
        .tabla-g3 th { background:#e0e0e0; color:#000; font-weight:bold;
            text-align:center; padding:8px 4px; border:1px solid #444; }
        .tabla-g3 td { padding:8px 4px; text-align:center;
            border:1px solid #bbb; color:#111; }
        .tabla-g3 .borde-grueso { border-right:3px solid #222; }
        .tabla-g3 td.algo-name { font-weight:bold; background:#f5f5f5;
            border-left:3px solid #222; text-align:left; padding-left:10px; }
    </style>
    <h3 style="color:inherit;">Tabla 1.2. Resultados Globales</h3>
    <table class="tabla-g3"><thead><tr>
        <th rowspan="2" style="border-left:3px solid #222;">Modelo</th>
        <th colspan="4" class="borde-grueso">Caso 1</th>
        <th colspan="4" class="borde-grueso">Caso 2</th>
        <th colspan="4" class="borde-grueso">Caso 3</th>
    </tr><tr>
    """
    for _ in range(3):
        html += '<th>#Ev</th><th>F OBJ</th><th>Kms</th><th class="borde-grueso">Entr</th>'
    html += '</tr></thead><tbody>'

    orden_algoritmos = [
        ('Greedy', ['greedy']),
        ('P.Mejor', ['primer mejor', 'p.mejor']),
        ('Básico', ['basico', 'básico']),
        ('CHC', ['chc']),
        ('Multimodal', ['multimodal']),
    ]
    casos = ['Caso 1', 'Caso 2', 'Caso 3']

    def _clave(keywords):
        for k in diccionario_resultados.keys():
            if any(kw in k.lower() for kw in keywords):
                return k
        return None

    def _valores(datos):
        ev = datos.get('ev_media', datos.get('evaluaciones'))
        ev = float(ev) if isinstance(ev, (int, float, np.floating)) else None
        return {'ev': ev, 'fobj': datos.get('score_universal', datos.get('fobj')),
                'kms': datos.get('kms'), 'entropia': datos.get('entropia')}

    # Mejor por (caso, métrica) para resaltar en negrita (entropía=máx, resto=mín).
    mejor_es_max = {'ev': False, 'fobj': False, 'kms': False, 'entropia': True}
    mejor_val = {}
    for caso in casos:
        for m in ('ev', 'fobj', 'kms', 'entropia'):
            vals = []
            for _, kws in orden_algoritmos:
                k = _clave(kws)
                if k and caso in diccionario_resultados[k]:
                    v = _valores(diccionario_resultados[k][caso])[m]
                    if v is not None:
                        vals.append(v)
            if vals:
                mejor_val[(caso, m)] = max(vals) if mejor_es_max[m] else min(vals)

    def _celda(caso, m, valor, fmt, borde=False):
        clase = ' class="borde-grueso"' if borde else ''
        if valor is None:
            return f'<td{clase}>-</td>'
        txt = fmt(valor)
        if (caso, m) in mejor_val and valor == mejor_val[(caso, m)]:
            txt = f"<b>{txt}</b>"
        return f"<td{clase}>{txt}</td>"

    for nombre_display, keywords in orden_algoritmos:
        html += f'<tr><td class="algo-name">{nombre_display}</td>'
        clave_algo = _clave(keywords)
        for caso in casos:
            if clave_algo and caso in diccionario_resultados[clave_algo]:
                v = _valores(diccionario_resultados[clave_algo][caso])
                html += _celda(caso, 'ev', v['ev'], lambda x: f"{x:.1f}")
                html += _celda(caso, 'fobj', v['fobj'], lambda x: f"{x:.4f}")
                html += _celda(caso, 'kms', v['kms'], lambda x: f"{x:.2f}")
                html += _celda(caso, 'entropia', v['entropia'],
                               lambda x: f"{x:.4f}", borde=True)
            else:
                html += '<td>-</td><td>-</td><td>-</td><td class="borde-grueso">-</td>'
        html += '</tr>'

    html += '</tbody></table>'
    display(HTML(html))

# =============================================================================
# ESTUDIO PARAMÉTRICO — TABLA COMPARATIVA (estilo Tabla RCL de la Práctica 2)
# =============================================================================

def tabla_estudio_parametrico(nombre_algo, parametro, filas, columna_extra=None):
    """
    Tabla del estudio paramétrico: una fila por valor del parámetro barrido, con
    Media y Mejor F. Objetivo (fobj_ratio), desviación, CV y —opcionalmente— una
    métrica extra propia del algoritmo (reinicios en CHC, dominantes en Clearing).
    Marca en negrita la mejor Media y el mejor 'Mejor' (menor es mejor).

    filas: lista de dicts {'valor', 'media', 'best', 'sigma', 'cv', ['extra']}.
    """
    if not filas:
        display(HTML(f"<p>Sin datos para el estudio de <code>{parametro}</code>.</p>"))
        return

    idx_mejor_media = min(range(len(filas)), key=lambda i: filas[i]['media'])
    idx_mejor_best = min(range(len(filas)), key=lambda i: filas[i]['best'])

    html = """
    <style>
        .tabla-pp { width:75%; border-collapse:collapse;
            font-family:Arial,sans-serif; margin-top:15px;
            background:#fff; color:#000; font-size:0.9em; }
        .tabla-pp th { background:#e0e0e0; color:#000; font-weight:bold;
            text-align:center; padding:7px 8px; border:1px solid #444; }
        .tabla-pp td { padding:7px 8px; text-align:center;
            border:1px solid #bbb; color:#111; }
        .tabla-pp td.col-param { font-weight:bold; background:#f5f5f5;
            border-left:3px solid #222; }
    </style>
    """
    html += (f'<h3 style="color:inherit;">Estudio paramétrico — {nombre_algo} '
             f'(parámetro: <code>{parametro}</code>)</h3>')
    html += '<table class="tabla-pp"><thead><tr>'
    html += (f'<th>{parametro}</th><th>Media F OBJ</th><th>Mejor F OBJ</th>'
             f'<th>&sigma;</th><th>CV (%)</th>')
    if columna_extra:
        html += f'<th>{columna_extra}</th>'
    html += '</tr></thead><tbody>'

    for i, f in enumerate(filas):
        media_txt = f"{f['media']:.4f}"
        best_txt = f"{f['best']:.4f}"
        if i == idx_mejor_media:
            media_txt = f"<b>{media_txt}</b>"
        if i == idx_mejor_best:
            best_txt = f"<b>{best_txt}</b>"
        html += (f'<tr><td class="col-param">{f["valor"]}</td>'
                 f'<td>{media_txt}</td><td>{best_txt}</td>'
                 f'<td>{f["sigma"]:.4f}</td><td>{f["cv"]:.2f}</td>')
        if columna_extra:
            extra = f.get('extra')
            html += (f'<td>{extra:.1f}</td>' if extra is not None else '<td>-</td>')
        html += '</tr>'

    html += '</tbody></table>'
    display(HTML(html))
