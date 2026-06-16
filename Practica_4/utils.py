import sys
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

from Practica_1.utils import distancia_manhattan_km, evaluar_ruta

# =============================================================================
# MATRIZ DE DISTANCIAS Y VISIBILIDAD
# =============================================================================

def construir_matriz_distancias(coordenadas):
    """
    Matriz NxN de distancias Manhattan (km) entre TODAS las estaciones, incluida
    la 0 (depósito). Es la representación matricial que pide la consigna y la
    base tanto de la visibilidad como del coste del tour.
    """
    n = len(coordenadas)
    matriz = np.zeros((n, n))
    for i in range(n):
        lat1, lon1 = coordenadas[i]['lat'], coordenadas[i]['lon']
        for j in range(i + 1, n):
            lat2, lon2 = coordenadas[j]['lat'], coordenadas[j]['lon']
            d = distancia_manhattan_km(lat1, lon1, lat2, lon2)
            matriz[i][j] = matriz[j][i] = d
    return matriz


def calcular_visibilidad(matriz_dist):
    """
    Visibilidad η = 1 / distancia (heurística de la regla de transición). La
    diagonal se deja a 0 (una estación no viaja a sí misma).
    """
    with np.errstate(divide='ignore'):
        eta = np.where(matriz_dist > 0, 1.0 / matriz_dist, 0.0)
    np.fill_diagonal(eta, 0.0)
    return eta


def arcos_tour(ruta, inicio=0):
    """Arcos (i, j) del circuito cerrado: inicio→r0→…→r_last→inicio."""
    secuencia = [inicio] + list(ruta) + [inicio]
    return [(secuencia[k], secuencia[k + 1]) for k in range(len(secuencia) - 1)]

# =============================================================================
# EVALUACIÓN DEL TOUR (SIN RESTRICCIÓN DE CAPACIDAD)
# =============================================================================

def evaluar_tour_sin_capacidad(ruta, caso_bicis, caso_capacidad, coordenadas,
                               inicio=0):
    """
    Evalúa un tour reutilizando `evaluar_ruta` pero ELIMINANDO la restricción de
    capacidad del camión (consigna P4).

    Para que el camión NO esté limitado hace falta que en todo momento pueda tanto
    SOLTAR (carga > 0) como RECOGER (carga < tope). Por eso se arranca con una
    carga INTERMEDIA (5·S) y un tope muy alto (10·S), con S = suma de capacidades:
    así nunca se queda a 0 ni topa el máximo, toda estación llega a int(cap·0.5) y
    la entropía resulta CONSTANTE (independiente del orden) = entropía del sistema
    completamente equilibrado. El coste que guía a las hormigas es C(S) = kms.

    (Arrancar lleno —bicis_ini = l_max— sería un error: el camión no podría recoger
    hasta soltar algo, y el balanceo dependería del orden de visita.)
    """
    s = int(sum(caso_capacidad))
    return evaluar_ruta(ruta, caso_bicis, caso_capacidad, coordenadas,
                        inicio=inicio, l_max=10 * s + 1, bicis_ini=5 * s + 1)


def coste_tour(matriz_dist, ruta, inicio=0):
    """Kilómetros del circuito cerrado a partir de la matriz de distancias."""
    return float(sum(matriz_dist[i][j] for i, j in arcos_tour(ruta, inicio)))

# =============================================================================
# FEROMONA
# =============================================================================

def calcular_tau0(n, coste_greedy):
    """Feromona inicial τ0 = 1 / (n · L), con L = coste del circuito greedy."""
    if coste_greedy <= 0:
        return 1.0
    return 1.0 / (n * coste_greedy)


def inicializar_feromona(n, tau0):
    """Matriz de feromona homogénea τ0 (diagonal a 0)."""
    feromona = np.full((n, n), tau0, dtype=float)
    np.fill_diagonal(feromona, 0.0)
    return feromona

# =============================================================================
# REGLA DE TRANSICIÓN / CONSTRUCCIÓN DEL TOUR DE UNA HORMIGA
# =============================================================================

def construir_tour_hormiga(feromona, visibilidad, ciudades, inicio, alpha, beta,
                           rng, q0=None, phi=None, tau0=None):
    """
    Construye el tour de una hormiga: partiendo del depósito 'inicio', elige la
    siguiente estación entre las 'ciudades' no visitadas hasta recorrerlas todas.
    Devuelve la permutación de estaciones (sin el depósito; éste se añade en los
    extremos al evaluar el circuito cerrado).

    - SH / SHE: regla proporcional p_ij ∝ τ_ij^α · η_ij^β (q0=None).
    - SCH (ACS): regla pseudo-aleatoria proporcional — con probabilidad q0 se
      explota (argmax τ_ij · η_ij^β); en otro caso se explora con la proporcional.
      Si se pasan 'phi' y 'tau0' se aplica la actualización LOCAL de feromona
      τ_ij ← (1-φ)·τ_ij + φ·τ0 sobre cada arco recorrido.

    'rng' es un generador `random.Random` propio (reproducibilidad sin pisar el
    estado global de `random`).
    """
    no_visitadas = [c for c in ciudades if c != inicio]
    actual = inicio
    ruta = []

    while no_visitadas:
        pesos = []
        for j in no_visitadas:
            tau = feromona[actual][j] ** alpha
            eta = visibilidad[actual][j] ** beta
            pesos.append(tau * eta)
        total = sum(pesos)

        if total <= 0:
            siguiente = rng.choice(no_visitadas)
        elif q0 is not None and rng.random() < q0:
            # Explotación (SCH): mejor arco según τ·η^β.
            siguiente = max(no_visitadas,
                            key=lambda j: (feromona[actual][j] *
                                           visibilidad[actual][j] ** beta))
        else:
            # Exploración proporcional (ruleta).
            umbral = rng.random() * total
            acumulado = 0.0
            siguiente = no_visitadas[-1]
            for idx, j in enumerate(no_visitadas):
                acumulado += pesos[idx]
                if acumulado >= umbral:
                    siguiente = j
                    break

        # Actualización local de feromona (sólo SCH).
        if phi is not None and tau0 is not None:
            feromona[actual][siguiente] = ((1 - phi) * feromona[actual][siguiente]
                                           + phi * tau0)
            feromona[siguiente][actual] = feromona[actual][siguiente]

        ruta.append(siguiente)
        no_visitadas.remove(siguiente)
        actual = siguiente

    return ruta


def depositar_feromona(feromona, ruta, cantidad, inicio=0):
    """Suma 'cantidad' de feromona en los arcos del tour (simétrico)."""
    for i, j in arcos_tour(ruta, inicio):
        feromona[i][j] += cantidad
        feromona[j][i] += cantidad

# =============================================================================
# COMPARADOR: GREEDY (DETERMINISTA Y ALEATORIZADO DE LISTA CORTA)
# =============================================================================

def coste_greedy_determinista(matriz_dist, n, inicio=0):
    """
    Vecino más cercano determinista sobre todas las estaciones. Devuelve
    (ruta, coste). Su coste es la L empleada en τ0 = 1/(n·L).
    """
    pendientes = [c for c in range(n) if c != inicio]
    actual = inicio
    ruta = []
    while pendientes:
        siguiente = min(pendientes, key=lambda j: matriz_dist[actual][j])
        ruta.append(siguiente)
        pendientes.remove(siguiente)
        actual = siguiente
    return ruta, coste_tour(matriz_dist, ruta, inicio)


def greedy_aleatorizado_lista_corta(matriz_dist, n, semilla, tam_rcl=3, inicio=0):
    """
    Greedy aleatorizado de lista corta (consigna P4): en cada paso construye la
    RCL con las 'tam_rcl' estaciones no visitadas más cercanas y elige una al
    azar. Comparador estocástico de los algoritmos de hormigas.
    """
    rng = random.Random(semilla)
    pendientes = [c for c in range(n) if c != inicio]
    actual = inicio
    ruta = []
    while pendientes:
        candidatas = sorted(pendientes, key=lambda j: matriz_dist[actual][j])
        rcl = candidatas[:tam_rcl]
        siguiente = rng.choice(rcl)
        ruta.append(siguiente)
        pendientes.remove(siguiente)
        actual = siguiente
    return ruta

# =============================================================================
# TABLA 1.1 — RESULTADOS GLOBALES (SH / SHE / SCH / GREEDY)
# =============================================================================

def generar_tabla_global_aco(diccionario_resultados):
    """
    Genera la Tabla 1.1 de la consigna: filas SH, SHE, SCH y Greedy A.; columnas
    por caso × {#Ev, F OBJ, Kms, Entr} (mejor resultado de cada algoritmo y caso).
    Marca en negrita el mejor de cada columna (entropía: mayor es mejor; resto:
    menor es mejor).

    diccionario_resultados: { '<clave_algo>': { 'Caso X': {'ev','fobj','kms','entropia'} } }
    """
    orden_algoritmos = [
        ('SH', ['sh', 'sistema de hormigas']),
        ('SHE', ['she', 'elitista']),
        ('SCH', ['sch', 'colonias']),
        ('Greedy A.', ['greedy']),
    ]
    casos = ['Caso 1', 'Caso 2', 'Caso 3']
    mejor_es_max = {'ev': False, 'fobj': False, 'kms': False, 'entropia': True}

    def _clave(keywords):
        for k in diccionario_resultados.keys():
            if any(kw in k.lower() for kw in keywords):
                return k
        return None

    # Mejor valor por (caso, métrica) para resaltar en negrita.
    mejor_val = {}
    for caso in casos:
        for m in ('ev', 'fobj', 'kms', 'entropia'):
            vals = []
            for _, kws in orden_algoritmos:
                k = _clave(kws)
                if k and caso in diccionario_resultados.get(k, {}):
                    v = diccionario_resultados[k][caso].get(m)
                    if v is not None:
                        vals.append(v)
            if vals:
                mejor_val[(caso, m)] = max(vals) if mejor_es_max[m] else min(vals)

    html = """
    <style>
        .tabla-aco { width:100%; border-collapse:collapse;
            font-family:Arial,sans-serif; margin-top:15px;
            background:#fff; color:#000; font-size:0.9em; }
        .tabla-aco th { background:#e0e0e0; color:#000; font-weight:bold;
            text-align:center; padding:8px 4px; border:1px solid #444; }
        .tabla-aco td { padding:8px 4px; text-align:center;
            border:1px solid #bbb; color:#111; }
        .tabla-aco .borde-grueso { border-right:3px solid #222; }
        .tabla-aco td.algo-name { font-weight:bold; background:#f5f5f5;
            border-left:3px solid #222; text-align:left; padding-left:10px; }
    </style>
    <h3 style="color:inherit;">Tabla 1.1. Resultados Globales</h3>
    <table class="tabla-aco"><thead><tr>
        <th rowspan="2" style="border-left:3px solid #222;">Modelo</th>
        <th colspan="4" class="borde-grueso">Caso 1</th>
        <th colspan="4" class="borde-grueso">Caso 2</th>
        <th colspan="4" class="borde-grueso">Caso 3</th>
    </tr><tr>
    """
    for _ in casos:
        html += '<th>#Ev</th><th>F OBJ</th><th>Kms</th><th class="borde-grueso">Entr</th>'
    html += '</tr></thead><tbody>'

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
            datos = (diccionario_resultados.get(clave_algo, {}).get(caso)
                     if clave_algo else None)
            if datos:
                html += _celda(caso, 'ev', datos.get('ev'), lambda x: f"{x:.0f}")
                html += _celda(caso, 'fobj', datos.get('fobj'), lambda x: f"{x:.4f}")
                html += _celda(caso, 'kms', datos.get('kms'), lambda x: f"{x:.2f}")
                html += _celda(caso, 'entropia', datos.get('entropia'),
                               lambda x: f"{x:.4f}", borde=True)
            else:
                html += '<td>-</td><td>-</td><td>-</td><td class="borde-grueso">-</td>'
        html += '</tr>'

    html += '</tbody></table>'
    display(HTML(html))

# =============================================================================
# TABLA 1.1 — RESULTADOS PARCIALES (MEDIA Y DESVIACIÓN TÍPICA)
# =============================================================================

def generar_tabla_parcial_aco(nombre_algo, datos_por_caso):
    """
    Tabla de Resultados Parciales de un algoritmo: filas Ej1..Ejk + Media +
    Desviación; columnas {#Ev, F OBJ, Kms, Entr} para cada uno de los 3 casos.
    Cubre el requisito del PDF de reportar media y desviación típica. Marca en
    negrita el mejor resultado de cada columna (entropía: mayor es mejor; resto:
    menor es mejor).

    datos_por_caso: { 'Caso X': [ {'ev','fobj','kms','entropia'}, ... ] }
    """
    casos = ['Caso 1', 'Caso 2', 'Caso 3']
    metricas = ['ev', 'fobj', 'kms', 'entropia']
    mejor_es_max = {'ev': False, 'fobj': False, 'kms': False, 'entropia': True}

    # Índice de la mejor ejecución por (caso, métrica) para resaltar.
    mejor_idx = {}
    for caso in casos:
        ejec = datos_por_caso.get(caso, [])
        for m in metricas:
            if not ejec:
                mejor_idx[(caso, m)] = None
                continue
            valores = [e[m] for e in ejec]
            mejor_idx[(caso, m)] = (int(np.argmax(valores)) if mejor_es_max[m]
                                    else int(np.argmin(valores)))

    html = """
    <style>
        .tabla-pa { width:100%; border-collapse:collapse;
            font-family:Arial,sans-serif; margin-top:15px;
            background:#fff; color:#000; font-size:0.88em; }
        .tabla-pa th { background:#e0e0e0; color:#000; font-weight:bold;
            text-align:center; padding:7px 4px; border:1px solid #444; }
        .tabla-pa td { padding:7px 4px; text-align:center;
            border:1px solid #bbb; color:#111; }
        .tabla-pa .borde-grueso { border-right:3px solid #222; }
        .tabla-pa td.fila-name { font-weight:bold; background:#f5f5f5;
            border-left:3px solid #222; }
        .tabla-pa tr.resumen td { background:#eef3ff; font-weight:bold; }
    </style>
    """
    html += (f'<h3 style="color:inherit;">Tabla 1.1. Resultados Parciales — '
             f'{nombre_algo}</h3>')
    html += '<table class="tabla-pa"><thead><tr>'
    html += '<th rowspan="2" style="border-left:3px solid #222;">Ejecuciones</th>'
    for caso in casos:
        html += f'<th colspan="4" class="borde-grueso">{caso}</th>'
    html += '</tr><tr>'
    for _ in casos:
        html += '<th>#Ev</th><th>F OBJ</th><th>Kms</th><th class="borde-grueso">Entr</th>'
    html += '</tr></thead><tbody>'

    def _fmt(m, valor):
        if m == 'ev':
            return f"{valor:.0f}"
        if m in ('fobj', 'entropia'):
            return f"{valor:.4f}"
        return f"{valor:.2f}"

    def _celda(caso, m, idx, valor):
        clase = ' class="borde-grueso"' if m == 'entropia' else ''
        txt = _fmt(m, valor)
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
                    html += _celda(caso, m, e, ejec[e][m])
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
                    html += f"<td{clase}>{_fmt(m, val)}</td>"
            else:
                html += '<td>-</td><td>-</td><td>-</td><td class="borde-grueso">-</td>'
        html += '</tr>'

    html += '</tbody></table>'
    display(HTML(html))

# =============================================================================
# CONVERGENCIA
# =============================================================================

def graficar_convergencia_aco(datos_para_graficar, nombre_algoritmo=""):
    """
    Convergencia, una columna por caso:
    - 'Mejor global' (best-so-far): mejor coste encontrado hasta la iteración
      (línea sólida, monótona).
    - 'Mejor de la iteración': mejor hormiga de cada iteración (línea tenue), que
      revela la dinámica de exploración. En SH el best-so-far forma un "ángulo
      recto" (se estanca muy pronto por la concentración rápida de feromona: todas
      las hormigas refuerzan y la evaporación es baja), mientras esta curva muestra
      que la colonia sigue explorando sin mejorar el récord. SCH, en cambio,
      converge de forma gradual y su curva de iteración es casi plana (explotación).
    datos_para_graficar: lista de dicts {'historial', 'nombre_caso', 'semilla'}
    con 'iteracion', 'mejor_coste' y, si está presente, 'coste_iteracion'.
    """
    n = len(datos_para_graficar)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, dato in zip(axes, datos_para_graficar):
        h = dato['historial']
        it = h.get('iteracion', [])
        coste_it = h.get('coste_iteracion', [])
        if coste_it:
            ax.plot(it, coste_it, color='orange', alpha=0.55, linewidth=1,
                    label='Mejor de la iteración (exploración)')
        ax.plot(it, h.get('mejor_coste', []), color='navy', linewidth=2,
                label='Mejor global (best-so-far)')
        ax.set_title(f"{dato['nombre_caso']} (Semilla: {dato['semilla']})")
        ax.set_xlabel('Iteración')
        ax.set_ylabel('Kms del tour')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f'Convergencia de {nombre_algoritmo}'.strip(), fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()

# =============================================================================
# ANIMACIÓN INTERACTIVA DEL RASTRO DE FEROMONA
# =============================================================================

def _intensidad_arista(valor, baseline, escala):
    """
    Intensidad normalizada [0,1] de una arista respecto a la baseline (τ0) y la
    escala (τ_max − τ0). Restar la baseline evita pintar las aristas "suelo"
    (congeladas en τ0, sin reforzar) y hace que la traza emerja de forma gradual.
    """
    if escala <= 0:
        return 0.0
    return max(0.0, min(1.0, (valor - baseline) / escala))


def dibujar_mapa_feromona(coordenadas, feromona, mejor_ruta=None, inicio=0,
                          tau0=None, umbral=0.08, grosor_max=9.0):
    """
    Mapa Folium con el rastro de feromona: una línea por arista REFORZADA (τ por
    encima de la baseline τ0), con grosor/opacidad ∝ a su intensidad relativa. Si
    se pasa 'mejor_ruta', su circuito se resalta en rojo por encima del rastro.
    Requiere `folium` (ya es dependencia base del proyecto).
    """
    import folium
    mapa = folium.Map(location=[43.4647, -3.8044], zoom_start=14)

    positivos = feromona[feromona > 0]
    baseline = tau0 if tau0 is not None else (float(positivos.min())
                                              if positivos.size else 0.0)
    tau_max = float(np.max(feromona)) if np.max(feromona) > 0 else 1.0
    escala = tau_max - baseline
    n = len(coordenadas)
    for i in range(n):
        for j in range(i + 1, n):
            intensidad = _intensidad_arista(feromona[i][j], baseline, escala)
            if intensidad < umbral:
                continue
            folium.PolyLine(
                locations=[[coordenadas[i]['lat'], coordenadas[i]['lon']],
                           [coordenadas[j]['lat'], coordenadas[j]['lon']]],
                color='#6a0dad', weight=max(0.5, grosor_max * intensidad),
                opacity=min(0.9, 0.15 + 0.85 * intensidad),
            ).add_to(mapa)

    if mejor_ruta is not None:
        for i, j in arcos_tour(mejor_ruta, inicio):
            folium.PolyLine(
                locations=[[coordenadas[i]['lat'], coordenadas[i]['lon']],
                           [coordenadas[j]['lat'], coordenadas[j]['lon']]],
                color='red', weight=3, opacity=0.9,
            ).add_to(mapa)

    for i, c in enumerate(coordenadas):
        folium.CircleMarker(
            location=[c['lat'], c['lon']], radius=5,
            color='black', fill=True, fill_color='white', fill_opacity=0.9,
            popup=f"Estación {i}",
        ).add_to(mapa)

    return mapa


def generar_animacion_feromona(coordenadas, snapshots, inicio=0, umbral=0.08,
                               grosor_max=9.0, periodo='PT1M', guardar_html=None):
    """
    Animación INTERACTIVA del rastro de feromona sobre el mapa real de Santander,
    con barra de tiempo (slider) y botón de play. Cada snapshot del historial
    (campo 'snapshots_feromona', generado con registrar_feromona=True) se convierte
    en un fotograma temporal: las aristas se dibujan con grosor/opacidad ∝ τ
    normalizado y el mejor tour de ese instante se resalta en rojo.

    Usa `folium.plugins.TimestampedGeoJson` (incluido en folium): NO requiere
    selenium ni navegador externo y soporta tantos fotogramas como snapshots haya.
    Devuelve el mapa Folium (se muestra inline en el notebook); si se pasa
    'guardar_html', también lo guarda en disco.

    'periodo' es la duración de cada fotograma en formato ISO-8601 (p. ej. 'PT1M').
    'umbral' descarta aristas con feromona normalizada por debajo de ese valor
    (mantiene el mapa legible).
    """
    import datetime
    import folium
    from folium.plugins import TimestampedGeoJson

    if not snapshots:
        print(">> [Animación] No hay snapshots de feromona. Ejecuta el algoritmo "
              "con registrar_feromona=True.")
        return None

    mapa = folium.Map(location=[43.4647, -3.8044], zoom_start=14)

    # Estaciones como capa estática (siempre visibles).
    for i, c in enumerate(coordenadas):
        folium.CircleMarker(
            location=[c['lat'], c['lon']], radius=5, color='black', weight=1,
            fill=True, fill_color='white', fill_opacity=0.9,
            popup=f"Estación {i}",
        ).add_to(mapa)

    # Escala GLOBAL y baseline τ0 comunes a todos los fotogramas: así la traza
    # crece de forma comparable entre frames y el arranque sale limpio (las
    # aristas-suelo congeladas en τ0 dan intensidad 0 y no se dibujan), en vez de
    # "todo lleno → vacío de golpe" que produce el normalizar por el máx de cada frame.
    baseline = snapshots[0].get('tau0')
    if baseline is None:
        positivos = np.concatenate([s['feromona'][s['feromona'] > 0].ravel()
                                    for s in snapshots])
        baseline = float(positivos.min()) if positivos.size else 0.0
    global_max = max(float(np.max(s['feromona'])) for s in snapshots)
    escala = global_max - baseline

    base = datetime.datetime(2020, 1, 1)
    n = len(coordenadas)
    features = []
    for idx, snap in enumerate(snapshots):
        feromona = snap['feromona']
        t_iso = (base + datetime.timedelta(minutes=idx)).isoformat()

        # Aristas del rastro de feromona (grosor/opacidad ∝ intensidad relativa).
        for i in range(n):
            for j in range(i + 1, n):
                intensidad = _intensidad_arista(feromona[i][j], baseline, escala)
                if intensidad < umbral:
                    continue
                features.append({
                    'type': 'Feature',
                    'geometry': {'type': 'LineString', 'coordinates': [
                        [coordenadas[i]['lon'], coordenadas[i]['lat']],
                        [coordenadas[j]['lon'], coordenadas[j]['lat']]]},
                    'properties': {
                        'times': [t_iso, t_iso],
                        'style': {'color': '#6a0dad',
                                  'weight': max(0.5, grosor_max * intensidad),
                                  'opacity': min(0.9, 0.15 + 0.85 * intensidad)},
                    },
                })

        # Mejor tour del instante, resaltado en rojo.
        mejor_ruta = snap.get('mejor_ruta')
        if mejor_ruta:
            coords_tour = [[coordenadas[a]['lon'], coordenadas[a]['lat']]
                           for a, _ in arcos_tour(mejor_ruta, inicio)]
            coords_tour.append([coordenadas[inicio]['lon'],
                                coordenadas[inicio]['lat']])
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'LineString', 'coordinates': coords_tour},
                'properties': {
                    'times': [t_iso] * len(coords_tour),
                    'style': {'color': 'red', 'weight': 3, 'opacity': 0.9},
                },
            })

    TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period=periodo, duration=periodo, transition_time=300,
        auto_play=False, loop=False, add_last_point=False,
        date_options='YYYY-MM-DD HH:mm',
    ).add_to(mapa)

    if guardar_html:
        mapa.save(guardar_html)
        print(f">> [Animación] Mapa interactivo guardado en: {guardar_html}")
    return mapa
