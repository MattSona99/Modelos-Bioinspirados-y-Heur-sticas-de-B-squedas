import json
import math
import random
import matplotlib.pyplot as plt
import folium
from IPython.display import display, HTML

def cargar_coordenadas(ruta_archivio):
    """Carga las coordenadas de las estaciones desde un archivo JSON."""
    with open(ruta_archivio, 'r') as f:
        return json.load(f)

def distancia_manhattan_km(lat1, lon1, lat2, lon2):
    """Calcula la distancia Manhattan en kilómetros."""
    R = 6371.0
    dlat_rad = math.radians(abs(lat2 - lat1))
    dlon_rad = math.radians(abs(lon2 - lon1))
    lat_media_rad = math.radians((lat1 + lat2) / 2.0)
    return R * dlat_rad + R * dlon_rad * math.cos(lat_media_rad)

def calcular_entropia(b, c):
    """Calcula la entropía de una distribución."""
    entropia = 0.0
    for bi, ci in zip(b, c):
        if bi == 0 or bi == ci: continue
        p = bi / ci
        entropia += -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return entropia

def obtener_estaciones_a_visitar(caso_bicis, caso_capacidad, tolerancia):
    """Obtiene las estaciones que deben ser visitadas."""
    estaciones_validas = []
    for i in range(1, len(caso_bicis)):
        objetivo = int(caso_capacidad[i] * 0.5)
        if abs(caso_bicis[i] - objetivo) > tolerancia:
            estaciones_validas.append(i)
    return estaciones_validas

def generar_solucion_inicial_aleatoria(estaciones_base):
    """Genera una solución inicial aleatoria a partir de las estaciones base."""
    solucion = list(estaciones_base)
    random.shuffle(solucion)
    return solucion

def generar_vecino_swap(ruta, i, j):
    """Genera una nueva ruta intercambiando dos posiciones."""
    nueva_ruta = list(ruta)
    nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]
    return nueva_ruta

def evaluar_ruta(ruta, caso_bicis, caso_capacidad, coordenadas, inicio=0, l_max=20, bicis_ini=7):
    """Evalua la calidad de una ruta."""
    bicis_actuales = list(caso_bicis)
    distancia_total = 0.0
    camion_bicis = bicis_ini
    estacion_actual = inicio
    movimientos_mapa = []
    
    def procesar_intercambio(id_est):
        """Procesa el intercambio de bicicletas en una estación dada."""
        nonlocal camion_bicis
        objetivo = int(caso_capacidad[id_est] * 0.5)
        accion = 'nada'
        cantidad = 0
        
        if bicis_actuales[id_est] > objetivo:
            recogidas = min(bicis_actuales[id_est] - objetivo, l_max - camion_bicis)
            bicis_actuales[id_est] -= recogidas
            camion_bicis += recogidas
            accion = 'coger'
            cantidad = recogidas
            
        elif bicis_actuales[id_est] < objetivo:
            dejadas = min(objetivo - bicis_actuales[id_est], camion_bicis)
            bicis_actuales[id_est] += dejadas
            camion_bicis -= dejadas
            accion = 'dejar'
            cantidad = dejadas
            
        return accion, cantidad
            
    procesar_intercambio(inicio)
    
    for proxima in ruta:
        lat1, lon1 = coordenadas[estacion_actual]['lat'], coordenadas[estacion_actual]['lon']
        lat2, lon2 = coordenadas[proxima]['lat'], coordenadas[proxima]['lon']
        distancia_total += distancia_manhattan_km(lat1, lon1, lat2, lon2)
        
        accion, cantidad = procesar_intercambio(proxima)
        movimientos_mapa.append((estacion_actual, proxima, accion, cantidad))
        
        estacion_actual = proxima
        
    lat1, lon1 = coordenadas[estacion_actual]['lat'], coordenadas[estacion_actual]['lon']
    lat2, lon2 = coordenadas[inicio]['lat'], coordenadas[inicio]['lon']
    distancia_total += distancia_manhattan_km(lat1, lon1, lat2, lon2)
    
    movimientos_mapa.append((estacion_actual, inicio, 'nada', 0))
    
    return {
        'kms_recorridos': distancia_total,
        'entropia': calcular_entropia(bicis_actuales, caso_capacidad),
        'inventario_final': bicis_actuales,
        'movimientos_mapa': movimientos_mapa
    }
    
def calibrar_mu_phi(ruta_base, fobj_base, coordenadas, caso_bicis, caso_capacidad,
                    evaluar_ruta, funcion_objetivo, num_vecinos=100, **kwargs):
    """Calibra los parámetros mu y phi para encontras los mejores parámetros
    que logren aproximadamente un 20% de rechazo en la temperatura inicial T0."""
    print("\nCalibrando parámetros mu y phi...")
    print(f"Costo inicial Greedy C(Si): {fobj_base:.4f}")
    
    # Valores a probar para mu y phi (entre 0.1 y 0.3)
    valores_mu = [0.1, 0.15, 0.2, 0.25, 0.3]
    valores_phi = [0.1, 0.15, 0.2, 0.25, 0.3]
    n = len(ruta_base)
    deltas_vecinos = []
    
    # Pre-generar vecinos y calcular sus deltas reales
    for _ in range(num_vecinos):
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)
        vecino = generar_vecino_swap(ruta_base, i, j)
        
        res_vecino = evaluar_ruta(vecino, caso_bicis, caso_capacidad, coordenadas)
        fobj_vecino = funcion_objetivo(kms=res_vecino['kms_recorridos'], entropia=res_vecino['entropia'], **kwargs)
        delta_real = fobj_vecino - fobj_base
        deltas_vecinos.append(delta_real)
        
    mejor_combinacion = None
    mejor_diferencia = float('inf')
    
    print(f"| {'μ (mu)':<6} | {'Φ (phi)':<7} | {'T0 Inicial':<10} | {'% Rechazo':<10} | {'Dif. al 20%':<11} |")
    print("-" * 63)
    
    # Experimentar con todas las combinaciones de mu y phi
    for mu in valores_mu:
        for phi in valores_phi:
            T0 = (mu / -math.log(phi)) * abs(fobj_base) if fobj_base != float('inf') else 100.0
            
            rechazados = 0
            for delta in deltas_vecinos:
                if delta < 0:
                    pass # Siempre se acepta una mejora
                else:
                    # Se acepta con probabilidad exp(-delta / T0)
                    prob_aceptacion = math.exp(-delta / T0) if T0 > 0 else 0
                    if random.random() >= prob_aceptacion:
                        rechazados += 1
                        
            porcentaje_rechazo = (rechazados / num_vecinos) * 100
            diferencia = abs(porcentaje_rechazo - 20)
            
            
            print(f"| {mu:<6.2f} | {phi:<7.2f} | {T0:<10.4f} | {porcentaje_rechazo:>8.1f}% | {diferencia:>10.1f}% |")
            
            # Guardar la combinación que más se acerca al 20% de rechazo
            if diferencia < mejor_diferencia:
                mejor_diferencia = diferencia
                mejor_combinacion = {'mu': mu, 'phi': phi, 'T0': T0, 'rechazo': porcentaje_rechazo}
                
    print("-" * 63)
    print(f">> PARÁMETROS ÓPTIMOS ENCONTRADOS: μ = {mejor_combinacion['mu']} y Φ = {mejor_combinacion['phi']}")
    print(f">> Generan un rechazo del {mejor_combinacion['rechazo']}% (T0 = {mejor_combinacion['T0']:.4f})")
    
    return mejor_combinacion['mu'], mejor_combinacion['phi']

def generar_greedy_probabilistico(frecuencia, estaciones_base):
    """
    Construye una solución greedy probabilística basándose en la memoria a largo plazo.
    Aplica la "Selección por Ruleta Inversa de Frecuencias" sobre el Top 5 de estaciones.
    """
    ruta_greedy = []
    estaciones_pendientes = list(estaciones_base)
    estacion_actual = 0
    
    while estaciones_pendientes:
        # Identificar y ordenar candidatas por frecuencia
        candidatas_con_freq = []
        for candidata in estaciones_pendientes:
            freq = frecuencia[estacion_actual][candidata]
            candidatas_con_freq.append((candidata, freq))
            
        # Ordenar de menor a mayor frecuencia
        candidatas_con_freq.sort(key=lambda x: x[1]) 
        
        # Crear la Lista Restringida (Top 5)
        top_5 = candidatas_con_freq[:5]
        
        # Calcular el "Peso Inverso"
        pesos = []
        for candidata, freq in top_5:
            # Fórmula: W_i = 1 / (F_i + 1)
            peso = 1.0 / (freq + 1.0)
            pesos.append(peso)
            
        # Calcular las probabilidades (La Ruleta)
        peso_total = sum(pesos)
        probabilidades = [p / peso_total for p in pesos]
        
        # Tirar el dado
        numero = random.random()
        suma_acumulada = 0.0
        estacion_elegida = top_5[-1][0] # Fallback por seguridad de decimales
        
        for i, (candidata, _) in enumerate(top_5):
            suma_acumulada += probabilidades[i]
            if numero <= suma_acumulada:
                estacion_elegida = candidata
                break
                
        ruta_greedy.append(estacion_elegida)
        estaciones_pendientes.remove(estacion_elegida)
        estacion_actual = estacion_elegida
        
    return ruta_greedy
    
def graficar_historiales(datos_graficas, algoritmo):
    """
    Recibe una lista de diccionarios con los historiales de los casos
    y los dibuja uno al lado del otro, incluyendo los valores finales.
    """
    n_graficas = len(datos_graficas)
    fig, axes = plt.subplots(1, n_graficas, figsize=(16, 5))
    
    if n_graficas == 1:
        axes = [axes]

    color_fobj = 'tab:blue'
    color_ent = 'tab:green'
    color_kms = 'tab:red'

    for i, dato in enumerate(datos_graficas):
        ax1 = axes[i]
        historial = dato['historial']
        nombre_caso = dato['nombre_caso']
        semilla = dato['semilla']

        ax1.set_xlabel('Iteraciones')
        if i == 0:
            ax1.set_ylabel('F. Objetivo / Entropía', color='black')
        
        line1, = ax1.plot(historial['iteracion'], historial['fobj'], 
                          color=color_fobj, marker='o', markersize=4, 
                          label='Mejor F. Objetivo', linewidth=2)
        
        line2, = ax1.plot(historial['iteracion'], historial['entropia'], 
                          color=color_ent, linestyle='--', 
                          label='Entropía', linewidth=2)
        
        ax1.tick_params(axis='y', labelcolor='black')
        
        ax2 = ax1.twinx()  
        
        if i == n_graficas - 1:
            ax2.set_ylabel('Kilómetros (Kms)', color=color_kms)  
        
        line3, = ax2.plot(historial['iteracion'], historial['kms'], 
                          color=color_kms, linestyle=':', 
                          label='Kms Recorridos', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color_kms)

        ultima_iter = historial['iteracion'][-1]
        ultima_ent = historial['entropia'][-1]
        ultimo_kms = historial['kms'][-1]
        
        ax1.set_xlim(left=-2, right=ultima_iter + (ultima_iter * 0.15))

        ax1.annotate(f'{ultima_ent:.2f}', 
                     xy=(ultima_iter, ultima_ent),
                     xytext=(8, 0),
                     textcoords='offset points',
                     color=color_ent, fontweight='bold', fontsize=11)

        ax2.annotate(f'{ultimo_kms:.2f} Km', 
                     xy=(ultima_iter, ultimo_kms),
                     xytext=(8, -10),
                     textcoords='offset points',
                     color=color_kms, fontweight='bold', fontsize=11)
        # ------------------------------------------------------------

        ax1.set_title(f'{nombre_caso} (Semilla: {semilla})')
        ax1.grid(True, alpha=0.3)

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    fig.suptitle(f'Evolución de {algoritmo}', fontsize=16, y=1.05)

    plt.tight_layout()
    plt.show()

def dibujar_mapa_trayecto(coordenadas, movimientos_mapa):
    """
    Genera la Mappa dl Trayecto del Camión.
    Líneas de colores según la acción: Rojo (coger), Azul (dejar), Verde (ningún cambio).
    Las estaciones no visitadas no intervienen en la ruta.
    """
    mapa = folium.Map(location=[43.4647, -3.8044], zoom_start=14)
    
    # Dibujar todas las estaciones como puntos base pequeños para contexto geográfico
    for i, coords in enumerate(coordenadas):
        folium.CircleMarker(
            location=[coords['lat'], coords['lon']],
            radius=3, color='gray', fill=True, popup=f"Estación {i}"
        ).add_to(mapa)

    # Dibujar el Recorrido del Camión y pintar el nodo según la acción en ese instante
    for origen, destino, accion, cantidad in movimientos_mapa:
        origen_idx = int(origen)
        destino_idx = int(destino)
        
        lat_origen, lon_origen = coordenadas[origen_idx]['lat'], coordenadas[origen_idx]['lon']
        lat_destino, lon_destino = coordenadas[destino_idx]['lat'], coordenadas[destino_idx]['lon']

        # Determinar el color de la línea y del nodo destino de la ruta
        color_accion = 'green' # verde para ningún cambio
        if accion == 'coger':
            color_accion = 'red' # rojo para coger
        elif accion == 'dejar':
            color_accion = 'blue' # azul para dejar
            
        texto_linea = f"De {origen_idx} a {destino_idx}<br>Acción: {accion} {cantidad} bicis"
        
        # Dibujar la línea de la ruta
        folium.PolyLine(
            locations=[[lat_origen, lon_origen], [lat_destino, lon_destino]],
            color=color_accion,
            weight=4,
            opacity=0.8,
            popup=texto_linea
        ).add_to(mapa)
        
        # Dibujar el nodo destino resaltado con el color de la acción que se hizo allí
        folium.CircleMarker(
            location=[lat_destino, lon_destino],
            radius=6,
            color='black', # Borde negro para distinguir los nodos visitados
            weight=1,
            fill=True,
            fill_color=color_accion,
            fill_opacity=0.9,
            popup=f"Estación {destino_idx} (Visitada)<br>{accion.capitalize()} {cantidad} bicis"
        ).add_to(mapa)

    return mapa


def dibujar_mapa_estado(coordenadas, caso_capacidad, inventario_final):
    """
    Genera la Mappa del Comparativo del Porcentaje de Llenado, sin Trayecto.
    Azul (>50%), Rojo (<50%), Verde (50% exacto).
    """
    mapa = folium.Map(location=[43.4647, -3.8044], zoom_start=14)
    
    for i, coords in enumerate(coordenadas):
        lat, lon = coords['lat'], coords['lon']
        cap_max = caso_capacidad[i]
        bicis_actuales = inventario_final[i]
        porcentaje_llenado = (bicis_actuales / cap_max) * 100 if cap_max > 0 else 0
        
        color_circulo = 'green' # Punto Verde si están justos al 50%
        radio = 5 # Tamaño base mínimo
        
        # La tolerancia para el 50% se ajusta a un margen estrecho debido a los decimales
        if porcentaje_llenado > 51:
            color_circulo = 'blue' # Azul % por encima de 50%
            diferencia = porcentaje_llenado - 50
            radio = 5 + (diferencia * 0.3) # Multiplicador para hacer más visible la proporción
        elif porcentaje_llenado < 49:
            color_circulo = 'red' # Rojo % por debajo del 50%
            diferencia = 50 - porcentaje_llenado
            radio = 5 + (diferencia * 0.3)
            
        tooltip_texto = f"Estación {i}<br>Bicis: {bicis_actuales}/{cap_max} ({porcentaje_llenado:.1f}%)"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=radio,
            popup=tooltip_texto,
            color=color_circulo,
            fill=True,
            fill_color=color_circulo,
            fill_opacity=0.7
        ).add_to(mapa)

    return mapa

def generar_tabla_global(diccionario_resultados):
    """
    Genera una única Tabla Global resumen de Resultados.
    """
    html = """
    <style>
        .tabla-resultados { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: Arial, sans-serif; 
            margin-top: 15px; 
            background-color: #ffffff; 
            color: #000000; 
        }
        .tabla-resultados th { 
            background-color: #e0e0e0; 
            color: #000000; 
            font-weight: bold; 
            text-align: center; 
            padding: 12px 8px; 
            border-bottom: 3px solid #222222; 
            border-top: 3px solid #222222; 
        }
        .tabla-resultados td { 
            padding: 10px 8px; 
            text-align: center; 
            border-bottom: 1px solid #bbbbbb; 
            color: #111111;
        }
        .tabla-resultados tr.borde-grueso td { 
            border-bottom: 3px solid #222222; 
        }
        .tabla-resultados td.algo-name { 
            font-weight: bold; 
            vertical-align: middle; 
            border-right: 2px solid #bbbbbb; 
            background-color: #f5f5f5; 
            color: #000000;
        }
    </style>
    
    <h3 style="color: inherit;">Tabla Global Resumen de Resultados</h3>
    <table class="tabla-resultados">
        <thead>
            <tr>
                <th>Algoritmo</th><th>Caso</th><th>Ev. Medias</th><th>Ev. Mejor</th><th>σ EV</th><th>Mejor Kms</th><th>Mejor Entropía</th><th>F. Objetivo</th>
            </tr>
        </thead>
        <tbody>
    """
    
    orden_algoritmos = ['Greedy Algorithm', 'Búsqueda Aleatoria', 'Búsqueda Local Primer Mejor', 
                        'Búsqueda Local Mejor Vecino', 'Enfriamiento Simulado', 'Búsqueda Tabú']
    
    casos = ['Caso 1', 'Caso 2', 'Caso 3']
    
    for nombre_algo in orden_algoritmos:
        clave_algo = next((k for k in diccionario_resultados.keys() if k.lower().replace('ú', 'u') in nombre_algo.lower().replace('ú', 'u')), None)
        
        if clave_algo:
            casos_presentes = [c for c in casos if c in diccionario_resultados[clave_algo]]
            n_casos = len(casos_presentes)
            
            if n_casos == 0: continue
            
            is_greedy = 'greedy' in nombre_algo.lower()
            nombre_corto = nombre_algo.replace(' Algorithm', '').replace('Búsqueda ', 'B. ').replace('Local ', 'L. ').replace('Enfriamiento', 'Enf.')
            
            for i, caso in enumerate(casos_presentes):
                datos = diccionario_resultados[clave_algo][caso]
                
                # Extracción de datos
                ev_media = datos.get('evaluaciones', 0) if is_greedy else datos.get('ev_media', datos.get('evaluaciones', 0))
                ev_mejor = datos.get('evaluaciones', 0)
                sigma_ev = 0.0 if is_greedy else datos.get('sigma_ev', 0.0) 
                
                kms = datos['kms']
                entropia = datos['entropia']
                fobj_name = datos.get('nombre_fobj', 'N/A').replace('fobj_', '')[:10]
                fobj_val = datos['fobj']
                
                fobj_str = f"{fobj_val:.4f}<br><span style='font-size:0.85em; font-style:italic; color:#555;'>({fobj_name})</span>"
                
                clase_fila = "borde-grueso" if i == n_casos - 1 else ""
                
                html += f'<tr class="{clase_fila}">'
                
                if i == 0:
                    html += f'<td rowspan="{n_casos}" class="algo-name">{nombre_corto}</td>'
                    
                html += f"<td>{caso}</td>"
                html += f"<td>{ev_media:.1f}</td>"
                html += f"<td>{ev_mejor}</td>"
                html += f"<td>{sigma_ev:.2f}</td>"
                html += f"<td>{kms:.2f}</td>"
                html += f"<td>{entropia:.4f}</td>"
                html += f"<td>{fobj_str}</td>"
                html += "</tr>\n"

    html += "</tbody></table>"
    display(HTML(html))

# ---------------------------------------------------------------------------------------
# --------------------------------- Funciones Objetivos ---------------------------------
# ---------------------------------------------------------------------------------------

def fobj_ratio(kms, entropia, **kwargs):
    """Ratio simple: penaliza fuertemente la baja entropia."""
    return kms / entropia if entropia > 0 else float('inf')

def fobj_suma_ponderada(kms, entropia, alpha=2.0, **kwargs):
    """Suma ponderada: combina ambos factores con un peso alpha para la entropía."""
    return kms - (alpha * entropia)

def fobj_ratio_cuadratico(kms, entropia, **kwargs):
    """Ratio cuadrático: penaliza aún más la baja entropía."""
    return kms / (entropia ** 2) if entropia > 0 else float('inf')

def fobj_exponencial(kms, entropia, beta=0.1, **kwargs):
    """Exponencial decreciente: suaviza la penalización cuando la entropía es muy alta."""
    return kms * math.exp(-beta * entropia)

FUNCIONES_OBJETIVO = [fobj_ratio, fobj_suma_ponderada, fobj_ratio_cuadratico, fobj_exponencial]