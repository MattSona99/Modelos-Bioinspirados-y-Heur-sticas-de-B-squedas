import json
import math
import random
import matplotlib.pyplot as plt

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
    
    def procesar_intercambio(id_est):
        """Procesa el intercambio de bicicletas en una estación dada."""
        nonlocal camion_bicis
        objetivo = int(caso_capacidad[id_est] * 0.5)
        if bicis_actuales[id_est] > objetivo:
            recogidas = min(bicis_actuales[id_est] - objetivo, l_max - camion_bicis)
            bicis_actuales[id_est] -= recogidas
            camion_bicis += recogidas
        elif bicis_actuales[id_est] < objetivo:
            dejadas = min(objetivo - bicis_actuales[id_est], camion_bicis)
            bicis_actuales[id_est] += dejadas
            camion_bicis -= dejadas
            
    procesar_intercambio(inicio)
    for proxima in ruta:
        lat1, lon1 = coordenadas[estacion_actual]['lat'], coordenadas[estacion_actual]['lon']
        lat2, lon2 = coordenadas[proxima]['lat'], coordenadas[proxima]['lon']
        distancia_total += distancia_manhattan_km(lat1, lon1, lat2, lon2)
        estacion_actual = proxima
        procesar_intercambio(estacion_actual)
        
    lat1, lon1 = coordenadas[estacion_actual]['lat'], coordenadas[estacion_actual]['lon']
    lat2, lon2 = coordenadas[inicio]['lat'], coordenadas[inicio]['lon']
    distancia_total += distancia_manhattan_km(lat1, lon1, lat2, lon2)
    
    return {
        'kms_recorridos': distancia_total,
        'entropia': calcular_entropia(bicis_actuales, caso_capacidad)
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
    
def graficar_historiales(datos_graficas, algoritmo):
    """
    Recibe una lista de diccionarios con los historiales de los casos
    y los dibuja uno al lado del otro.
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

        ax1.set_title(f'{nombre_caso} (Semilla: {semilla})')
        ax1.grid(True, alpha=0.3)

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    fig.suptitle(f'Evolución de {algoritmo}', fontsize=16, y=1.05)

    plt.tight_layout()
    plt.show()
    
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