import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

from Practica_1.utils import distancia_manhattan_km

# =============================================================================
# FUNCIONES ALGORÍTMICAS AUXILIARES
# =============================================================================

def construir_solucion_grasp(
    estaciones_base,
    coordenadas,
    caso_bicis,
    caso_capacidad,
    tamano_rcl=3
    ):
    """
    Fase 1 de GRASP: Construye una ruta iterativamente utilizando una Lista Restringida 
    de Candidatos (RCL) basada en una heurística de potencial (Necesidad / Distancia).
    """
    ruta_construida = []
    pendientes = set(estaciones_base)
    estacion_actual = 0
    
    while pendientes:
        candidatos = []
        lat_act, lon_act = coordenadas[estacion_actual]['lat'], coordenadas[estacion_actual]['lon']

        # Iteración ordenada para garantizar reproducibilidad algorítmica con random.seed
        for cand in sorted(pendientes):
            lat_cand, lon_cand = coordenadas[cand]['lat'], coordenadas[cand]['lon']
            dist = distancia_manhattan_km(lat_act, lon_act, lat_cand, lon_cand)
            if dist == 0: dist = 0.0001
            
            objetivo = caso_capacidad[cand] / 2.0
            necesidad = abs(caso_bicis[cand] - objetivo)
            potencial = necesidad / dist
            
            candidatos.append((cand, potencial))
        
        candidatos.sort(key=lambda x: x[1], reverse=True)
        
        rcl = candidatos[:tamano_rcl]
        
        suma_potenciales = sum(c[1] for c in rcl)
        if suma_potenciales > 0:
            pesos = [c[1] / suma_potenciales for c in rcl]
        else:
            pesos = [1.0] * len(rcl)
            
        estacion_elegida = random.choices([c[0] for c in rcl], weights=pesos, k=1)[0]
        
        ruta_construida.append(estacion_elegida)
        pendientes.remove(estacion_elegida)
        estacion_actual = estacion_elegida
    
    return ruta_construida

def aplicar_busqueda_local_primer_mejor(
    ruta_base, fobj_base, res_base,
    evaluar_ruta, funcion_objetivo,
    caso_bicis, caso_capacidad, coordenadas,
    evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
):
    """
    Motor de optimización de Búsqueda Local (Primer Mejor).
    Extraído para ser utilizado modularmente por GRASP, ILS y VNS. 
    Incluye registro automático de trayectoria para análisis de Caja Blanca.
    """
    ruta_actual = list(ruta_base)
    fobj_actual = fobj_base
    res_actual = res_base.copy()

    n = len(ruta_actual)
    mejorando = True

    while mejorando:
        mejorando = False
        movimientos_posibles = [(i, j) for i in range(0, n - 1) for j in range(i + 1, n)]
        random.shuffle(movimientos_posibles)

        for i, j in movimientos_posibles:
            vecino = generar_vecino_swap(ruta_actual, i, j)
            res_vecino = evaluar_ruta(vecino, caso_bicis, caso_capacidad, coordenadas)
            evaluaciones_totales += 1

            kms_vec = res_vecino['kms_recorridos']
            ent_vec = res_vecino['entropia']
            fobj_vec = funcion_objetivo(kms=kms_vec, entropia=ent_vec, **kwargs)

            if fobj_vec < fobj_actual:
                ruta_actual = vecino
                fobj_actual = fobj_vec
                res_actual['kms_recorridos'] = kms_vec
                res_actual['entropia'] = ent_vec
                mejorando = True

                if fobj_actual < record_absoluto['fobj']:
                    record_absoluto['ruta'] = list(ruta_actual)
                    record_absoluto['fobj'] = fobj_actual
                    record_absoluto['kms'] = kms_vec
                    record_absoluto['entropia'] = ent_vec

                historial['evaluaciones'].append(evaluaciones_totales)
                historial['fobj_actual'].append(fobj_actual)
                historial['fobj'].append(record_absoluto['fobj'])
                historial['kms'].append(record_absoluto['kms'])
                historial['entropia'].append(record_absoluto['entropia'])

                break 

    return ruta_actual, fobj_actual, res_actual, evaluaciones_totales

def mutacion_fuerte_sublista(ruta, tamaño=4):
    """
    Operador de mutación fuerte para ILS y VNS.
    Extrae una sublista de tamaño fijo de forma cíclica, la desordena y la reinserta.
    """
    nueva_ruta = list(ruta)
    n = len(nueva_ruta)
    
    if n < tamaño:
        tamaño = n

    pos_inicial = random.randint(0, n - 1)
    indices_sublista = [(pos_inicial + i) % n for i in range(tamaño)]
    
    valores_extraidos = [nueva_ruta[i] for i in indices_sublista]
    random.shuffle(valores_extraidos)

    for idx, valor_mezclado in zip(indices_sublista, valores_extraidos):
        nueva_ruta[idx] = valor_mezclado

    return nueva_ruta

# =============================================================================
# 1. MEDIDAS DE CAJA NEGRA (BLACK-BOX)
# =============================================================================

def calcular_metricas_caja_negra(resultados_5_ejecuciones, fobj_mejor_conocido):
    """
    Calcula la Robustez Matemática (Media, Desviación Estándar, CV) y el RPD
    a partir de las ejecuciones registradas.
    """
    print(f"{'='*65}")
    print(f"| {'Algoritmo':<10} | {'Media':<8} | {'Std (σ)':<8} | {'CV (%)':<8} | {'RPD (%)':<8} |")
    print(f"{'-'*65}")

    rpd_negativos = []
    for algoritmo, valores_fobj in resultados_5_ejecuciones.items():
        media = np.mean(valores_fobj)
        desviacion = np.std(valores_fobj)

        cv = (desviacion / media) * 100 if media > 0 else 0
        mejor_obtenido = np.min(valores_fobj)
        rpd = ((mejor_obtenido - fobj_mejor_conocido) / fobj_mejor_conocido) * 100

        if rpd < -0.1:
            rpd_negativos.append((algoritmo, mejor_obtenido, rpd))

        print(f"| {algoritmo:<10} | {media:<8.2f} | {desviacion:<8.2f} | {cv:<8.2f} | {rpd:<8.2f} |")
    print(f"{'='*65}")

    if rpd_negativos:
        print(f"  ℹ️  RPD negativo detectado vs. fobj_mejor_conocido={fobj_mejor_conocido:.4f}:")
        for algo, best, rpd in rpd_negativos:
            print(f"     {algo} obtiene {best:.4f} (RPD={rpd:+.2f}%) — supera el mejor de P1.")
        print("     Se debe documentar la mejora en el informe final.")

def graficar_boxplot_caja_negra(resultados_5_ejecuciones, nombre_caso):
    """
    Genera un Diagrama de Caja y Bigotes para evaluar visualmente 
    la dispersión y calidad de las soluciones.
    """
    algoritmos = list(resultados_5_ejecuciones.keys())
    datos = list(resultados_5_ejecuciones.values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bplot = ax.boxplot(datos, labels=algoritmos, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', color='black'),
                       medianprops=dict(color='orange', linewidth=2))
    
    colores = ['skyblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bplot['boxes'], colores):
        patch.set_facecolor(color)
        
    ax.set_title(f'Robustez: Dispersión en 5 Ejecuciones ({nombre_caso})', fontsize=14)
    ax.set_ylabel('Función Objetivo', fontsize=12)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. MEDIDAS DE CAJA BLANCA (WHITE-BOX)
# =============================================================================

def calcular_tasa_overlap_hamming(rutas, n_posiciones_prefijo=5):
    """
    Calcula la Tasa de Diversificación (Distancia de Hamming) 
    comparando prefijos de rutas iniciales generadas por diferentes semillas.
    """
    n_rutas = len(rutas)
    if n_rutas < 2:
        return 0.0
    
    similitudes = 0
    comparaciones_totales = 0
    
    for i in range(n_rutas):
        for j in range(i + 1, n_rutas):
            ruta_a = rutas[i][:n_posiciones_prefijo]
            ruta_b = rutas[j][:n_posiciones_prefijo]
            
            coincidencias = sum(1 for a, b in zip(ruta_a, ruta_b) if a == b)
            similitudes += coincidencias
            comparaciones_totales += n_posiciones_prefijo
            
    porcentaje_overlap = (similitudes / comparaciones_totales) * 100
    
    print(f">> Tasa de Overlap en las primeras {n_posiciones_prefijo} posiciones: {porcentaje_overlap:.1f}%")
    if porcentaje_overlap > 50:
        print("   ¡Alerta! Falta de diversidad. La heurística Greedy domina la selección probabilística.")
    else:
        print("   Buena diversificación. El algoritmo explora topologías estructuralmente distintas.")
        
    return porcentaje_overlap

def comparar_grasp_rcl(busqueda_grasp_fn, funcion_objetivo, estaciones_base, coordenadas,
                       caso_bicis, caso_capacidad, evaluar_ruta, semillas,
                       rcl_values=(3, 4, 5), n_posiciones_prefijo=4):
    """
    Estudio paramétrico del impacto del tamaño de la Lista Restringida de Candidatos (RCL) 
    en GRASP. Evalúa la Media, Desviación, CV y Overlap de Hamming.
    """
    print(f"{'='*78}")
    print(f"| {'RCL':<4} | {'Media':<8} | {'Best':<8} | {'σ':<8} | {'CV (%)':<8} | {'Overlap (%)':<12} |")
    print(f"{'-'*78}")

    for rcl in rcl_values:
        scores = []
        rutas = []
        for sem in semillas:
            res = busqueda_grasp_fn(funcion_objetivo, estaciones_base, coordenadas,
                                    caso_bicis, caso_capacidad, evaluar_ruta, sem,
                                    tamano_rcl=rcl)
            scores.append(res['fobj'])
            rutas.append(res['ruta'])

        media = np.mean(scores)
        best = np.min(scores)
        desv = np.std(scores)
        cv = (desv / media * 100) if media > 0 else 0.0

        sim, comp = 0, 0
        for i in range(len(rutas)):
            for j in range(i + 1, len(rutas)):
                a, b = rutas[i][:n_posiciones_prefijo], rutas[j][:n_posiciones_prefijo]
                sim += sum(1 for x, y in zip(a, b) if x == y)
                comp += n_posiciones_prefijo
        overlap = (sim / comp * 100) if comp > 0 else 0.0

        print(f"| {rcl:<4} | {media:<8.4f} | {best:<8.4f} | {desv:<8.4f} | {cv:<8.2f} | {overlap:<12.1f} |")
    print(f"{'='*78}")
    print("El incremento de RCL relaja el determinismo constructivo, inyectando mayor exploración espacial.")

def calcular_profundidad_estancamiento_ils(historiales_ils):
    """
    Calcula el 'Stagnation Depth' para ILS. Mide el coste computacional promedio 
    (en evaluaciones) que requiere la Búsqueda Local antes de detenerse y permitir una nueva mutación.
    """
    todas_evals = []
    for h in historiales_ils:
        todas_evals.extend(h.get('evals_por_bl', []))

    if not todas_evals:
        print(">> ILS Stagnation Depth: Sin datos registrados.")
        return

    media = np.mean(todas_evals)
    desv = np.std(todas_evals)
    minimo = np.min(todas_evals)
    maximo = np.max(todas_evals)
    n_ciclos = len(todas_evals)

    print(f">> ILS Stagnation Depth (evaluaciones de BL por ciclo, n={n_ciclos}):")
    print(f"   Media={media:.1f}   σ={desv:.1f}   Min={minimo}   Max={maximo}")
    if media > 1500:
        print("   ⚠️ Estancamiento profundo: La explotación consume iteraciones excesivas.")
    else:
        print("   ✓ Profundidad controlada. La optimización finaliza rápidamente, favoreciendo el ciclo de mutación.")

def calcular_profundidad_estancamiento_vns(historiales_vns, k_max=4):
    """
    Calcula el 'Stagnation Depth' para VNS. Muestra la distribución de mejoras 
    por entorno (k) para detectar redundancias matemáticas en vecindarios pequeños.
    """
    mejoras_acum = {k: 0 for k in range(1, k_max + 1)}
    intentos_acum = {k: 0 for k in range(1, k_max + 1)}

    for h in historiales_vns:
        for k, count in h.get('mejoras_por_k', {}).items():
            mejoras_acum[k] += count
        for k, count in h.get('intentos_por_k', {}).items():
            intentos_acum[k] += count

    total_mejoras = sum(mejoras_acum.values())
    
    print(f">> VNS Stagnation Depth (acumulado en {len(historiales_vns)} ejecuciones):")
    print(f"   {'k':<4} | {'Intentos':<9} | {'Mejoras':<8} | {'Tasa éxito':<10} | {'% del total mejoras':<20}")
    print(f"   {'-'*65}")
    for k in range(1, k_max + 1):
        intentos = intentos_acum[k]
        mejoras = mejoras_acum[k]
        tasa = (mejoras / intentos * 100) if intentos > 0 else 0.0
        pct_total = (mejoras / total_mejoras * 100) if total_mejoras > 0 else 0.0
        print(f"   {k:<4} | {intentos:<9} | {mejoras:<8} | {tasa:<9.1f}% | {pct_total:<19.1f}%")

    if total_mejoras == 0:
        print("   ⚠️ Cero mejoras registradas por VNS — evaluar topología del escenario.")
        return

    pct_grande = mejoras_acum[k_max] / total_mejoras * 100
    pct_pequenos = (mejoras_acum[1] + mejoras_acum[2]) / total_mejoras * 100
    if pct_grande > 50:
        print(f"   ⚠️ {pct_grande:.0f}% de las mejoras provienen del entorno k={k_max}. Posible redundancia en k pequeños.")
    elif pct_pequenos > 60:
        print(f"   ✓ Entornos pequeños (k=1,2) generan el {pct_pequenos:.0f}% de las mejoras. Diversificación progresiva eficaz.")
    else:
        print("   ✓ Distribución equilibrada. El algoritmo explota toda la jerarquía de entornos.")

# =============================================================================
# GENERACIÓN DE REPORTES HTML
# =============================================================================

def generar_tabla_global(diccionario_resultados):
    """
    Genera y formatea la Tabla Global de Resultados en estructura HTML horizontal.
    """
    html = """
    <style>
        .tabla-p2 { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: Arial, sans-serif; 
            margin-top: 15px; 
            background-color: #ffffff; 
            color: #000000; 
            font-size: 0.9em;
        }
        .tabla-p2 th { 
            background-color: #e0e0e0; 
            color: #000000; 
            font-weight: bold; 
            text-align: center; 
            padding: 8px 4px; 
            border: 1px solid #444;
        }
        .tabla-p2 .borde-grueso {
            border-right: 3px solid #222;
        }
        .tabla-p2 td { 
            padding: 8px 4px; 
            text-align: center; 
            border: 1px solid #bbbbbb; 
            color: #111111;
        }
        .tabla-p2 td.algo-name { 
            font-weight: bold; 
            background-color: #f5f5f5; 
            border-right: 3px solid #222;
            border-left: 3px solid #222;
            text-align: left;
            padding-left: 10px;
        }
        .tabla-p2 tbody tr:last-child td {
            border-bottom: 3px solid #222;
        }
        .tabla-p2 thead tr:first-child th {
            border-top: 3px solid #222;
        }
    </style>
    
    <h3 style="color: inherit;">Tabla 1.1. Resultados Globales</h3>
    <table class="tabla-p2">
        <thead>
            <tr>
                <th rowspan="2" class="borde-grueso" style="border-left: 3px solid #222;">Modelo</th>
                <th colspan="4" class="borde-grueso">Caso 1</th>
                <th colspan="4" class="borde-grueso">Caso 2</th>
                <th colspan="4" class="borde-grueso">Caso 3</th>
            </tr>
            <tr>
    """
    
    for _ in range(3):
        html += '<th>#Ev</th><th>F OBJ</th><th>Kms</th><th class="borde-grueso">Entr</th>'
        
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    orden_algoritmos = [
        ('Greedy', ['greedy']),
        ('P.Mejor', ['primer mejor', 'p.mejor']),
        ('GRASP', ['grasp']),
        ('ILS', ['ils']),
        ('VNS', ['vns'])
    ]
    
    casos = ['Caso 1', 'Caso 2', 'Caso 3']
    
    for nombre_display, keywords in orden_algoritmos:
        html += f'<tr><td class="algo-name">{nombre_display}</td>'
        
        clave_algo = None
        for k in diccionario_resultados.keys():
            if any(kw in k.lower() for kw in keywords):
                clave_algo = k
                break
                
        for caso in casos:
            clase_td_final = ' class="borde-grueso"'
            
            if clave_algo and caso in diccionario_resultados[clave_algo]:
                datos = diccionario_resultados[clave_algo][caso]
                
                ev = datos.get('ev_media', datos.get('evaluaciones', '-'))
                if isinstance(ev, (float, np.floating)): 
                    ev = f"{ev:.1f}"
                    
                fobj_val = datos.get('score_universal', datos['fobj'])
                fobj = f"{fobj_val:.4f}"
                kms = f"{datos['kms']:.2f}"
                entr = f"{datos['entropia']:.4f}"
                
                html += f"<td>{ev}</td><td>{fobj}</td><td>{kms}</td><td{clase_td_final}>{entr}</td>"
            else:
                html += f"<td>-</td><td>-</td><td>-</td><td{clase_td_final}>-</td>"
                
        html += "</tr>\n"

    html += "</tbody></table>"
    display(HTML(html))