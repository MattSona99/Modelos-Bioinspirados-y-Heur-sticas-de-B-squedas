# Práctica 2: Modelos Bioinspirados y Heurísticas de Búsqueda

Este directorio contiene la implementación de la **Práctica 2** para la resolución del **Problema de Optimización y Balanceo de una Red de Bicicletas** en la ciudad de Santander.

A diferencia de la Práctica 1 (centrada en heurísticas constructivas, búsquedas locales y trayectorias clásicas), esta práctica se enfoca en **metaheurísticas avanzadas basadas en trayectorias múltiples**: GRASP, ILS y VNS. El proyecto evalúa y compara su rendimiento sobre las mismas variables operativas (kilómetros recorridos y Entropía del sistema), añadiendo además un **análisis de Caja Negra y Caja Blanca** (RPD, Coeficiente de Variación, Distancia de Hamming, profundidad de estancamiento) para discutir robustez, calidad y diversidad de las soluciones.

> ⚠️ La Práctica 2 **reutiliza** el motor de la Práctica 1: importa `generar_vecino_swap`, `generar_solucion_inicial_aleatoria`, `evaluar_ruta`, `fobj_ratio` y la configuración de casos/semillas desde `Practica_1/`. Por tanto, mantener intacta la carpeta `Practica_1/` es un requisito previo de ejecución.

---

## ⚙️ Requisitos Previos

Para garantizar la máxima compatibilidad y evitar errores de ejecución, se recomienda estrictamente el siguiente entorno:

* **Python:** Versión **3.13.5** (versión recomendada y utilizada durante el desarrollo).
* **Editor:** [Visual Studio Code (VSCode)](https://code.visualstudio.com/).
* **Extensiones de VSCode recomendadas:**
  * *Python* (de Microsoft)
  * *Jupyter* (de Microsoft, necesario para ejecutar el `.ipynb`)

---

## 🚀 Guía de Instalación y Configuración (VSCode)

Sigue estos pasos para preparar tu entorno de desarrollo local:

1. **Abrir el proyecto:**
   Abre Visual Studio Code, ve a `Archivo > Abrir Carpeta...` y selecciona la carpeta raíz de este repositorio (no únicamente la carpeta `Practica_2`, ya que esta práctica importa módulos desde `Practica_1`).

2. **Abrir la terminal:**
   Abre una nueva terminal integrada en VSCode (`Terminal > Nuevo Terminal` o `Ctrl + ñ`).

3. **Crear un entorno virtual (Recomendado):**
   Para no interferir con las librerías globales de tu sistema, crea un entorno virtual ejecutando:
   ```bash
   python -m venv venv
   ```

4. **Activar el entorno virtual:**
   * **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   * **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
   *(Sabrás que está activado porque aparecerá `(venv)` al inicio de la línea de comandos).*

5. **Instalar las dependencias:**
   Instala todas las librerías necesarias (`matplotlib`, `folium`, `numpy`, `ipython`) ejecutando desde el directorio `Practica_2/`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📁 Estructura del Proyecto

La estructura de archivos de la práctica es la siguiente:

```text
Practica_2/
├── Documents                           # Documentación y análisis de la práctica
│     ├── Practica 2_V2026_V2b.pdf      # Guía y directivas originales
│     ├── Analisi.md                    # Borrador interno del análisis
│     └── Analysis.pdf                  # Análisis de resultados (memoria final)
├── algorithms.py                       # Implementación de GRASP, ILS y VNS
├── notebook.ipynb                      # Cuaderno interactivo con análisis visual
├── requirements.txt                    # Dependencias de Python
├── runner.py                           # Script principal de ejecución por consola
└── utils.py                            # Funciones auxiliares específicas de la P2
```

El código fuente ha sido modularizado para separar la lógica algorítmica, los análisis estadísticos y la visualización:

* **`Documents`**: Carpeta que contiene la documentación oficial (`Practica 2_V2026_V2b.pdf` con el guion de la práctica y el anecdotario sobre funciones objetivo avanzadas) y la memoria final redactada con el análisis exhaustivo de los resultados (`Analysis.pdf`).
* **`algorithms.py`**: Contiene las tres metaheurísticas exigidas:
  * `busqueda_grasp` — GRASP con construcción Greedy probabilística (RCL de tamaño fijo) + Búsqueda Local del Primer Mejor.
  * `busqueda_ils` — ILS con perturbación de tipo *mutación fuerte* sobre sub-lista y criterio de aceptación basado en `record_absoluto`.
  * `busqueda_vns` — VNS con estructuras de entorno crecientes ($k=1..k_{max}$), reset a $k=1$ tras mejora y *shaking* sobre sub-lista variable.
* **`utils.py`**: Funciones auxiliares específicas de la Práctica 2:
  * Constructores y operadores: `construir_solucion_grasp`, `aplicar_busqueda_local_primer_mejor`, `mutacion_fuerte_sublista`.
  * Métricas de Caja Negra: `calcular_metricas_caja_negra` (RPD, CV), `calcular_tasa_overlap_hamming`.
  * Métricas de Caja Blanca: `calcular_profundidad_estancamiento_ils`, `calcular_profundidad_estancamiento_vns`, `comparar_grasp_rcl`.
  * Visualización: `graficar_boxplot_caja_negra`, `generar_tabla_global`.
* **`runner.py`**: Script ejecutable por consola. Orquesta GRASP, ILS y VNS sobre los 3 casos de prueba y las semillas definidas en `Practica_1/config.py`, recopila los datos brutos y dispara los análisis avanzados de Caja Negra y Caja Blanca.
* **`notebook.ipynb`**: Cuaderno de Jupyter interactivo. Es la forma más visual de ejecutar la práctica: muestra paso a paso la evolución de las gráficas duales (`fobj_actual` vs. `fobj`), los boxplots de robustez, la tabla comparativa final y el estudio comparativo de tamaños de RCL.
* **`requirements.txt`**: Lista estricta de dependencias y librerías de Python necesarias.

> 🔗 **Dependencias cruzadas con Práctica 1:** `algorithms.py` y `runner.py` añaden la raíz del repositorio al `sys.path` para importar `Practica_1.utils` y `Practica_1.config`. Por eso es necesario abrir VSCode sobre la **raíz del repositorio**, no sobre `Practica_2/`.

---

## 🧠 Algoritmos Implementados

| Algoritmo | Estrategia clave | Parámetros principales |
|-----------|------------------|------------------------|
| **GRASP** | Multi-arranque Greedy probabilístico + BL Primer Mejor | `max_iter_grasp=10`, `tamano_rcl=3` |
| **ILS** | BL + perturbación fuerte sobre sub-lista del récord absoluto | `max_iter_ils`, `tamaño` de sub-lista |
| **VNS** | Estructuras de entorno crecientes con *shaking* y reset | `k_max`, `bl_max` |

Las tres comparten la firma estándar `funcion_objetivo(kms, entropia, **kwargs)` y el diccionario `historial = {evaluaciones, fobj_actual, fobj, kms, entropia}`, lo que permite reutilizar los plots de convergencia de la Práctica 1.

---

## 📊 Análisis Caja Blanca y Caja Negra

La memoria final (`Documents/Analysis.pdf`) incluye, además de las tablas globales clásicas:

* **RPD** (*Relative Percentage Deviation*) — distancia relativa al mejor conocido $f^*$.
* **CV** (*Coeficiente de Variación*) — robustez entre semillas, $\sigma / \bar{x}$.
* **Distancia de Hamming / Overlap** sobre prefijos de ruta — diversidad real entre las soluciones.
* **Profundidad de estancamiento** — número de iteraciones consecutivas sin mejora en ILS y VNS.
* **Estudio comparativo del tamaño de la RCL** en GRASP (`comparar_grasp_rcl`).

---

## 🏃‍♂️ Cómo Ejecutar la Práctica

Tienes dos formas principales de visualizar los resultados:

**Opción A: A través de Jupyter Notebook (Recomendado para la visualización gráfica)**
1. Abre el archivo `Practica_2/notebook.ipynb` desde el panel izquierdo de VSCode.
2. Asegúrate de que en la esquina superior derecha del notebook, el "Kernel" seleccionado sea el de tu entorno virtual (`venv`).
3. Haz clic en el botón **"Ejecutar todo"** o reproduce celda por celda usando `Shift + Enter`.
4. Podrás ver los gráficos evolutivos duales, los boxplots de Caja Negra y la Tabla Global renderizada al final.

**Opción B: A través de la Terminal**
1. Con tu entorno virtual activado y desde la **raíz del repositorio**, ejecuta:
   ```bash
   python -m Practica_2.runner
   ```
   o, alternativamente, desde dentro de `Practica_2/`:
   ```bash
   python runner.py
   ```
2. La consola imprimirá los logs de progreso de GRASP, ILS y VNS sobre los 3 casos, los scores y las estadísticas de Caja Negra/Blanca.
