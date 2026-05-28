# Práctica 3: Modelos Bioinspirados y Heurísticas de Búsqueda

Este directorio contiene la implementación de la **Práctica 3** para la resolución del **Problema de Optimización y Balanceo de una Red de Bicicletas** en la ciudad de Santander.

A diferencia de la Práctica 1 (heurísticas constructivas y búsquedas locales/trayectoria) y de la Práctica 2 (metaheurísticas de trayectorias múltiples: GRASP, ILS, VNS), esta práctica se centra en **algoritmos evolutivos y poblacionales**: el **Algoritmo Genético Básico**, el **CHC** y un **Algoritmo Genético Multimodal** (variante *Clearing*). El comportamiento de los tres se compara entre sí y con las técnicas **Greedy** y **Búsqueda Local del Primer Mejor** de la Práctica 1.

> ⚠️ La Práctica 3 **reutiliza** el motor de las Prácticas 1 y 2: importa la representación (permutación de estaciones) y la evaluación (`evaluar_ruta`, `fobj_ratio`, `FUNCIONES_OBJETIVO`), la configuración de casos/semillas (`Practica_1/config.py`) y el análisis de Caja Negra (`Practica_2/utils.py`). Por tanto, mantener intactas las carpetas `Practica_1/` y `Practica_2/` es un requisito previo de ejecución.

---

## ⚙️ Requisitos Previos

* **Python:** Versión **3.13.5** (recomendada y utilizada durante el desarrollo).
* **Editor:** [Visual Studio Code (VSCode)](https://code.visualstudio.com/).
* **Extensiones recomendadas:** *Python* y *Jupyter* (de Microsoft).

---

## 🚀 Guía de Instalación (VSCode)

1. **Abrir el proyecto:** abre la carpeta **raíz del repositorio** (no solo `Practica_3`, ya que esta práctica importa módulos de `Practica_1` y `Practica_2`).
2. **Abrir la terminal integrada** (`Terminal > Nuevo Terminal`).
3. **Crear el entorno virtual:** `python -m venv venv`
4. **Activarlo:**
   * **Windows:** `.\venv\Scripts\activate`
   * **macOS/Linux:** `source venv/bin/activate`
5. **Instalar dependencias** desde `Practica_3/`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📁 Estructura del Proyecto

```text
Practica_3/
├── Documents
│     └── Instructions.pdf          # Guion y directivas oficiales (Versión 2026, 1.0)
├── algorithms.py                   # AG Básico, CHC y Multimodal (Clearing)
├── notebook.ipynb                  # Cuaderno interactivo con tablas y análisis
├── requirements.txt                # Dependencias de Python
├── runner.py                       # Script de ejecución y orquestación de tablas
└── utils.py                        # Operadores genéticos, distancia por arcos y reportes
```

* **`algorithms.py`** — Los tres algoritmos evolutivos exigidos:
  * `algoritmo_genetico_basico` — Generacional con **elitismo**, población de 30, selección por **torneo** (K = 10% de la población), cruce **OX** al 90% y, cuando no hay cruce, **mutación 2-opt** del 5–10% de la longitud del cromosoma.
  * `algoritmo_genetico_chc` — CHC con codificación de orden: prevención de incesto mediante la **distancia de Hamming por arcos** (umbral inicial `L/4`), recombinación **HUX adaptada a orden**, supervivencia elitista (μ+λ) **sin mutación**, descenso del umbral cuando ningún hijo entra y **reinicio** (copiar el mejor + resto aleatorio) al converger; los reinicios cuentan como la misma ejecución.
  * `algoritmo_genetico_multimodal` — AG Básico + **Clearing**: el radio del nicho se mide con la distancia de Hamming por arcos; cada generación se conservan los dominantes y se penaliza al resto.
* **`utils.py`** — Operadores (`cruce_ox`, `mutacion_2opt`, `cruce_hux_orden`, `seleccion_torneo`), la **`distancia_hamming_arcos`** (Hamming adaptada a secuencia), métricas de diversificación estructural y los generadores HTML de la **Tabla 1.1** (parciales) y **Tabla 1.2** (global), más la Caja Blanca de CHC (convergencia/reinicios) y Clearing (dominantes por generación).
* **`runner.py`** — `ejecutar_experimento_ga` (barrido de todas las `FUNCIONES_OBJETIVO` × semillas × casos, mejor por *score universal* `fobj_ratio`), `ejecutar_tabla_parcial` (Tabla 1.1) y `ejecutar_analisis_cajas_p3` (Caja Negra/Blanca). Incluye un bloque `__main__` con un *smoke test* reducido.
* **`notebook.ipynb`** — Forma recomendada de ejecución: Greedy y P.Mejor (reutilizados de P1), los 3 AG, Tabla 1.2, Tablas 1.1 y el análisis de Caja Negra/Blanca por caso.

---

## 🧠 Reflexión clave: distancia de Hamming adaptada a secuencia

La Hamming clásica (cuántas posiciones difieren) **falsea los datos** en una permutación: dos rutas que recorren exactamente los mismos tramos pero desplazadas una posición tendrían distancia máxima pese a ser casi idénticas en coste. Lo que hace una solución parecida a otra **no es la posición absoluta** de una estación, sino **qué estación se visita justo después de cuál** (el *arco*), ya que de los arcos dependen los kilómetros y la entropía.

Por ello `distancia_hamming_arcos` define la distancia como el número de **arcos** (pares consecutivos, incluido el ida/vuelta al depósito) presentes en una solución y ausentes en la otra. Esta única definición es el punto de verdad compartido por **CHC** (umbral de incesto), **Clearing** (radio de nicho) y la **métrica de diversificación estructural**.

---

## 📊 Estudio Experimental

* **Tabla 1.1 — Resultados Parciales:** por algoritmo, las 5 ejecuciones (una por semilla) en cada caso, con Media y Desviación. El mejor de cada columna se marca en negrita.
* **Tabla 1.2 — Resultados Globales:** mejor resultado de cada algoritmo (Greedy, P.Mejor, Básico, CHC, Multimodal) y caso.
* **Caja Negra (P2):** Coeficiente de Variación y RPD/RE frente al mejor conocido de P1/P2; Diagrama de Caja y Bigotes para la robustez.
* **Caja Blanca:** convergencia y nº de reinicios de CHC (estudio de cuántos rearranques convienen) y dinámica de Clearing del Multimodal (dominantes/nichos por generación), además de la diversificación estructural por arcos.

---

## 🏃‍♂️ Cómo Ejecutar la Práctica

**Opción A: Jupyter Notebook (recomendado)**
1. Abre `Practica_3/notebook.ipynb`.
2. Selecciona el kernel del entorno virtual (`venv`).
3. Pulsa **"Ejecutar todo"**.

**Opción B: Terminal**
Desde la **raíz del repositorio**:
```bash
python -m Practica_3.runner
```
Ejecuta un *smoke test* reducido (1 caso, presupuesto corto) que valida los tres algoritmos.
