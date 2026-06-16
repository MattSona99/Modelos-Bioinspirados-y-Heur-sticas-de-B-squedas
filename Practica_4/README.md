# Práctica 4: Algoritmos de Optimización Basados en Colonias de Hormigas

Este directorio contiene la implementación de la **Práctica 4** (Actividad Académica Dirigida) para la resolución del problema de la **red de bicicletas de Santander** mediante **Algoritmos de Optimización Basados en Colonias de Hormigas (OCH)**: el **Sistema de Hormigas (SH)**, el **Sistema de Hormigas Elitista (SHE)** y el **Sistema de Colonias de Hormigas (SCH)**. Los tres se comparan entre sí y con un **Greedy aleatorizado de lista corta (tamaño 3)**.

> ⚠️ **Cambio de problema respecto a P1–P3.** En esta práctica se **elimina la restricción de capacidad del camión**: el objetivo se reduce a **recorrer todas las estaciones en la mínima distancia posible** (un TSP). Con capacidad ilimitada, todas las estaciones se equilibran al 50% y la entropía resulta **constante por caso** (independiente del orden de visita), por lo que el coste que guía a las hormigas es **C(S) = kilómetros recorridos** y el discriminante entre algoritmos es la distancia. La entropía se sigue reportando en las tablas como métrica de balanceo del sistema.

> 🔁 La Práctica 4 **reutiliza** el motor de la Práctica 1: la representación matricial (distancias entre estaciones), la evaluación (`evaluar_ruta`, `fobj_ratio`), la configuración de casos/semillas (`Practica_1/config.py`) y los mapas Folium. Mantener intacta la carpeta `Practica_1/` es un requisito previo de ejecución.

---

## ⚙️ Requisitos Previos

* **Python:** 3.13+ (recomendado).
* **Editor:** Visual Studio Code con las extensiones *Python* y *Jupyter*.
* **Dependencias:** `matplotlib`, `folium`, `numpy`, `ipython` (la animación usa `folium.plugins`, ya incluido; **sin selenium**).

---

## 🚀 Instalación (VSCode)

1. Abre la carpeta **raíz del repositorio** (no solo `Practica_4`, ya que importa módulos de `Practica_1`).
2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate      # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Instala dependencias desde `Practica_4/`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📁 Estructura del Proyecto

```text
Practica_4/
├── Documents
│     ├── Instructions.pdf          # Guion oficial (AAD, Versión 2026, 1.0)
│     └── Analysis.pdf              # Informe final (resultados y análisis)
├── algorithms.py                   # SH, SHE, SCH y Greedy aleatorizado
├── notebook.ipynb                  # Cuaderno interactivo con tablas, convergencia y mapas
├── requirements.txt                # Dependencias de Python
├── runner.py                       # Orquestación de experimentos y Tabla 1.1
└── utils.py                        # Matriz de distancias, feromona, reportes y GIF
```

* **`algorithms.py`** — Los tres algoritmos de hormigas + el comparador:
  * `sistema_hormigas` (**SH**) — regla de transición proporcional `τ^α·η^β`; tras evaporar (ρ), **todas** las hormigas depositan `1/C(S)`.
  * `sistema_hormigas_elitista` (**SHE**) — SH + `e=5` hormigas elitistas que refuerzan los arcos del **mejor tour global** en cada iteración.
  * `sistema_colonias_hormigas` (**SCH** / ACS) — regla **pseudo-aleatoria** (con prob. `q0` explota el mejor arco), **actualización local** de feromona (`φ`) durante la construcción y **actualización global** sólo por la mejor hormiga.
  * `greedy_aleatorizado` — vecino más cercano estocástico sobre la lista corta de las 3 estaciones más próximas (línea base de la consigna).
* **`utils.py`** — `construir_matriz_distancias`, `calcular_visibilidad`, `construir_tour_hormiga` (reglas de transición), helpers de feromona, `coste_greedy_determinista` (la `L` de `τ0 = 1/(n·L)`), `evaluar_tour_sin_capacidad`, los generadores de la **Tabla 1.1** (global y parcial con media/desviación) y la **animación interactiva** del rastro de feromona (`generar_animacion_feromona`, Folium time-slider).
* **`runner.py`** — `ejecutar_experimento_aco` (barre casos × semillas, mejor por `fobj_ratio`, convergencia y mapas), `ejecutar_tabla_global_aco` (Tabla 1.1 — mejor por algoritmo), `ejecutar_tabla_parcial_aco` (Tabla 1.1 — ejecuciones + media/desviación) y un `__main__` con un *smoke test* reducido.
* **`notebook.ipynb`** — Forma recomendada de ejecución.

---

## 🐜 Parámetros de la consigna

| Parámetro | Valor | Aplicación |
|-----------|-------|------------|
| Ejecuciones | 3 (semillas distintas) | todos |
| Hormigas `m` | 10 | todos |
| Iteraciones | 1000 | todos |
| `α`, `β` | 1, 2 | regla de transición |
| Hormigas elitistas `e` | 5 | SHE |
| Evaporación `ρ` | 0.1 | actualización de feromona |
| Feromona inicial `τ0` | `1/(n·L)` | `L` = coste del greedy |
| Aporte | `1/C(Sₖ)` | depósito de feromona |
| Actualización local `φ` | 0.1 | SCH |
| Regla pseudo-aleatoria `q0` | 0.98 | SCH |

El número de **evaluaciones** (llamadas a la función de coste) se contabiliza como métrica adicional de comparación: `m × iteraciones` por ejecución.

---

## 📊 Estudio Experimental

* **Tabla 1.1 — Resultados Globales:** mejor resultado de cada algoritmo (SH, SHE, SCH, Greedy A.) y caso, con `#Ev`, `F OBJ`, `Kms` y `Entr`. El mejor de cada columna se marca en negrita.
* **Tabla 1.1 — Resultados Parciales:** las 3 ejecuciones (una por semilla) por algoritmo y caso, con **media y desviación típica**.
* **Convergencia:** por iteración se trazan dos curvas — *mejor global* (best-so-far) y *mejor de la iteración* (exploración). En **SH** la curva global forma un "ángulo recto": el récord se estanca muy pronto porque la feromona se concentra rápido (todas las hormigas refuerzan y la evaporación es baja `ρ=0.1`), mientras la curva de iteración sigue explorando sin mejorar el récord. **SCH**, por el contrario, converge de forma gradual y su curva de iteración es casi plana (explotación con `q0=0.98`). El notebook incluye un estudio de convergencia de SH variando **ρ** (evaporación) y **β** (visibilidad), sin cambiar los valores por defecto de la consigna.
* **Mapas Folium:** mejor tour por caso y **animación interactiva** del rastro de feromona (slider/play sobre el mapa de Santander, muchos fotogramas). La traza se dibuja con **escala global** y baseline `τ0`: sólo aparecen las aristas reforzadas y la traza **emerge de forma gradual** (en ACS la mayoría de aristas quedan congeladas en `τ0` y no se pintan, evitando el "todo lleno → vacío de golpe").

---

## 🏃 Cómo Ejecutar

**Opción A: Jupyter Notebook (recomendado)**
1. Abre `Practica_4/notebook.ipynb`.
2. Selecciona el kernel del entorno virtual (`venv`).
3. Pulsa **"Ejecutar todo"**.

**Opción B: Terminal** (desde la raíz del repositorio):
```bash
python -m Practica_4.runner
```
Ejecuta un *smoke test* reducido (1 caso, 2 semillas, 30 iteraciones) que valida SH, SHE, SCH y la Tabla 1.1.
