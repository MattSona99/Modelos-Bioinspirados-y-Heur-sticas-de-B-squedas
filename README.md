# Modelos Bioinspirados y Heurísticas de Búsqueda 🧬🔍

Repositorio global de proyectos y prácticas desarrolladas para la asignatura **Modelos Bioinspirados y Heurísticas de Búsqueda**.

Este repositorio servirá como portafolio y registro de la evolución en la implementación de diferentes algoritmos de optimización, desde heurísticas constructivas y locales, hasta metaheurísticas complejas basadas en trayectorias o poblaciones.

---

## 📋 Índice de Prácticas

A continuación se listan las prácticas desarrolladas durante el curso. Puedes acceder a la carpeta de cada práctica para ver su código fuente, documentación específica e instrucciones de ejecución.

### [✅ Práctica 1: Algoritmos basados en Entornos y Trayectorias](./Practica_1)
* **Objetivo:** Estudiar, implementar y comparar el funcionamiento de distintos Algoritmos de Búsqueda Aleatoria, Local, Enfriamiento Simulado y Búsqueda Tabú frente a un algoritmo base (Greedy).
* **Problema a resolver:** Optimización operativa de las rutas de reubicación de una red de bicicletas en la ciudad de Santander. Se cuenta con un camión de capacidad limitada ($L=20$) que debe equilibrar la red, minimizando los kilómetros recorridos y optimizando el balanceo de las estaciones (Entropía).
* **Algoritmos Implementados:**
  * Algoritmo Constructivo: *Greedy*
  * Búsqueda Ciega: *Búsqueda Aleatoria*
  * Búsquedas Locales: *Primer Mejor* y *Mejor Vecino*
  * Metaheurísticas Avanzadas: *Enfriamiento Simulado* (Esquema de Cauchy) y *Búsqueda Tabú* (con memoria a largo/corto plazo y reinicializaciones).
* 🔗 **[Ir a la documentación y código de la Práctica 1](./Practica_1)**

### [✅ Práctica 2: Metaheurísticas Multi-arranque y Basadas en Entornos Variables](./Practica_2)
* **Objetivo:** Implementar y comparar metaheurísticas avanzadas que extienden las búsquedas locales clásicas mediante mecanismos de diversificación (multi-arranque, perturbación, entornos variables), añadiendo además un análisis estadístico riguroso de Caja Blanca y Caja Negra (RPD, CV, Distancia de Hamming, profundidad de estancamiento).
* **Problema a resolver:** El mismo problema de balanceo de la red de bicicletas de Santander, reutilizando la infraestructura de la Práctica 1 (`evaluar_ruta`, `fobj_ratio`, casos y semillas).
* **Algoritmos Implementados:**
  * *GRASP* (Greedy Randomized Adaptive Search Procedure) — construcción probabilística con RCL de tamaño fijo + Búsqueda Local Primer Mejor.
  * *ILS* (Iterated Local Search) — BL + *mutación fuerte* sobre sub-lista del récord absoluto.
  * *VNS* (Variable Neighborhood Search) — estructuras de entorno crecientes ($k=1..k_{max}$) con *shaking* y reset tras mejora.
* 🔗 **[Ir a la documentación y código de la Práctica 2](./Practica_2)**

### [✅ Práctica 3: Algoritmos Genéticos y Evolutivos Poblacionales](./Practica_3)
* **Objetivo:** Diseñar, implementar y comparar algoritmos evolutivos de población (AG Básico, CHC y AG Multimodal con *Clearing*) frente a las líneas base *Greedy* y *Búsqueda Local Primer Mejor*, con un análisis estadístico de Caja Negra (CV, RPD) y Caja Blanca (convergencia/reinicios, dinámica de nichos, diversidad por arcos).
* **Problema a resolver:** El mismo problema de balanceo de la red de bicicletas de Santander, reutilizando el motor de las Prácticas 1 y 2 (`evaluar_ruta`, `fobj_ratio`, `FUNCIONES_OBJETIVO`, casos y semillas).
* **Algoritmos Implementados:**
  * *AG Básico* — generacional con elitismo, selección por torneo, cruce OX circular y mutación 2-opt.
  * *CHC* — prevención de incesto por **distancia de Hamming por arcos**, recombinación HUX adaptada a orden, reinicios cataclísmicos y supervivencia elitista sin mutación.
  * *AG Multimodal (Clearing)* — mantenimiento de múltiples óptimos mediante nichos definidos por radio de Hamming por arcos.
* 🔗 **[Ir a la documentación y código de la Práctica 3](./Practica_3)**

### [✅ Práctica 4: Algoritmos de Optimización Basados en Colonias de Hormigas (OCH)](./Practica_4)
* **Objetivo:** Implementar y comparar los tres modelos clásicos de Optimización por Colonias de Hormigas (SH, SHE y SCH) frente a un *Greedy* aleatorizado de lista corta (tamaño 3), añadiendo análisis de convergencia, un estudio paramétrico (ρ, β) y una **animación interactiva** del rastro de feromona sobre el mapa de Santander.
* **Problema a resolver:** Variante **TSP** del problema de Santander: se **elimina la restricción de capacidad del camión** y el objetivo se reduce a recorrer todas las estaciones en la mínima distancia posible (con capacidad ilimitada la entropía resulta constante por caso, por lo que el coste que guía a las hormigas es la distancia). Reutiliza el motor de la Práctica 1 (matriz de distancias, `evaluar_ruta`, `fobj_ratio`, casos y semillas).
* **Algoritmos Implementados:**
  * *SH (Sistema de Hormigas)* — regla de transición proporcional `τ^α·η^β`; tras evaporar (ρ), **todas** las hormigas depositan `1/C(S)`.
  * *SHE (Sistema de Hormigas Elitista)* — SH + `e=5` hormigas elitistas que refuerzan los arcos del **mejor tour global** en cada iteración.
  * *SCH (Sistema de Colonias de Hormigas / ACS)* — regla **pseudo-aleatoria** (`q0`), **actualización local** de feromona (`φ`) durante la construcción y **actualización global** sólo por la mejor hormiga.
  * Línea base: *Greedy aleatorizado* — vecino más cercano estocástico sobre la lista corta de las 3 estaciones más próximas.
* 🔗 **[Ir a la documentación y código de la Práctica 4](./Practica_4)**

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.13+
* **Entorno:** Visual Studio Code / Jupyter Notebooks
* **Librerías principales:** `numpy`, `matplotlib`, `folium`, `ipython`

---
*Desarrollado para el curso académico 2025/2026.*
