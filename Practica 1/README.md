# Práctica 1: Modelos Bioinspirados y Heurísticas de Búsqueda

Este repositorio contiene la implementación de la Práctica 1 para la resolución del **Problema de Optimización y Balanceo de una Red de Bicicletas** en la ciudad de Santander. 

El proyecto evalúa y compara el rendimiento de distintas metaheurísticas (Algoritmo Greedy, Búsqueda Aleatoria, Búsqueda Local Primer Mejor/Mejor Vecino, Enfriamiento Simulado y Búsqueda Tabú) analizando variables como la distancia recorrida (Kms) y la Entropía del sistema.

---

## ⚙️ Requisitos Previos

Para garantizar la máxima compatibilidad y evitar errores de ejecución, se recomienda estrictamente el siguiente entorno:

* **Python:** Versión **3.13.5** (versión recomendada y utilizada durante el desarrollo).
* **Editor:** [Visual Studio Code (VSCode)](https://code.visualstudio.com/).
* **Extensiones de VSCode recomendadas:** * *Python* (de Microsoft)
  * *Jupyter* (de Microsoft, necesario para ejecutar el `.ipynb`)

---

## 🚀 Guía de Instalación y Configuración (VSCode)

Sigue estos pasos para preparar tu entorno de desarrollo local:

1. **Abrir el proyecto:**
   Abre Visual Studio Code, ve a `Archivo > Abrir Carpeta...` y selecciona la carpeta raíz de este proyecto.

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
   Instala todas las librerías necesarias (como `folium`, `matplotlib`, `numpy`, etc.) ejecutando:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📁 Estructura del Proyecto

La estructura de archivos de la práctica es la siguiente:

```text
Practica 1/
├── Documents                # Documentación y análisis de la práctica
|     ├── Instructions.pdf   # Guía y directivas originales
|     └── Análysis.pdf       # Análisis de resultados
├── algorithms.py            # Lógica matemática y algoritmos
├── config.py                # Variables globales y definición de casos
├── coords.json              # Base de datos geoespacial (lat/lon)
├── notebook.ipynb           # Cuaderno interactivo con análisis visual
├── requirements.txt         # Dependencias de Python
├── runner.py                # Script principal de ejecución por consola
└── utils.py                 # Librería de funciones auxiliares y gráficos
```

El código fuente ha sido modularizado para separar la lógica algorítmica, la visualización y la configuración:

* **`Documents`**: Carpeta que contiene la documentación oficial (`Instructions.pdf` con el guion de la práctica) y la memoria final redactada con el análisis exhaustivo de los resultados obtenidos (`Analysis.pdf`).
* **`algorithms.py`**: Contiene la lógica matemática y estructural de las metaheurísticas exigidas (Búsqueda Aleatoria, Primer Mejor, Mejor Vecino, Enfriamiento Simulado y Búsqueda Tabú).
* **`config.py`**: Define los 3 casos de prueba base (estado del inventario de bicicletas, capacidades máximas de las estaciones) y las variables globales del entorno.
* **`runner.py`**: Script ejecutable por consola. Se encarga de lanzar los algoritmos secuencialmente, inyectar las semillas, recopilar los datos y calcular estadísticas globales de tiempo y esfuerzo computacional.
* **`utils.py`**: Librería de funciones auxiliares. Incluye el cálculo de distancias Manhattan, cálculo de Entropía, rutinas para renderizar los mapas geográficos con Folium, graficadores de convergencia y el generador de la Tabla Global.
* **`notebook.ipynb`**: Cuaderno de Jupyter interactivo. Es la forma más visual de ejecutar la práctica, ya que permite ver paso a paso la evolución de las gráficas, el renderizado de los mapas de Santander y la Tabla Global final de resultados.
* **`coords.json`**: Base de datos geoespacial que contiene la Latitud y Longitud exacta de las estaciones de bicicletas.
* **`requirements.txt`**: Archivo de texto con la lista estricta de dependencias y librerías de Python necesarias para que el proyecto funcione.

---

## 🏃‍♂️ Cómo Ejecutar la Práctica

Tienes dos formas principales de visualizar los resultados:

**Opción A: A través de Jupyter Notebook (Recomendado para la visualización gráfica)**
1. Abre el archivo `notebook.ipynb` desde el panel izquierdo de VSCode.
2. Asegúrate de que en la esquina superior derecha del notebook, el "Kernel" seleccionado sea el de tu entorno virtual (`venv`).
3. Haz clic en el botón **"Ejecutar todo"** o reproduce celda por celda usando `Shift + Enter`.
4. Podrás interactuar con los mapas generados por Folium y ver la tabla HTML/Markdown renderizada al final.

**Opción B: A través de la Terminal**
1. Con tu entorno virtual activado, ejecuta el lanzador base:
   ```bash
   python runner.py
   ```
2. La consola imprimirá los logs de progreso, los scores y generará las estadísticas solicitadas.
