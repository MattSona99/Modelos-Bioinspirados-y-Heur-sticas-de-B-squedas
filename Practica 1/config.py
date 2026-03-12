L = 20 # Capacidad máxima camion
BICIS_INICIALES = 7 # Bicis iniciales en el camión
OBJETIVO = 0.50 # Proporción objetivo de llenado
ESTACION_INICIO = 0 # Estación de inicio
TOLERANCIA = 1 # Margen para considerar una estación "optima" y no visitarla

# Seeds para reproducibilidad
SEMILLAS = [42, 123, 987, 555, 2024]

# Datos de los Escenarios (Casos)
CASOS = {
    "Caso 1": {
        "bicis": [5, 7, 13, 6, 8, 13, 8, 9, 6, 10, 10, 18, 8, 13, 15, 14],
        "capacidad": [16, 16, 23, 13, 21, 17, 14, 13, 17, 16, 27, 18, 12, 18, 18, 19]
    },
    "Caso 2": {
        "bicis": [15, 14, 3, 10, 2, 1, 8, 13, 17, 1, 20, 9, 6, 6, 15, 14],
        "capacidad": [16, 16, 23, 13, 21, 17, 14, 13, 17, 16, 27, 18, 12, 18, 18, 19]
    },
    "Caso 3": {
        "bicis": [15, 14, 3, 10, 2, 1, 8, 13, 10, 0, 0, 0, 0, 0, 0, 0],
        "capacidad": [15, 14, 3, 10, 2, 1, 8, 13, 17, 16, 27, 18, 12, 18, 18, 19]
    }
}