import sys
import os

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

# Reuso de la configuración base (casos y semillas vigentes) de la Práctica 1.
from Practica_1.config import CASOS, SEMILLAS_P2, L

# -----------------------------------------------------------------------------
# Parámetros de los algoritmos de hormigas (valores de la consigna)
# -----------------------------------------------------------------------------
M_HORMIGAS = 10        # Número de hormigas por iteración (m)
N_ITERACIONES = 1000   # Criterio de parada: número de iteraciones
ALPHA = 1              # Peso de la feromona en la regla de transición (α)
BETA = 2               # Peso de la visibilidad (1/distancia) (β)
E_ELITISTAS = 5        # Nº de hormigas elitistas para SHE (e)
RHO = 0.1              # Tasa de evaporación de feromona (ρ)

# Específicos del Sistema de Colonias de Hormigas (SCH / ACS)
PHI = 0.1              # Tasa de actualización local de feromona (φ)
Q0 = 0.98             # Umbral de la regla pseudo-aleatoria proporcional (q0)

# Experimentación
N_EJECUCIONES = 3      # Nº de ejecuciones (semillas distintas) por algoritmo y caso
RCL_GREEDY = 3         # Tamaño de la lista corta del Greedy aleatorizado (comparador)

# 3 ejecuciones → 3 semillas distintas (subconjunto de las vigentes de la P2)
SEMILLAS_P4 = SEMILLAS_P2[:N_EJECUCIONES]

__all__ = [
    'CASOS', 'SEMILLAS_P2', 'SEMILLAS_P4', 'L',
    'M_HORMIGAS', 'N_ITERACIONES', 'ALPHA', 'BETA', 'E_ELITISTAS', 'RHO',
    'PHI', 'Q0', 'N_EJECUCIONES', 'RCL_GREEDY',
]
