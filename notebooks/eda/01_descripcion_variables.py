"""
PUNTO 1 — Descripción de variables
Tipos de datos, dimensiones, rangos y estructura general del dataset.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from _datos import cargar, CLASES, N_CLASES, RESULTADOS

x_train, y_train, x_test, y_test = cargar()

print("=" * 55)
print("DESCRIPCIÓN DE VARIABLES — CIFAR-10")
print("=" * 55)

print(f"\n{'Variable':<30} {'Detalle'}")
print("-" * 55)
print(f"{'Imágenes entrenamiento':<30} {len(x_train):,}")
print(f"{'Imágenes prueba':<30} {len(x_test):,}")
print(f"{'Total':<30} {len(x_train)+len(x_test):,}")
print(f"{'Forma por imagen':<30} {x_train.shape[1:]}  (alto × ancho × canales)")
print(f"{'Tipo de dato (píxeles)':<30} {x_train.dtype}  (entero sin signo 0-255)")
print(f"{'Valor mínimo':<30} {x_train.min()}")
print(f"{'Valor máximo':<30} {x_train.max()}")
print(f"{'Número de clases':<30} {N_CLASES}")
print(f"{'Tipo de dato (etiquetas)':<30} {y_train.dtype}  (categórico entero 0-9)")
print(f"{'Total características por imagen':<30} {3*32*32:,}  (32×32×3 píxeles)")

print("\n\nTABLA DE VARIABLES:")
tabla = pd.DataFrame({
    "Variable":       ["Píxeles (entrada)", "Etiqueta (salida)"],
    "Tipo":           ["uint8 -> float32 tras normalizar", "int (categórico nominal)"],
    "Forma":          ["(N, 32, 32, 3)", "(N,)"],
    "Rango":          ["[0, 255]  ->  [~-2, ~2] normalizado", "[0, 9]"],
    "Cardinalidad":   ["Continua (256 valores/canal/píxel)", "10 clases"],
    "Nulos":          ["0", "0"],
})
print(tabla.to_string(index=False))

print("\n\nCLASES DEL DATASET:")
tabla_clases = pd.DataFrame({
    "ID":    range(N_CLASES),
    "Clase": CLASES,
    "Categoría": ["Vehículo","Vehículo","Animal","Animal","Animal",
                  "Animal","Animal","Animal","Vehículo","Vehículo"],
})
print(tabla_clases.to_string(index=False))

print(f"\n-> Guardado en consola. No genera imagen.")

