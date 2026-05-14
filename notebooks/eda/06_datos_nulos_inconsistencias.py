"""
PUNTO 2 — Identificar y tratar datos nulos o inconsistentes
Verificación completa de integridad del dataset CIFAR-10.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from _datos import cargar, N_CLASES, RESULTADOS

x_train, y_train, x_test, y_test = cargar()

print("=" * 55)
print("REPORTE DE CALIDAD DE DATOS — CIFAR-10")
print("=" * 55)

# 1. Valores NaN
nan_train = int(np.isnan(x_train.astype(np.float32)).sum())
nan_test  = int(np.isnan(x_test.astype(np.float32)).sum())

# 2. Etiquetas fuera de rango
etiq_invalidas_train = int(np.sum((y_train < 0) | (y_train >= N_CLASES)))
etiq_invalidas_test  = int(np.sum((y_test  < 0) | (y_test  >= N_CLASES)))

# 3. Imágenes con tamaño incorrecto
tam_incorrecto_train = int(np.sum(
    [x_train[i].shape != (32, 32, 3) for i in range(len(x_train))]
))

# 4. Píxeles fuera de rango uint8
fuera_rango_train = int(np.sum((x_train < 0) | (x_train > 255)))

# 5. Imágenes completamente negras o blancas
negras = int(np.sum(x_train.max(axis=(1,2,3)) == 0))
blancas = int(np.sum(x_train.min(axis=(1,2,3)) == 255))

# 6. Duplicados exactos (usando hash de brillo+etiqueta como proxy rápido)
brillo = x_train.mean(axis=(1,2,3))
hashes = list(zip(brillo.round(4), y_train))
duplicados_aprox = len(hashes) - len(set(hashes))

reporte = pd.DataFrame({
    "Verificación": [
        "Valores NaN en píxeles (train)",
        "Valores NaN en píxeles (test)",
        "Etiquetas fuera de rango [0,9] (train)",
        "Etiquetas fuera de rango [0,9] (test)",
        "Imágenes con tamaño ≠ (32,32,3)",
        "Píxeles fuera de rango [0,255]",
        "Imágenes completamente negras",
        "Imágenes completamente blancas",
        "Posibles duplicados (hash brillo+clase)",
    ],
    "Resultado": [
        nan_train, nan_test,
        etiq_invalidas_train, etiq_invalidas_test,
        tam_incorrecto_train,
        fuera_rango_train,
        negras, blancas,
        duplicados_aprox,
    ],
    "¿Problema?": [
        "No" if nan_train == 0 else "SÍ",
        "No" if nan_test  == 0 else "SÍ",
        "No" if etiq_invalidas_train == 0 else "SÍ",
        "No" if etiq_invalidas_test  == 0 else "SÍ",
        "No" if tam_incorrecto_train == 0 else "SÍ",
        "No" if fuera_rango_train    == 0 else "SÍ",
        "No" if negras  == 0 else "SÍ",
        "No" if blancas == 0 else "SÍ",
        "No" if duplicados_aprox == 0 else "Revisar",
    ],
})

print(reporte.to_string(index=False))

total_problemas = sum([
    nan_train, nan_test,
    etiq_invalidas_train, etiq_invalidas_test,
    tam_incorrecto_train, fuera_rango_train,
    negras, blancas,
])

print()
if total_problemas == 0:
    print("OK CONCLUSIÓN: Dataset íntegro. No se requiere ninguna limpieza de datos.")
    print("  CIFAR-10 es un benchmark curado — todos los registros son válidos.")
else:
    print(f"! Se encontraron {total_problemas} problemas. Revisar arriba.")

print()
print("DECISIÓN DE LIMPIEZA: No aplica.")
print("No se eliminan ni modifican registros. El dataset se usa tal cual.")

