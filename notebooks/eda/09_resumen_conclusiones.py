"""
PUNTO FINAL — Resumen y conclusiones del análisis exploratorio.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from _datos import cargar, CLASES, N_CLASES

x_train, y_train, x_test, y_test = cargar()

print("=" * 65)
print("RESUMEN DEL ANÁLISIS EXPLORATORIO DE DATOS — CIFAR-10")
print("=" * 65)

resumen = pd.DataFrame({
    "Aspecto": [
        "Total de imágenes",
        "Partición oficial",
        "Partición usada en el proyecto",
        "Número de clases",
        "Balance de clases",
        "Resolución de imagen",
        "Total de características (features)",
        "Tipo variable de entrada",
        "Tipo variable objetivo",
        "Valores nulos",
        "Datos inconsistentes",
        "Limpieza requerida",
        "Outliers eliminados",
        "Normalización aplicada",
        "Data augmentation (train)",
        "Separabilidad lineal (PCA)",
    ],
    "Valor / Conclusión": [
        f"{len(x_train)+len(x_test):,}",
        "50,000 train / 10,000 test",
        "45,000 train / 5,000 val / 10,000 test",
        f"{N_CLASES} (animales y vehículos mezclados)",
        "Perfectamente balanceado: 6,000 por clase",
        "32 × 32 × 3 píxeles (RGB)",
        "3,072 por imagen",
        "uint8 -> float32 normalizado",
        "Entero categórico 0-9 / one-hot",
        "0 — ninguno",
        "0 — dataset curado",
        "No requerida",
        "0 — no se eliminaron imágenes",
        "Media y std por canal RGB",
        "RandomCrop(32, pad=4) + RandomHorizontalFlip",
        "Baja — clases muy solapadas en PCA 2D",
    ],
})
print(resumen.to_string(index=False))

print()
print("=" * 65)
print("CONCLUSIONES TÉCNICAS")
print("=" * 65)
print("""
1. DATASET LIMPIO Y BALANCEADO
   CIFAR-10 no requiere ninguna limpieza. No hay valores nulos,
   etiquetas inválidas ni imágenes corruptas. La distribución
   perfectamente uniforme elimina la necesidad de técnicas de
   balanceo como SMOTE o class weighting.

2. ALTA DIFICULTAD VISUAL
   Las imágenes de 32×32 px son de baja resolución. Clases como
   gato/perro o camión/automóvil comparten paletas de color
   similares. El análisis PCA confirma que las clases NO son
   linealmente separables en el espacio de píxeles crudos.

3. ALTA VARIANZA INTRA-CLASE
   Cada clase contiene objetos en diversas poses, iluminaciones y
   fondos. Esto justifica el uso de data augmentation (flip y crop)
   y una red profunda que aprenda representaciones invariantes.

4. NORMALIZACIÓN VÁLIDA
   Los valores de media y std calculados empíricamente coinciden
   exactamente con los hardcodeados en el código del proyecto.

5. JUSTIFICACIÓN DEL MODELO
   Dado que los datos no son linealmente separables y tienen alta
   varianza intra-clase, modelos superficiales (regresión logística,
   SVM lineal) tienen limitaciones importantes. El uso de ResNet-18
   con pérdida compuesta SparseLoss está plenamente justificado.
""")

