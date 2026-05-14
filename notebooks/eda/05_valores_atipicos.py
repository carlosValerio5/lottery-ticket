"""
PUNTO 1 (cont.) — Detección de valores atípicos
Análisis de brillo por imagen usando método IQR.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datos import cargar, CLASES, N_CLASES, RESULTADOS

x_train, y_train, x_test, y_test = cargar()

# Brillo medio por imagen (promedio de los 3 canales)
brillo = x_train.mean(axis=(1, 2, 3))  # (50000,)

q1, q3 = np.percentile(brillo, [25, 75])
iqr     = q3 - q1
lim_inf = q1 - 1.5 * iqr
lim_sup = q3 + 1.5 * iqr

atipicos_oscuros = int(np.sum(brillo < lim_inf))
atipicos_claros  = int(np.sum(brillo > lim_sup))

print("ANÁLISIS DE VALORES ATÍPICOS (método IQR sobre brillo promedio)")
print("=" * 55)
print(f"  Media            : {brillo.mean():.2f}")
print(f"  Desv. estándar   : {brillo.std():.2f}")
print(f"  Mínimo           : {brillo.min():.2f}")
print(f"  Q1               : {q1:.2f}")
print(f"  Q3               : {q3:.2f}")
print(f"  Máximo           : {brillo.max():.2f}")
print(f"  Límite inferior  : {lim_inf:.2f}")
print(f"  Límite superior  : {lim_sup:.2f}")
print()
print(f"  Imágenes atípicamente oscuras : {atipicos_oscuros}  ({atipicos_oscuros/len(brillo)*100:.2f}%)")
print(f"  Imágenes atípicamente claras  : {atipicos_claros}  ({atipicos_claros/len(brillo)*100:.2f}%)")
print()
print("Conclusión: No son errores. Son fotos naturalmente oscuras/claras.")
print("No se requiere eliminar ninguna imagen del dataset.")

# --- Boxplot + histograma ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(brillo, bins=80, color="steelblue", alpha=0.8, edgecolor="none")
axes[0].axvline(lim_inf, color="red",    linestyle="--", label=f"Lím. inf. ({lim_inf:.1f})")
axes[0].axvline(lim_sup, color="orange", linestyle="--", label=f"Lím. sup. ({lim_sup:.1f})")
axes[0].set_xlabel("Brillo promedio por imagen")
axes[0].set_ylabel("Frecuencia")
axes[0].set_title("Distribución de brillo — train set")
axes[0].legend(fontsize=8)

brillo_por_clase = [brillo[y_train == c] for c in range(N_CLASES)]
axes[1].boxplot(brillo_por_clase, labels=CLASES, vert=True)
axes[1].set_xticklabels(CLASES, rotation=35, ha="right", fontsize=8)
axes[1].set_ylabel("Brillo promedio")
axes[1].set_title("Brillo por clase (boxplot)")

fig.suptitle("Análisis de valores atípicos por brillo", fontsize=12)
fig.tight_layout()
ruta = os.path.join(RESULTADOS, "05_valores_atipicos.png")
plt.savefig(ruta, dpi=120, bbox_inches="tight")
print(f"\n-> Gráfica guardada en: {ruta}")
plt.show()

# --- Las 5 imágenes más oscuras y más claras ---
idx_oscuras = np.argsort(brillo)[:5]
idx_claras  = np.argsort(brillo)[-5:][::-1]

fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
for col, idx in enumerate(idx_oscuras):
    axes2[0][col].imshow(x_train[idx])
    axes2[0][col].set_title(f"{CLASES[y_train[idx]]}\nbr={brillo[idx]:.1f}", fontsize=8)
    axes2[0][col].axis("off")
for col, idx in enumerate(idx_claras):
    axes2[1][col].imshow(x_train[idx])
    axes2[1][col].set_title(f"{CLASES[y_train[idx]]}\nbr={brillo[idx]:.1f}", fontsize=8)
    axes2[1][col].axis("off")

axes2[0][0].set_ylabel("Más oscuras", fontsize=9)
axes2[1][0].set_ylabel("Más claras",  fontsize=9)
fig2.suptitle("Imágenes extremas por brillo", fontsize=12)
fig2.tight_layout()
ruta2 = os.path.join(RESULTADOS, "05_imagenes_extremas.png")
plt.savefig(ruta2, dpi=120, bbox_inches="tight")
print(f"-> Gráfica guardada en: {ruta2}")
plt.show()

