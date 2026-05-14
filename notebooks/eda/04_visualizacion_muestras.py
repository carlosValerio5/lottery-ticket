"""
PUNTO 1 (cont.) — Visualización de muestras e imágenes promedio por clase.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from _datos import cargar, CLASES, N_CLASES, RESULTADOS

x_train, y_train, x_test, y_test = cargar()
np.random.seed(42)

# --- 5 muestras por clase ---
n_muestras = 5
fig, axes = plt.subplots(N_CLASES, n_muestras, figsize=(n_muestras * 1.8, N_CLASES * 1.8))

for clase_id in range(N_CLASES):
    indices = np.where(y_train == clase_id)[0]
    seleccionados = np.random.choice(indices, n_muestras, replace=False)
    for col, idx in enumerate(seleccionados):
        ax = axes[clase_id][col]
        ax.imshow(x_train[idx])
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(CLASES[clase_id], fontsize=9,
                          rotation=0, labelpad=45, va="center")

fig.suptitle("5 muestras aleatorias por clase — CIFAR-10 (32×32 px)", fontsize=12, y=1.01)
fig.tight_layout()
ruta = os.path.join(RESULTADOS, "04_muestras_por_clase.png")
plt.savefig(ruta, dpi=120, bbox_inches="tight")
print(f"-> Gráfica guardada en: {ruta}")
plt.show()

# --- Imagen promedio por clase ---
fig2, axes2 = plt.subplots(2, 5, figsize=(13, 5))
axes_flat = axes2.flatten()

for clase_id in range(N_CLASES):
    promedio = x_train[y_train == clase_id].mean(axis=0).astype(np.uint8)
    axes_flat[clase_id].imshow(promedio)
    axes_flat[clase_id].set_title(
        f"{CLASES[clase_id]}\n(n={np.sum(y_train==clase_id):,})", fontsize=9)
    axes_flat[clase_id].axis("off")

fig2.suptitle("Imagen promedio por clase (train set)", fontsize=13)
fig2.tight_layout()
ruta2 = os.path.join(RESULTADOS, "04_imagenes_promedio.png")
plt.savefig(ruta2, dpi=120, bbox_inches="tight")
print(f"-> Gráfica guardada en: {ruta2}")
plt.show()

print("\nObservaciones:")
print("• Avión y barco -> fondos azules/grises dominantes.")
print("• Los promedios son borrosos -> alta varianza intra-clase.")
print("• Varias clases comparten paletas de color -> el color solo no basta para clasificar.")

