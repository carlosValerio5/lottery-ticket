"""
PUNTO 1 (cont.) — Estadísticas descriptivas por canal RGB
Media, desviación estándar, cuartiles, histogramas por canal.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datos import cargar, CLASES, N_CLASES, RESULTADOS

x_train, y_train, x_test, y_test = cargar()

# --- Tabla estadísticas por canal ---
filas = []
for c, nombre in enumerate(["Rojo (R)", "Verde (G)", "Azul (B)"]):
    canal = x_train[:, :, :, c].astype(np.float32)
    filas.append({
        "Canal":          nombre,
        "Media":          round(canal.mean(), 3),
        "Desv. Estándar": round(canal.std(),  3),
        "Mínimo":         int(canal.min()),
        "Q25":            round(float(np.percentile(canal, 25)), 1),
        "Mediana":        round(float(np.median(canal)), 1),
        "Q75":            round(float(np.percentile(canal, 75)), 1),
        "Máximo":         int(canal.max()),
    })

stats_df = pd.DataFrame(filas)
print("ESTADÍSTICAS DESCRIPTIVAS POR CANAL RGB (valores 0-255):")
print(stats_df.to_string(index=False))

# Valores de normalización calculados empíricamente
means = x_train.mean(axis=(0, 1, 2)) / 255.0
stds  = x_train.std(axis=(0, 1, 2))  / 255.0
print(f"\nMedia por canal (normalizada a [0,1]): R={means[0]:.4f}  G={means[1]:.4f}  B={means[2]:.4f}")
print(f"Desv. estándar  (normalizada a [0,1]): R={stds[0]:.4f}  G={stds[1]:.4f}  B={stds[2]:.4f}")
print("(Estos valores se usan en el preprocesamiento del modelo)")

# --- Histogramas por canal ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colores = ["#E74C3C", "#2ECC71", "#3498DB"]
nombres = ["Canal Rojo (R)", "Canal Verde (G)", "Canal Azul (B)"]

for c, (color, nombre) in enumerate(zip(colores, nombres)):
    canal = x_train[:, :, :, c].flatten()
    axes[c].hist(canal, bins=64, color=color, alpha=0.75, edgecolor="none")
    axes[c].axvline(canal.mean(),    color="black", linestyle="--", lw=1.5,
                    label=f"Media={canal.mean():.1f}")
    axes[c].axvline(np.median(canal), color="gray",  linestyle=":",  lw=1.5,
                    label=f"Mediana={np.median(canal):.1f}")
    axes[c].set_title(nombre)
    axes[c].set_xlabel("Valor de píxel (0-255)")
    axes[c].set_ylabel("Frecuencia")
    axes[c].legend(fontsize=8)

fig.suptitle("Distribución de valores de píxel por canal RGB", fontsize=12)
fig.tight_layout()
ruta = os.path.join(RESULTADOS, "03_estadisticas_descriptivas.png")
plt.savefig(ruta, dpi=120, bbox_inches="tight")
print(f"\n-> Gráfica guardada en: {ruta}")
plt.show()

# --- Histograma canal R por clase ---
fig2, ax = plt.subplots(figsize=(12, 5))
cmap = plt.get_cmap("tab10")
for clase_id in range(N_CLASES):
    canal_r = x_train[y_train == clase_id, :, :, 0].flatten()
    ax.hist(canal_r, bins=50, alpha=0.4, color=cmap(clase_id),
            label=CLASES[clase_id], density=True)
ax.set_xlabel("Valor de píxel — Canal Rojo")
ax.set_ylabel("Densidad")
ax.set_title("Distribución del Canal Rojo por clase")
ax.legend(ncol=2, fontsize=8)
fig2.tight_layout()
ruta2 = os.path.join(RESULTADOS, "03_canal_rojo_por_clase.png")
plt.savefig(ruta2, dpi=120, bbox_inches="tight")
print(f"-> Gráfica guardada en: {ruta2}")
plt.show()

