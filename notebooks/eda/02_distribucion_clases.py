"""
PUNTO 1 (cont.) — Distribución de la variable objetivo
Balance de clases en train y test.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datos import cargar, CLASES, N_CLASES, RESULTADOS

x_train, y_train, x_test, y_test = cargar()

train_counts = np.bincount(y_train, minlength=N_CLASES)
test_counts  = np.bincount(y_test,  minlength=N_CLASES)

dist_df = pd.DataFrame({
    "Clase":   CLASES,
    "Train":   train_counts,
    "Test":    test_counts,
    "Total":   train_counts + test_counts,
    "% Train": (train_counts / len(y_train) * 100).round(2),
    "% Test":  (test_counts  / len(y_test)  * 100).round(2),
})
print("DISTRIBUCIÓN DE CLASES:")
print(dist_df.to_string(index=False))
print(f"\nDesv. estándar conteos train : {train_counts.std():.4f}")
print("-> Dataset PERFECTAMENTE BALANCEADO (misma cantidad en todas las clases)")

# --- Gráfica ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(N_CLASES)
w = 0.35
axes[0].bar(x - w/2, train_counts, w, label="Train", color="steelblue", alpha=0.85)
axes[0].bar(x + w/2, test_counts,  w, label="Test",  color="coral",     alpha=0.85)
axes[0].axhline(train_counts.mean(), color="navy", linestyle="--", alpha=0.6, label="Media train")
axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASES, rotation=35, ha="right", fontsize=9)
axes[0].set_ylabel("Cantidad de imágenes")
axes[0].set_title("Distribución de clases — Train vs Test")
axes[0].legend()

axes[1].pie(train_counts + test_counts, labels=CLASES,
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
axes[1].set_title("Proporción total por clase")

fig.suptitle("Balance de clases en CIFAR-10", fontsize=13)
fig.tight_layout()
ruta = os.path.join(RESULTADOS, "02_distribucion_clases.png")
plt.savefig(ruta, dpi=120, bbox_inches="tight")
print(f"\n-> Gráfica guardada en: {ruta}")
plt.show()

