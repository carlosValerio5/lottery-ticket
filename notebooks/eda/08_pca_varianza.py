"""
PUNTO 1 (cont.) — Reducción de dimensionalidad con PCA
Justifica por qué se necesita una red neuronal (no separabilidad lineal).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from _datos import cargar, CLASES, N_CLASES, RESULTADOS

x_train, y_train, x_test, y_test = cargar()

# Submuestra para PCA (más rápido)
N_PCA = 5000
np.random.seed(42)
idx = np.random.choice(len(x_train), N_PCA, replace=False)
X   = x_train[idx].reshape(N_PCA, -1).astype(np.float32) / 255.0
y   = y_train[idx]

print(f"Aplicando PCA a {N_PCA} imágenes de {X.shape[1]} dimensiones...")
X_scaled = StandardScaler().fit_transform(X)

pca50 = PCA(n_components=50, random_state=42)
X_50  = pca50.fit_transform(X_scaled)
var_acum = np.cumsum(pca50.explained_variance_ratio_)

print(f"Varianza explicada con  2 componentes: {var_acum[1]*100:.2f}%")
print(f"Varianza explicada con 10 componentes: {var_acum[9]*100:.2f}%")
print(f"Varianza explicada con 50 componentes: {var_acum[49]*100:.2f}%")
print()
print("-> Se necesitan MUCHOS componentes para capturar la varianza.")
print("-> Los datos NO son linealmente separables -> justifica el uso de CNN (ResNet-18).")

# ── Varianza intra-clase ──────────────────────────────────────────
print("\nVARIANZA INTRA-CLASE (mayor = más diversidad visual dentro de la clase):")
for clase_id in range(N_CLASES):
    v = x_train[y_train == clase_id].astype(np.float32).var(axis=0).mean()
    print(f"  {CLASES[clase_id]:<12}: {v:.2f}")

# ── Gráficas ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Varianza acumulada
axes[0].plot(range(1, 51), var_acum * 100, "o-", markersize=3, color="C0")
axes[0].axhline(80, color="red",    linestyle="--", alpha=0.7, label="80%")
axes[0].axhline(90, color="orange", linestyle="--", alpha=0.7, label="90%")
axes[0].set_xlabel("Número de componentes")
axes[0].set_ylabel("Varianza explicada acumulada (%)")
axes[0].set_title("Varianza acumulada por PCA")
axes[0].legend()

# Scatter PCA 2D
cmap = plt.get_cmap("tab10")
for clase_id in range(N_CLASES):
    mask = y == clase_id
    axes[1].scatter(X_50[mask, 0], X_50[mask, 1],
                    c=[cmap(clase_id)], alpha=0.3, s=8, label=CLASES[clase_id])
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("PCA 2D — primeras 2 componentes")
axes[1].legend(markerscale=3, fontsize=8, ncol=2)

fig.suptitle("PCA sobre CIFAR-10 (5,000 imágenes)", fontsize=12)
fig.tight_layout()
ruta = os.path.join(RESULTADOS, "08_pca.png")
plt.savefig(ruta, dpi=120, bbox_inches="tight")
print(f"\n-> Gráfica guardada en: {ruta}")
plt.show()

# ── Mapas de desviación estándar ──────────────────────────────────
fig2, axes2 = plt.subplots(2, 5, figsize=(13, 5))
for clase_id, ax in enumerate(axes2.flatten()):
    std_map = x_train[y_train == clase_id].astype(np.float32).std(axis=0).mean(axis=2)
    im = ax.imshow(std_map, cmap="hot", vmin=0, vmax=80)
    ax.set_title(CLASES[clase_id], fontsize=9)
    ax.axis("off")

fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.6, label="Desv. estándar del píxel")
fig2.suptitle("Mapa de variabilidad por píxel y clase", fontsize=11)
ruta2 = os.path.join(RESULTADOS, "08_varianza_por_clase.png")
plt.savefig(ruta2, dpi=120, bbox_inches="tight")
print(f"-> Gráfica guardada en: {ruta2}")
plt.show()

