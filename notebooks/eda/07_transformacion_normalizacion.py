"""
PUNTO 3 — Limpieza, transformación y codificación de variables
Normalización, codificación de etiquetas y aumento de datos (data augmentation).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from _datos import cargar, CLASES, N_CLASES, RESULTADOS

x_train, y_train, x_test, y_test = cargar()

# ── 1. Normalización ──────────────────────────────────────────────
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

def normalizar(img_uint8: np.ndarray) -> np.ndarray:
    """uint8 (H,W,3) -> float32 normalizado por canal."""
    return (img_uint8.astype(np.float32) / 255.0 - MEAN) / STD

def desnormalizar(img_norm: np.ndarray) -> np.ndarray:
    """float32 normalizado -> uint8 para visualización."""
    return np.clip((img_norm * STD + MEAN) * 255, 0, 255).astype(np.uint8)

# Verificar estadísticas tras normalizar
muestra_norm = normalizar(x_train[:1000])
print("ESTADÍSTICAS TRAS NORMALIZACIÓN (submuestra 1000 imágenes):")
for c, nombre in enumerate(["R", "G", "B"]):
    canal = muestra_norm[:, :, :, c]
    print(f"  Canal {nombre}: media={canal.mean():.4f}  std={canal.std():.4f}  "
          f"(esperado: media≈0, std≈1)")

# ── 2. Codificación one-hot de etiquetas ─────────────────────────
def one_hot(etiquetas: np.ndarray, n: int = N_CLASES) -> np.ndarray:
    oh = np.zeros((len(etiquetas), n), dtype=np.float32)
    oh[np.arange(len(etiquetas)), etiquetas] = 1.0
    return oh

y_oh = one_hot(y_train[:5])
print("\nCODIFICACIÓN ONE-HOT (primeras 5 etiquetas):")
df_oh = pd.DataFrame(y_oh.astype(int), columns=CLASES)
df_oh.insert(0, "Etiqueta original", y_train[:5])
df_oh.insert(1, "Clase",             [CLASES[i] for i in y_train[:5]])
print(df_oh.to_string(index=False))

# ── 3. Aumento de datos (data augmentation) ───────────────────────
def random_crop(img: np.ndarray, padding: int = 4) -> np.ndarray:
    """Rellena con ceros y recorta aleatoriamente al tamaño original."""
    h, w = img.shape[:2]
    padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
    top  = np.random.randint(0, 2 * padding)
    left = np.random.randint(0, 2 * padding)
    return padded[top:top+h, left:left+w]

def horizontal_flip(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1, :]

# ── 4. Visualización comparativa ─────────────────────────────────
np.random.seed(7)
idx = np.random.randint(len(x_train))
img_orig = x_train[idx]
img_norm = normalizar(img_orig)
img_aug  = random_crop(horizontal_flip(img_orig))
img_aug_norm = normalizar(img_aug)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

axes[0].imshow(img_orig)
axes[0].set_title(f"Original\n{CLASES[y_train[idx]]} — uint8")
axes[0].axis("off")

axes[1].imshow(desnormalizar(img_norm))
axes[1].set_title("Normalizada\n(desnorm. para mostrar)")
axes[1].axis("off")

axes[2].imshow(img_aug)
axes[2].set_title("Augmentation\n(flip + crop)")
axes[2].axis("off")

axes[3].imshow(desnormalizar(img_aug_norm))
axes[3].set_title("Aug + Normalizada\n(desnorm. para mostrar)")
axes[3].axis("off")

fig.suptitle("Pipeline de transformación de imágenes", fontsize=12)
fig.tight_layout()
ruta = os.path.join(RESULTADOS, "07_transformacion_normalizacion.png")
plt.savefig(ruta, dpi=120, bbox_inches="tight")
print(f"\n-> Gráfica guardada en: {ruta}")
plt.show()

# ── 5. Resumen del pipeline ───────────────────────────────────────
print("\nPIPELINE DE PREPROCESAMIENTO APLICADO:")
pipeline = pd.DataFrame({
    "Paso": ["1", "2", "3", "4"],
    "Transformación": [
        "uint8 [0,255] -> float32 [0,1]  (÷ 255)",
        "Normalización por canal: (x − media) / std",
        "RandomCrop(32, padding=4)  [solo entrenamiento]",
        "RandomHorizontalFlip(p=0.5)  [solo entrenamiento]",
    ],
    "Aplica a": ["Train + Val + Test", "Train + Val + Test",
                 "Solo Train", "Solo Train"],
})
print(pipeline.to_string(index=False))

