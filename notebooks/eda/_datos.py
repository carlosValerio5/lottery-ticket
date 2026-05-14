"""Carga centralizada de CIFAR-10 via Keras. Importar desde los demás scripts."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silencia logs de TF

# Forzar UTF-8 en la salida estándar (necesario en Windows con cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from tensorflow.keras.datasets import cifar10

CLASES = ["avión", "automóvil", "pájaro", "gato", "ciervo",
          "perro", "rana", "caballo", "barco", "camión"]
N_CLASES = 10
RESULTADOS = os.path.join(os.path.dirname(__file__), "resultados")
os.makedirs(RESULTADOS, exist_ok=True)


def cargar():
    """Devuelve (x_train, y_train, x_test, y_test) como arrays numpy uint8."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test  = y_test.flatten()
    return x_train, y_train, x_test, y_test

