from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Carpeta donde está este script
BASE_DIR = Path(__file__).resolve().parent

# =========================
# FUNCIONES AUXILIARES
# =========================
def cargar_npy(nombre):
    return np.load(BASE_DIR / nombre)

def normalizar(x):
    x = x.copy()
    if x.max() != x.min():
        x = (x - x.min()) / (x.max() - x.min())
    return x

# =========================
# CARGA DE ARCHIVOS
# =========================
with open(BASE_DIR / "historial_apartado4.json", "r", encoding="utf-8") as f:
    hist = json.load(f)

epochs = list(range(1, len(hist["accuracy"]) + 1))

initial_first_filter = cargar_npy("initial_first_filter.npy")
initial_first_activation = cargar_npy("initial_first_activation.npy")

final_first_filter = cargar_npy("final_first_filter.npy")
final_first_activation = cargar_npy("final_first_activation.npy")

final_last_filter = cargar_npy("final_last_filter.npy")
final_last_activation = cargar_npy("final_last_activation.npy")

# El filtro de la última capa tiene muchos canales, así que lo resumimos a 2D
final_last_filter_2d = np.mean(np.abs(final_last_filter), axis=2)

# =========================
# 1) ACCURACY + LOSS EN UNA SOLA FIGURA
# =========================
plt.figure(figsize=(8, 5))

plt.plot(epochs, hist["accuracy"], marker="o", linewidth=2, label="Train accuracy")
plt.plot(epochs, hist["val_accuracy"], marker="s", linewidth=2, label="Dev accuracy")
plt.plot(epochs, hist["loss"], marker="o", linewidth=2, label="Train loss")
plt.plot(epochs, hist["val_loss"], marker="s", linewidth=2, label="Dev loss")

plt.title("Precisión y pérdida en entrenamiento/validación")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_DIR / "accuracy_y_loss_apartado4.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 2) FILTRO INICIAL + ACTIVACIÓN INICIAL
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(normalizar(initial_first_filter))
plt.title("Filtro inicial\nPrimera capa, filtro 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(normalizar(initial_first_activation), cmap="viridis")
plt.title("Mapa de activación inicial\nPrimera capa, filtro 1")
plt.axis("off")

plt.tight_layout()
plt.savefig(BASE_DIR / "figura_4a_filtro_inicial_primera_capa.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 3) FILTRO FINAL PRIMERA CAPA + ACTIVACIÓN FINAL
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(normalizar(final_first_filter))
plt.title("Filtro final\nPrimera capa, filtro 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(normalizar(final_first_activation), cmap="viridis")
plt.title("Mapa de activación final\nPrimera capa, filtro 1")
plt.axis("off")

plt.tight_layout()
plt.savefig(BASE_DIR / "figura_4b_filtro_final_primera_capa.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 4) FILTRO FINAL ÚLTIMA CAPA + ACTIVACIÓN FINAL
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(normalizar(final_last_filter_2d), cmap="viridis")
plt.title("Filtro final\nÚltima capa, filtro 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(normalizar(final_last_activation), cmap="viridis")
plt.title("Mapa de activación final\nÚltima capa, filtro 1")
plt.axis("off")

plt.tight_layout()
plt.savefig(BASE_DIR / "figura_4c_filtro_final_ultima_capa.png", dpi=300, bbox_inches="tight")
plt.show()