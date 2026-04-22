
"""
La base de datos la he dividido con la siguiente estructura:

dataset/
├── train/
│   ├── cats/
│   └── dogs/
├── train_small/
│   ├── cats/
│   └── dogs/
├── dev/
│   ├── cats/
│   └── dogs/
└── test1/

En train se encuentra todo el conjunto de ejemplares de la base de datos de kaggle.
Para poder entrenar el modelo de forma más rápida, he creado un conjunto de entrenamiento pequeño:
 - train_small: con 1000 imágenes de perrros y 1000 imágenes de gatos.
 - dev: con 50 imágenes de perros y 50 imágenes de gatos.
Ambos conjuntos se han creado de forma aleatoria, esto responde al objetivo de poder hacer ajustes empíricos en un entrenamiento rápido.
"""
import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# ============================================================
# CONFIGURACIÓN DE LAS BASES DE DATOS
# ============================================================

# Se definen dos configuraciones de carga, respectivas a su directorio:

DATASETS = {
    # 1) piloto: usada en la fase exploratoria para ajustar hiperparámetros
    "piloto": {
        "train": "dataset/piloto/train_small", # N = 1000 por clase
        "dev": "dataset/piloto/dev_small",     # N = 50 por clase
        #"test": "dataset/piloto/test"          # Comento test porque no existe (esta fase es exploratoria de ajuste).
    },
    # 2) grande: usada para el entrenamiento definitivo del modelo
    "grande": {
        "train": "dataset/grande/train",
        "dev": "dataset/grande/dev",
        "test": "dataset/grande/test"
    }
}

# SELECCIÓN DEL CONJUNTO DE DATOS
# Fase piloto realizada previamente para explorar la arquitectura:
# dataset_cfg = DATASETS["piloto"]

# Entrenamiento definitivo con el conjunto grande:
dataset_cfg = DATASETS["grande"]

TRAIN_DIR = dataset_cfg["train"]
DEV_DIR = dataset_cfg["dev"]
TEST_DIR = dataset_cfg["test"] # Comentar esto si se prueba el piloto, o añadir un test en la carpeta piloto.

# Se fija un tamaño de imagen moderado para que la red sea viable en el equipo disponible.
IMG_SIZE = (128, 128)  # Reducción del tamaño de entrada para disminuir el coste computacional.

# Se usa un mini-batch pequeño para no saturar memoria y mantener el entrenamiento estable.
BATCH_SIZE = 8

# Se usa una tasa de aprendizaje baja para favorecer una convergencia más suave.
LEARNING_RATE = 0.0001

# Ya demostró buen rendimeinto con 15, para valorar overfitting ajusto a 25.
#EPOCHS = 15
EPOCHS = 25

# Identificador del experimento para no sobrescribir historiales previos.
EXPERIMENTO_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
JSON_PATH = "historico_experimentos.json"

# ============================================================
# CALLBACK PARA GUARDAR HISTORIAL Y MÉTRICAS DE SOBREAJUSTE
# ============================================================
class GuardarHistorialJSON(Callback):
    """
    Guarda en JSON:
    1) métricas agregadas por época (accuracy, val_accuracy, loss, val_loss)
    2) métricas batch a batch sobre subconjuntos fijos de train y dev

    Esto permite:
    - representar curvas de entrenamiento
    - calcular medias móviles
    - comparar train vs dev con pruebas estadísticas para valorar sobreajuste.
    """

    def __init__(self, json_path, experimento_id, train_probe, dev_probe, n_batches_eval=20, threshold=0.5):
        super().__init__()
        self.json_path = json_path
        self.experimento_id = experimento_id
        self.train_probe = train_probe      # generador fijo de train sin augmentación
        self.dev_probe = dev_probe          # generador fijo de dev sin augmentación
        self.n_batches_eval = n_batches_eval
        self.threshold = threshold          # umbral para pasar probabilidad a clase binaria

    def _binary_crossentropy_batch(self, y_true, y_score):
        """
        Calcula la pérdida BCE de un batch.
        """
        eps = 1e-7
        y_score = np.clip(y_score, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score)
        )

    def _evaluar_batches(self, generator):
        """
        Evalúa el modelo sobre un número fijo de batches
        y devuelve accuracy y loss batch a batch.
        """
        accs = []
        losses = []

        generator.reset()
        n_batches = min(self.n_batches_eval, len(generator))

        for _ in range(n_batches):
            x_batch, y_batch = next(generator)

            # Probabilidades del modelo en el batch
            y_score = self.model.predict_on_batch(x_batch).ravel()

            # Etiquetas reales en formato binario
            if y_batch.ndim > 1:
                y_true = np.argmax(y_batch, axis=1).astype(int)
            else:
                y_true = y_batch.astype(int).ravel()

            # Predicción binaria a partir del umbral
            y_pred = (y_score >= self.threshold).astype(int)

            # Accuracy y loss del batch
            acc = np.mean(y_pred == y_true)
            loss = self._binary_crossentropy_batch(y_true, y_score)

            accs.append(float(acc))
            losses.append(float(loss))

        return accs, losses

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # ------------------------------------------------------------
        # 1) Cargar historial JSON existente
        # ------------------------------------------------------------
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        # ------------------------------------------------------------
        # 2) Evaluar train y dev batch a batch al final de la época
        # ------------------------------------------------------------
        train_acc_batches, train_loss_batches = self._evaluar_batches(self.train_probe)
        dev_acc_batches, dev_loss_batches = self._evaluar_batches(self.dev_probe)

        # Learning rate actual del optimizador
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)

        # ------------------------------------------------------------
        # 3) Guardar todo en el experimento correspondiente
        # ------------------------------------------------------------
        for exp in data:
            if exp["id"] == self.experimento_id:

                # ===== Historial agregado por época =====
                exp["epochs"].append(int(epoch + 1))                          # número de época
                exp["accuracy"].append(float(logs.get("accuracy", 0)))        # accuracy global en train
                exp["val_accuracy"].append(float(logs.get("val_accuracy", 0)))# accuracy global en dev
                exp["loss"].append(float(logs.get("loss", 0)))                # loss global en train
                exp["val_loss"].append(float(logs.get("val_loss", 0)))        # loss global en dev
                exp["learning_rate_history"].append(float(lr))                # learning rate usado en esa época

                # ===== Historial detallado para análisis de sobreajuste =====
                exp["batch_eval"].append({
                    "epoch": int(epoch + 1),

                    # Accuracy batch a batch
                    "train_acc_batches": train_acc_batches,   # lista de accuracies de train
                    "dev_acc_batches": dev_acc_batches,       # lista de accuracies de dev

                    # Loss batch a batch
                    "train_loss_batches": train_loss_batches, # lista de losses de train
                    "dev_loss_batches": dev_loss_batches,     # lista de losses de dev

                    # Resumen estadístico por época
                    "train_acc_mean": float(np.mean(train_acc_batches)),
                    "train_acc_sd": float(np.std(train_acc_batches, ddof=1)) if len(train_acc_batches) > 1 else 0.0,

                    "dev_acc_mean": float(np.mean(dev_acc_batches)),
                    "dev_acc_sd": float(np.std(dev_acc_batches, ddof=1)) if len(dev_acc_batches) > 1 else 0.0,

                    "train_loss_mean": float(np.mean(train_loss_batches)),
                    "train_loss_sd": float(np.std(train_loss_batches, ddof=1)) if len(train_loss_batches) > 1 else 0.0,

                    "dev_loss_mean": float(np.mean(dev_loss_batches)),
                    "dev_loss_sd": float(np.std(dev_loss_batches, ddof=1)) if len(dev_loss_batches) > 1 else 0.0
                })
                break

        # ------------------------------------------------------------
        # 4) Escribir el JSON actualizado
        # ------------------------------------------------------------
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=4)

# ============================================================
# REGISTRO DEL EXPERIMENTO/MODELO EN EL JSON
# ============================================================

nuevo_experimento = {
    "id": EXPERIMENTO_ID,
    "dataset_mode": "piloto" if dataset_cfg == DATASETS["piloto"] else "grande",
    "modelo": "vgg16_like_compacta",
    "train_dir": TRAIN_DIR,
    "dev_dir": DEV_DIR,
    "test_dir": TEST_DIR,
    "img_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs_total": EPOCHS,
    "epochs": [],
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": [],
    "learning_rate_history": [],
    "batch_eval": []
}

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
else:
    data = []

data.append(nuevo_experimento)

with open(JSON_PATH, "w") as f:
    json.dump(data, f, indent=4)

# ============================================================
# GENERADORES DE DATOS
# ============================================================
"""
# En la fase piloto se utilizó una augmentación moderada para aumentar la muestra
# sin distorsionar en exceso los datos originales.
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)
"""
# En el entrenamiento definitivo sobre el conjunto grande se prescinde de la augmentación y evaluar sobre-ajuste.
# Además, de reducir coste computacional.
# Para validación y test solo se aplica reescalado.
train_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

eval_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ============================================================
# PREPARACIÓN DEL DATA SET
# ============================================================
# Empleo una función por limpieza de código.
def cargar_directorio(datagen, ruta, shuffle):
    return datagen.flow_from_directory(
        directory=ruta,
        target_size=IMG_SIZE,
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

# Entrenamiento real
traindata = cargar_directorio(train_gen, TRAIN_DIR, shuffle=True)

# Validación durante el entrenamiento
devdata = cargar_directorio(eval_gen, DEV_DIR, shuffle=False)

# Test final local
testdata = None
# Control por si se prueba la fase piloto.
if TEST_DIR is not None:
    testdata = cargar_directorio(eval_gen, TEST_DIR, shuffle=False)


# Subconjuntos fijos sin augmentación para el análisis del sobreajuste
train_probe = cargar_directorio(eval_gen, TRAIN_DIR, shuffle=False)
dev_probe = cargar_directorio(eval_gen, DEV_DIR, shuffle=False)


# ============================================================
# CREACIÓN DEL MODELO
# ============================================================

# Se replica la lógica VGG: convoluciones 3x3 con padding same y pooling 2x2.
# Se reduce el número de filtros respecto a VGG16 original para ajustarlo al hardware disponible.
model = Sequential() # Apilación capa x capa.

# Bloque 1
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(128, 128, 3))) # Convolutional 2D: entrada 128x128x3 -> 128x128x32
model.add(BatchNormalization())                                  # mantiene la dimensionalidad -> 128x128x32
model.add(Conv2D(32, (3, 3), padding="same", activation="relu")) # Segunda convolución del bloque 1, salida 128x128x32
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))        # Pooling 2x2, reduce la resolución a salida 64x64x32

# Bloque 2
model.add(Conv2D(64, (3, 3), padding="same", activation="relu")) # convolucional 2D: 64x64x32 -> 64x64x64
model.add(BatchNormalization())                                  # mantiene la dimensionalidad -> 64x64x64
model.add(Conv2D(64, (3, 3), padding="same", activation="relu")) # 2nd convolucional 2d: 64x64x64 -> 64x64x64
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))        # Max pooling 2x2: reduce la resolución -> 32x32x64

# Bloque 3
model.add(Conv2D(128, (3, 3), padding="same", activation="relu")) # Capa convolucional 2D: 32x32x64 -> 32x32x128
model.add(BatchNormalization())                                   # mantiene la dimensionalidad -> 32x32x128
model.add(Conv2D(128, (3, 3), padding="same", activation="relu")) # 2nd convolucional 2d: 32x32x128 -> 32x32x128

model.add(Conv2D(128, (3, 3), padding="same", activation="relu")) # 3era convolución 2d: 32x32x128 -> 32x32x128
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))         # Max pooling 2x2: reduce la resolución -> 16x16x128

# Bloque 4
model.add(Conv2D(256, (3, 3), padding="same", activation="relu")) # Capa convolucional 2D: 16x16x128 -> 16x16x256
model.add(BatchNormalization())                                   # mantiene la dimensionalidad -> 16x16x256
model.add(Conv2D(256, (3, 3), padding="same", activation="relu")) # 2nd convolución: 16x16x256 -> 16x16x256
model.add(Conv2D(256, (3, 3), padding="same", activation="relu")) # 3era convolución: 16x16x256 -> 16x16x256
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))         # Max pooling 2x2: reduce la resolución -> 8x8x256

# Bloque 5
model.add(Conv2D(256, (3, 3), padding="same", activation="relu")) # Capa convolucional 2D: 8x8x256 -> 8x8x256
model.add(BatchNormalization())                                   # mantiene la dimensionalidad -> 8x8x256                
model.add(Conv2D(256, (3, 3), padding="same", activation="relu")) # 2nd convolución: 8x8x256 -> 8x8x256
model.add(Conv2D(256, (3, 3), padding="same", activation="relu")) # 3era convolución: 8x8x256 -> 8x8x256
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))         # Max pooling 2x2: reduce la resolución -> 4x4x256

# Cabeza fully connected
# Se conservan dos capas densas, pero reduzco a la mitad. Coherente con la reducción aplicada en las dimensiones de la VGG16 original.
model.add(Flatten())                     # Capa de aplanamiento: 4x4x256 -> vector de 4096 elementos
model.add(Dense(512, activation="relu")) # Capa densa totalmente conectada: 4096 -> 512 neuronas
model.add(Dropout(0.4))                  # Dropout para reducir sobreajuste aleatoriamente un 40%.
model.add(Dense(256, activation="relu")) # 2nd capa densa: 512 -> 256 neuronas
model.add(Dropout(0.3))                  # Otro dropout adicional del 30% para favorecer generalización.

# Se deja una sola neurona final con sigmoide porque el problema es clasificación binario (Softmax es para múltiple).
model.add(Dense(1, activation="sigmoid")) # Menos coste computacional, softamx consume dos neuronas (una por clase, en este caso).

# ============================================================
# COMPILACIÓN
# ============================================================
# Para entrada sigmoide configuro los hiperparámetros:
opt = Adam(learning_rate=LEARNING_RATE)  # Adam como punto de partida.

model.compile(
    optimizer=opt,              
    loss="binary_crossentropy",          # Función de perdida de entropía cruzada binaria.
    metrics=["accuracy"]                 # Métrica de rendimiento y ajuste.
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

# Guarda el mejor modelo según la pérdida en desarrollo.
checkpoint_best = ModelCheckpoint(
    "vgg16_like_best.keras",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min"
)

# Guarda también el último estado alcanzado al final del entrenamiento.
checkpoint_last = ModelCheckpoint(
    "vgg16_like_last.keras",
    verbose=0,
    save_best_only=False
)

# Reduce automáticamente el learning rate si la pérdida en dev se estanca.
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

# Guarda el historial agregado por época y las métricas batch a batch
# necesarias para el análisis posterior del sobreajuste.
guardador_json = GuardarHistorialJSON(
    json_path=JSON_PATH,
    experimento_id=EXPERIMENTO_ID,
    train_probe=train_probe,
    dev_probe=dev_probe,
    n_batches_eval=20
)

"""
Para estudiar el sobreajuste a lo largo de las épocas, se desactiva temporalmente el early stopping.
early = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="min",
    restore_best_weights=False
)
"""

# ============================================================
# GUARDAR INFORMACIÓN INICIAL PARA EL APARTADO 4
# ============================================================

# Se guarda el primer filtro de la primera capa convolucional antes del entrenamiento.
# Tensor Conv2D: (alto_kernel, ancho_kernel, canales_entrada, num_filtros)
kernel_init = model.layers[0].get_weights()[0][:, :, :, 0]

# Se fija una imagen de gato para usar siempre la misma en la visualización de activaciones.
gato_dir = os.path.join(TEST_DIR, "cats")
gato_files = sorted([
    f for f in os.listdir(gato_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
gato_example_path = os.path.join(gato_dir, gato_files[0])

# Se registra en el JSON del experimento.
with open(JSON_PATH, "r") as f:
    data = json.load(f)

for exp in data:
    if exp["id"] == EXPERIMENTO_ID:
        exp["dog_example_path"] = gato_example_path
        exp["kernel_init_first_conv_first_filter"] = kernel_init.tolist()
        break

with open(JSON_PATH, "w") as f:
    json.dump(data, f, indent=4)

# ============================================================
# ENTRENAMIENTO
# ============================================================

hist = model.fit(
    traindata,
    validation_data=devdata,
    epochs=EPOCHS,
    callbacks=[checkpoint_best, checkpoint_last, reduce_lr, guardador_json]         # Sin early stopping.
    # callbacks=[checkpoint_best, checkpoint_last, reduce_lr, early, guardador_json] # Con early stopping.
)

# ============================================================
# GUARDAR EL ÚLTIMO ESTADO DEL MODELO
# ============================================================

model.save("vgg16_like_overfited.keras")