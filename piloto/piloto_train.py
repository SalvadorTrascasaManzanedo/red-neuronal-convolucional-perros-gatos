import os
import random
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# REPRODUCIBILIDAD
# ============================================================

# Se fija una semilla para poder repetir el experimento en condiciones similares.
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# HIPERPARÁMETROS
# ============================================================

# Se mantiene el tamaño de entrada reducido para que el entrenamiento sea viable.
IMG_SIZE = (128, 128)

# Se usa un mini-batch pequeño para no saturar memoria.
BATCH_SIZE = 8

# Se usa una tasa de aprendizaje baja para estabilizar el ajuste.
LEARNING_RATE = 0.0001

# Se limita el número de épocas porque aquí interesa visualizar filtros y activaciones.
EPOCHS = 15

# Se usa train_small para acelerar esta fase específica del apartado 4.
TRAIN_DIR = "dataset/piloto/train_small"
DEV_DIR = "dataset/piloto/dev"

# ============================================================
# DATOS
# ============================================================

# Se añade una augmentación moderada para favorecer la generalización.
trdata = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

traindata = trdata.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMG_SIZE,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# Se deja el conjunto de desarrollo solo con reescalado.
tsdata = ImageDataGenerator(rescale=1.0 / 255)

testdata = tsdata.flow_from_directory(
    directory=DEV_DIR,
    target_size=IMG_SIZE,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED
)

# ============================================================
# MODELO TIPO VGG16 ADAPTADO
# ============================================================

# Se mantiene la lógica VGG: convoluciones 3x3 con padding same y pooling 2x2.
model = Sequential()

# Bloque 1
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(128, 128, 3), name="conv1_1"))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Bloque 2
model.add(Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Bloque 3
model.add(Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2"))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_3"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Bloque 4
model.add(Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1"))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_2"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_3"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Bloque 5
model.add(Conv2D(256, (3, 3), padding="same", activation="relu", name="conv5_1"))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", activation="relu", name="conv5_2"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu", name="conv5_3"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Parte fully connected
# Se reduce la cabeza densa respecto a VGG16 original para adaptarla al hardware disponible.
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.4))  # Se usa dropout para reducir sobreajuste.
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))  # Se añade una segunda regularización antes de la salida.
model.add(Dense(1, activation="sigmoid"))  # Se usa sigmoide porque el problema es binario.

opt = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
model.build((None, 128, 128, 3))  # Se construye la red para poder extraer activaciones sin error.

# ============================================================
# IMAGEN DE PERRO PARA LAS ACTIVACIONES
# ============================================================

# Se toma siempre la primera imagen de perros en dev para usar una referencia fija.
DOG_DIR = os.path.join(DEV_DIR, "dogs")
dog_files = sorted([
    f for f in os.listdir(DOG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if len(dog_files) == 0:
    raise ValueError("No se han encontrado imágenes en dataset/dev/dogs")

dog_path = os.path.join(DOG_DIR, dog_files[0])

img = image.load_img(dog_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# ============================================================
# FILTRO Y ACTIVACIÓN INICIALES
# ============================================================

# Se guardan los pesos iniciales antes del entrenamiento.
first_conv_layer = model.get_layer("conv1_1")
initial_weights = first_conv_layer.get_weights()[0]

# Se toma el primer filtro de la primera capa convolucional.
initial_first_filter = initial_weights[:, :, :, 0]

# Se extrae el mapa de activación inicial del primer filtro de la primera capa.
activation_model_first = Model(inputs=model.inputs, outputs=first_conv_layer.output)
initial_first_activation = activation_model_first.predict(img_batch, verbose=0)[0, :, :, 0]

# ============================================================
# ENTRENAMIENTO
# ============================================================

# Se usa early stopping para detener el aprendizaje cuando dev deja de mejorar.
early = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    verbose=1,
    mode="max",
    restore_best_weights=True
)

# Se reduce el learning rate cuando la mejora se estanca.
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

hist = model.fit(
    traindata,
    validation_data=testdata,
    epochs=EPOCHS,
    callbacks=[early, reduce_lr],
    verbose=1
)

# ============================================================
# FILTROS Y ACTIVACIONES FINALES
# ============================================================

# Se obtiene el primer filtro de la primera capa al final del aprendizaje.
final_weights_first = model.get_layer("conv1_1").get_weights()[0]
final_first_filter = final_weights_first[:, :, :, 0]

# Se obtiene el primer filtro de la última capa convolucional al final del aprendizaje.
last_conv_layer = model.get_layer("conv5_3")
final_weights_last = last_conv_layer.get_weights()[0]
final_last_filter = final_weights_last[:, :, :, 0]

# Se obtiene el mapa de activación final del primer filtro de la primera capa.
activation_model_first_final = Model(inputs=model.inputs, outputs=model.get_layer("conv1_1").output)
final_first_activation = activation_model_first_final.predict(img_batch, verbose=0)[0, :, :, 0]

# Se obtiene el mapa de activación final del primer filtro de la última capa convolucional.
activation_model_last_final = Model(inputs=model.inputs, outputs=model.get_layer("conv5_3").output)
final_last_activation = activation_model_last_final.predict(img_batch, verbose=0)[0, :, :, 0]

# ============================================================
# GUARDADO DE RESULTADOS PARA PLOTEO Y ANÁLISIS
# ============================================================

# Se guardan los arrays para poder analizar y dibujar en un script separado sin reentrenar.
np.save("initial_first_filter.npy", initial_first_filter)
np.save("initial_first_activation.npy", initial_first_activation)

np.save("final_first_filter.npy", final_first_filter)
np.save("final_first_activation.npy", final_first_activation)

np.save("final_last_filter.npy", final_last_filter)
np.save("final_last_activation.npy", final_last_activation)

# Se guardan etiquetas reales y probabilidades para construir la ROC después.
y_score = model.predict(testdata, verbose=0).ravel().tolist()
y_true = testdata.classes.tolist()

# Se guarda todo en un único JSON para poder plotear después sin reentrenar.
historial = hist.history
historial["y_true"] = y_true
historial["y_score"] = y_score

with open("historial_apartado4.json", "w") as f:
    json.dump(historial, f, indent=4)

# Se guarda el modelo entrenado por si hiciera falta reutilizarlo.
model.save("modelo_apartado4.keras")

print("Se han guardado los archivos del apartado 4:")
print("- initial_first_filter.npy")
print("- initial_first_activation.npy")
print("- final_first_filter.npy")
print("- final_first_activation.npy")
print("- final_last_filter.npy")
print("- final_last_activation.npy")
print("- historial_apartado4.json")
print("- modelo_apartado4.keras")