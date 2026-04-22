import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = "vgg16_like_best.keras"     # cambia a vgg16_like_overfited.keras si quieres ese
TEST_DIR = "dataset/grande/test"         # carpeta test con subcarpetas cats/ y dogs/
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
THRESHOLD = 0.5

# ============================================================
# CARGA DEL MODELO
# ============================================================

model = load_model(MODEL_PATH)

# ============================================================
# GENERADOR DE TEST
# ============================================================

test_gen = ImageDataGenerator(rescale=1.0 / 255)

testdata = test_gen.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMG_SIZE,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# PREDICCIONES
# ============================================================

y_score = model.predict(testdata, verbose=1).ravel()
y_pred = (y_score >= THRESHOLD).astype(int)
y_true = testdata.classes

# ============================================================
# MÉTRICAS
# ============================================================

acc = np.mean(y_pred == y_true)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["cats", "dogs"])
auc = roc_auc_score(y_true, y_score)

print("\nRESULTADOS EN TEST")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print("\nMatriz de confusión:")
print(cm)
print("\nClassification report:")
print(report)

# ============================================================
# GUARDAR PREDICCIONES EN CSV
# ============================================================

idx_to_class = {v: k for k, v in testdata.class_indices.items()}
true_labels = [idx_to_class[i] for i in y_true]
pred_labels = [idx_to_class[i] for i in y_pred]

resultados = pd.DataFrame({
    "filename": testdata.filenames,
    "true_label": true_labels,
    "pred_score_dog": y_score,
    "pred_label": pred_labels,
    "correct": y_true == y_pred
})

resultados.to_csv("predicciones_test.csv", index=False, encoding="utf-8")
print("\nArchivo guardado: predicciones_test.csv")