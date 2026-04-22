# Clasificación de perros y gatos con una RNN tipo VGG16

## Descripción del proyecto
Este proyecto desarrolla un modelo de clasificación binaria de imágenes mediante una red neuronal convolucional inspirada en la arquitectura VGG16. El objetivo es distinguir entre la clase perro y gato.

El trabajo incluye el diseño de la arquitectura, el entrenamiento del modelo piloto, el análisis del rendimiento, la evaluación del sobreajuste, la visualización de resultados y la generación de predicciones sobre nuevos casos.

## Arquitectura:
Red Connvolucional (RNN) compacta, reducida la dimensionalidad a la mitad de la VGG16 original. Con normalizacion por lotes en cada capa detras de la primera convolucional.
![Diagrama de la red](plot/VGG16-like.png)

## Base de datos:
Se utilizó el conjunto de datos Dogs vs Cats de Kaggle con 25000 imágenes etiquetadas (12500 por clase). Link:
https://www.kaggle.com/c/dogs-vs-cats/data 

Para reproducir el proyecto, es necesario descargar el dataset original y se aconseja organizarlo siguiendo la siguiente estructura y proporción:

```text
data/
├── train (80%)/
│   ├── dogs(40%)/
│   └── cats(40%)/
├── validation(10%)/
│   ├── dogs(5%)/
│   └── cats(5%)/
└── test(10%)/
```
Importante: siempre revisar la literatura para la organización de tamaño y proporción muestral, estudios de potencia, etc.

## Rendimiento y Overfiting:
Evolución de la precisión y la pérdida durante las épocas evidencia un sobreajuste en la época 8 (Significancia estadistica con prueba de diferencia de medias entre el accuracy del entrenamiento y de la validación).
![Diagrama de la red](plot/Overfitting_CNN.png)

## Prediccuones y Matriz de confusión.
El modelo clasificó correctamente 1139 gatos y 1133 perros, con 128 errores totales sobre 2400 imágenes de prueba.


|               | Predicción: gato | Predicción: perro |
|---------------|------------------:|------------------:|
| **Real: gato**  | 1139 | 61  |
| **Real: perro** | 67   | 1133 |