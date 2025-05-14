import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from keras.src import layers, models

# Ruta a los datos
data_path = "vidframes"

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True)
]

# Data Augmentation + Preprocesamiento compatible con MobileNetV2
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(160, 160),  # Tamaño óptimo para MobileNetV2
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)

# Base del modelo preentrenado (sin la cabeza final)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

for layer in base_model.layers:
    layer.trainable = False

# Una vez el modelo haya aprendido las nuevas capas
for layer in base_model.layers[-30:]:  # Últimas 30 capas
    layer.trainable = True

# Modelo completo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')  # Se adapta automáticamente al número de clases
])

# Compilación
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
history = model.fit(train_data, validation_data=val_data, epochs=50, callbacks=callbacks)
with open("train_history.json", "w") as f:
    json.dump(history.history, f)

model.save("gd_level_classifier.keras")
