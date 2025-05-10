import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import image_utils

data_path = "vidframes"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training')

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)
model.save('gd_level_classifier.h5')


def predict_image(img_path, model, class_indices):
    img = image_utils.load_img(img_path, target_size=(128, 128))  # Usando tensorflow.keras.preprocessing.image
    img_array = image_utils.img_to_array(img) / 255.0  # Convertir la imagen a array y normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para que sea compatible con el modelo

    prediction = model.predict(img_array)
    class_names = list(class_indices.keys())  # Obtener los nombres de las clases
    predicted_class = class_names[np.argmax(prediction)]  # Encontrar la clase con mayor probabilidad
    return predicted_class