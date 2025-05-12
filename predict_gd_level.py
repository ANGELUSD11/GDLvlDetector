import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import image_utils
import sys

# Cargar el modelo entrenado
model = tf.keras.models.load_model('gd_level_classifier.h5')

# Diccionario de clases (puedes ajustar según los nombres reales de tus carpetas)
class_indices = {
    'Back On Track': 0,
    'Polargeist': 1,
    'Stereo Madness': 2
}

# Invertir el diccionario para obtener el nombre desde el índice
index_to_class = {v: k for k, v in class_indices.items()}

img_path = "testing/2872315073_preview_2.jpg"

# Cargar y preparar la imagen
img = image_utils.load_img(img_path, target_size=(128, 128))
img_array = image_utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predecir
prediction = model.predict(img_array)[0]  # Primera (y única) predicción
predicted_index = np.argmax(prediction)
confidence = prediction[predicted_index] * 100

print(f"\n📷 Imagen: {img_path}")

confidence_threshold = 50.0

if confidence < confidence_threshold:
    print("No se encontró ningún nivel que coincida con la imagen dada")
else:
    # Mostrar resultado
    predicted_class = index_to_class[predicted_index]
    print(f"✅ El nivel detectado es: **{predicted_class}**")
    print(f"🔢 Confianza: {confidence:.2f}%")
