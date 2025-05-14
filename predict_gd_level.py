import numpy as np
import json
import tensorflow as tf
from keras.src.legacy.preprocessing.image import image_utils
from keras.src.applications.mobilenet_v2 import preprocess_input

# Diccionario de clases (puedes ajustar según los nombres reales de tus carpetas)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Invertir el diccionario para obtener el nombre desde el índice
index_to_class = {v: k for k, v in class_indices.items()}

img_path = "testing/dash.jpg"

# Cargar y preparar la imagen
img = image_utils.load_img(img_path, target_size=(160, 160))
img_array = image_utils.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

def reload_model(model_path='gd_level_classifier.keras'):
    return tf.keras.models.load_model(model_path)
model = reload_model()
print(model.summary())

# Predecir
prediction = model.predict(img_array)[0]

print(f"\n📷 Imagen: {img_path}")
print("🎯 Probabilidades por clase:\n")

# Ordenar índices por probabilidad descendente
sorted_indices = np.argsort(prediction)[::-1]

# Mostrar todas las clases con su porcentaje
for i in sorted_indices:
    class_name = index_to_class[i]
    prob = prediction[i] * 100
    print(f"{class_name:30} → {prob:.2f}%")
