import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

import json
with open("categorias.json", "r") as f:
    categorias = json.load(f)

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('modelo_granos_cafe.h5')

# Función para preprocesar la imagen
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((120, 120))  # Ajusta esto si tu modelo requiere otro tamaño
    img_array = np.array(image) / 255.0  # Normalizar
    if img_array.shape[-1] == 4:  # Eliminar canal alfa si existe
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión para batch
    return img_array

# Interfaz de usuario
st.title("Clasificación de Granos de Café ☕")
st.write("Sube una imagen de un grano de café para predecir si es bueno o malo.")

uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    with st.spinner("Realizando predicción..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction[0])
        predicted_class = categorias[predicted_index]
        confidence = prediction[0][predicted_index]

        st.success(f"Resultado: **{predicted_class}** ({confidence * 100:.2f}% de confianza)")
