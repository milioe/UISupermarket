import os
import streamlit as st
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from openai import AsyncAzureOpenAI
import asyncio
import time

# Configurar Streamlit para que se muestre en modo "wide"
st.set_page_config(layout="wide")

# Configuración de Azure OpenAI usando secretos de Streamlit
azure_oai_endpoint = st.secrets["AZURE_OAI_ENDPOINT"]
azure_oai_key = st.secrets["AZURE_OAI_KEY"]
azure_oai_deployment = st.secrets["AZURE_OAI_DEPLOYMENT"]
api_version = "2024-02-15-preview"

# Crear una instancia del cliente de Azure OpenAI
client = AsyncAzureOpenAI(
    azure_endpoint=azure_oai_endpoint, 
    api_key=azure_oai_key,  
    api_version=api_version
)

# Mensaje del sistema y rutas de imágenes predefinidas
SYSTEM_MESSAGE = open(file="system.txt", encoding="utf8").read().strip()
PREDEFINED_IMAGE_PATHS = [
    "ImagenesEntrenamiento/organizado.jpg",
    "ImagenesEntrenamiento/organizado2.jpg",
    "ImagenesEntrenamiento/desorganizado.jpg", 
]


# Directorio de imágenes pre-cargadas
PRELOADED_IMAGES_DIR = r"ImagenesPreCargadas"

# Función para procesar imágenes locales
def process_local_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
        return f"Image processed from path: {image_path}"
    except UnidentifiedImageError:
        raise ValueError("No se pudo identificar la imagen en el path proporcionado.")

# Función para procesar la imagen subida por el usuario
def process_uploaded_image(image_file) -> str:
    try:
        with Image.open(image_file) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
        return f"User uploaded image processed: {image_file.name}"
    except UnidentifiedImageError:
        raise ValueError("No se pudo identificar la imagen subida.")

# Función asíncrona para llamar al modelo de Azure OpenAI
async def call_azure_openai(client, messages):
    try:
        response = await client.chat.completions.create(
            model=azure_oai_deployment,
            messages=messages,
            temperature=0,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error al llamar al modelo de Azure OpenAI: {str(e)}")

# Interfaz de Streamlit
st.title("Detección de limpieza con GPT")

# Sidebar para drag and drop y selección de imágenes pre-cargadas
with st.sidebar:
    # Subir una imagen
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    st.write("**... o selecciona una imagen pre-cargada:**")
    
    # Listar imágenes pre-cargadas en el directorio
    preloaded_images = os.listdir(PRELOADED_IMAGES_DIR)
    selected_preloaded_image = None
    
    # Enumerar y mostrar botones con nombres "Imagen 1", "Imagen 2", etc.
    for index, image_name in enumerate(preloaded_images):
        display_name = f"Imagen {index + 1}"
        if st.button(display_name):
            selected_preloaded_image = os.path.join(PRELOADED_IMAGES_DIR, image_name)

# Crear columnas para mostrar la imagen y la respuesta lado a lado
col1, col2 = st.columns(2)

if uploaded_file or selected_preloaded_image:
    try:
        with col1:
            # Mostrar la imagen seleccionada o subida
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen Analizada", use_column_width=True)
            elif selected_preloaded_image:
                image = Image.open(selected_preloaded_image)
                st.image(image, caption="Imagen Pre-cargada", use_column_width=True)

        with col2:
            # Procesar imágenes predefinidas
            predefined_image_contexts = [process_local_image(image_path) for image_path in PREDEFINED_IMAGE_PATHS]
            
            # Procesar la imagen seleccionada o subida
            if uploaded_file:
                user_image_context = process_uploaded_image(uploaded_file)
            elif selected_preloaded_image:
                user_image_context = process_local_image(selected_preloaded_image)
            
            prompt = f"{SYSTEM_MESSAGE}\n\n{predefined_image_contexts}"
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_image_context},
            ]
            
            # Agregar un spinner mientras se genera la respuesta
            with st.spinner("Generando respuesta..."):
                # Llamar al modelo de Azure OpenAI y obtener la respuesta generada
                generated_text = asyncio.run(call_azure_openai(client, messages))
            
            # Mostrar la respuesta
            st.write(generated_text)

    except (ValueError, RuntimeError) as e:
        st.toast(f"🚨 No se ha podido procesar la imagen: {str(e)}")
