import base64
import os
from mimetypes import guess_type
import streamlit as st
from openai import AzureOpenAI

class ImageClassificator:
    def __init__(self):
        # Configuración de la API
        self.api_base = st.secrets["AZURE_OAI_ENDPOINT"]
        self.api_key = st.secrets["AZURE_OAI_KEY"]
        self.deployment_name = st.secrets["AZURE_OAI_DEPLOYMENT"]
        self.api_version = "2024-02-15-preview"

        # Inicializar el cliente de Azure OpenAI
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=f"{self.api_base}/openai/deployments/{self.deployment_name}"
        )

        # Rutas predefinidas para las imágenes de ejemplo
        self.desorganizado_path = r'ImagenesEntrenamiento\desorganizado.jpg'
        self.organizado_path = r'ImagenesEntrenamiento\organizado.jpg'
        self.organizado_2_path = r'ImagenesEntrenamiento\organizado.jpg'

    def local_image_to_data_url(self, image_path):
        """Codifica una imagen local en formato de data URL."""
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def clasificar_pasillo(self, imagen_evaluar_path):
        """Clasifica la imagen del pasillo en función de los ejemplos proporcionados."""
        # Codificar las imágenes en formato de data URL
        desorganizado_data_url = self.local_image_to_data_url(self.desorganizado_path)
        organizado_data_url = self.local_image_to_data_url(self.organizado_path)
        organizado_2_path_data_url = self.local_image_to_data_url(self.organizado_2_path)
        imagen_evaluar_data_url = self.local_image_to_data_url(imagen_evaluar_path)

        # Crear la lista de mensajes para enviar a la API
        messages = [
            { "role": "system", "content": """
             Tu objetivo es decir si un pasillo está organizado o no con base en los ejemplos que se te pasaran.
             Tipo de clasificación:
             * Mal organizado
             * Medianamente organizado
             * Organziado
             """ },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un pasillo desorganizado" },
                    { "type": "image_url", "image_url": { "url": desorganizado_data_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un pasillo organizado" },
                    { "type": "image_url", "image_url": { "url": organizado_data_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un pasillo medianamente organizado" },
                    { "type": "image_url", "image_url": { "url": organizado_2_path_data_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": """
                    Basándote en los ejemplos anteriores, clasifica la siguiente imagen. 
                    
                    El formato será el siguiente:

                    Decisión: (Puedes devolver Pasillo organizado / Pasillo desorganizado)
                    
                    Descripción: (detalladamente, qué ves en la imagen)

                    Justificación: (Aquí pones la justificación en un párrafo o en lista, como creas conveniente.)

                    Recomendaciones: (Una breve recomendación de qué mejorar para que esté organizado, solo en caso de que esté desorganizado)
                    """ },
                    { "type": "image_url", "image_url": { "url": imagen_evaluar_data_url } }
                ]
            }
        ]

        # Enviar la solicitud a la API
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=2000
        )

        # Devolver solo el contenido de la respuesta
        return response.choices[0].message.content

# Ejemplo de uso
# clasificador = ImageClassificator()
# resultado = clasificador.clasificar_pasillo(imagen_evaluar_path=r'ImagenesPreCargadas\71913150_ZAsw8uWgddbYuzBYMNkI9xFGaOqy08W6h4J_3uuI3ZA.jpg')
# print(resultado)
