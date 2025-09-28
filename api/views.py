import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io, base64
import cv2

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Cargar el modelo al iniciar el servidor
model = tf.keras.models.load_model("trained_model.h5")

class PredictDigit(APIView):
    def preprocess_image(self, image):
        # Convertir PIL a numpy array
        img_array = np.array(image)
        
        # Encontrar el bounding box del contenido (píxeles blancos o no negros)
        coords = cv2.findNonZero(img_array)
        if coords is None:
            # Si no hay contenido, devolver imagen vacía
            return np.zeros((28, 28), dtype=np.float32)
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Extraer la región del dígito
        digit_region = img_array[y:y+h, x:x+w]
        
        # Convertir de nuevo a PIL para redimensionamiento
        digit_pil = Image.fromarray(digit_region)
        
        # Calcular nuevo tamaño manteniendo aspect ratio
        # Tamaño máximo de la imagen es 20 píxeles + 4 de padding
        max_size = 20
        width, height = digit_pil.size
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        # Redimensionar manteniendo suavizado
        digit_resized = digit_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crear canvas de 28x28 negro
        final_image = Image.new('L', (28, 28), 0)
        
        # Calcular posición para centrar el dígito
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        
        # Pegar el dígito centrado
        final_image.paste(digit_resized, (paste_x, paste_y))
        return np.array(final_image).astype(np.float32)


    def post(self, request):
        try:
            data = request.data.get("image")
            if not data:
                return Response({"error": "No se envió la imagen"}, status=status.HTTP_400_BAD_REQUEST)

            # Decodificar la imagen base64
            image_data = base64.b64decode(data.split(",")[1])
            image = Image.open(io.BytesIO(image_data)).convert("L")  # escala de grises

            # Invertir colores si el fondo es blanco (con el fin de que quede fondo negro, dígito blanco)
            img_array = np.array(image)
            if np.mean(img_array) > 127: 
                image = ImageOps.invert(image)

            # Preprocesar la imagen con el método creado arriba
            processed_array = self.preprocess_image(image)

            # Normalizar (0-1)
            processed_array = processed_array / 255.0

            # Expandir dimensiones para batch y canales
            img_array = np.expand_dims(processed_array, axis=(0, -1))

            # Predicción
            pred = model.predict(img_array)
            digit = int(np.argmax(pred))
            confidence = float(np.max(pred))

            return Response({
                "prediction": digit, 
                "confidence": confidence
            })

        except Exception as e:
            print(f"Error en predicción: {str(e)}") 
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)