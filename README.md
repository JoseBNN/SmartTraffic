Guía de Usuario – Sistema de Detección de Vehículos en Carriles Exclusivos
 Este documento describe el uso del sistema SmartTraffic para la detección de invasión de carriles exclusivos utilizando modelos YOLO integrados en un servidor FastAPI y una interfaz web.
 1. Introducción|
 El sistema permite cargar una imagen, detectar vehículos y carriles exclusivos, y determinar si existe una invasión al carril. Presenta resultados visuales y un estado final.
 
 2. Requisitos- Python 3.10+- FastAPI, Uvicorn, OpenCV, Ultralytics- Modelos best_.pt y best.pt- Navegador web actualizado
 3. Cómo iniciar el sistema
 Ejecutar:
 uvicorn main:app --reload
 Luego ingresar a: http://127.0.0.1:8000
 
 4. Uso desde el navegador
 1. Cargar una imagen desde el botón 'Seleccionar archivo.
 
 2. Ver la previsualización.
 3. Presionar Analizar Imagen'.    
 4. Observar la imagen procesada y el estado del carril.
 5. Flujo interno- Recepción de imagen- Detección de vehículos- Segmentación de carriles- Superposición- Análisis de invasión- Respuesta JSON + imagen en Base64
6. Finalización
 Cerrar con CTRL + C en la consola
