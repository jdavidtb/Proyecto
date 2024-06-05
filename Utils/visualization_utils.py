import cv2
import numpy as np
import tensorflow as tf

def dibujar_cuadros_delimitadores(imagen, predicciones, clases, umbral_confianza=0.5):
    imagen_copia = imagen.copy()
    altura_imagen, ancho_imagen = imagen.shape[:2]
    
    for prediccion in predicciones:
        clase_id = np.argmax(prediccion[5:])
        confianza = prediccion[4]
        
        if confianza >= umbral_confianza:
            x, y, w, h = prediccion[:4]
            
            # Ajustar las coordenadas al tamaño de la imagen original
            x = int(x * ancho_imagen)
            y = int(y * altura_imagen)
            w = int(w * ancho_imagen)
            h = int(h * altura_imagen)
            
            # Obtener la etiqueta de la clase
            etiqueta_clase = clases[clase_id]
            
            # Dibujar el cuadro delimitador y la etiqueta de clase
            color = (0, 255, 0)  # Verde
            cv2.rectangle(imagen_copia, (x, y), (x + w, y + h), color, 2)
            cv2.putText(imagen_copia, f"{etiqueta_clase}: {confianza:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return imagen_copia

def visualizar_detecciones(imagen, predicciones, clases):
    imagen_con_detecciones = dibujar_cuadros_delimitadores(imagen, predicciones, clases)
    
    # Mostrar la imagen con las detecciones
    cv2.imshow("Detecciones", imagen_con_detecciones)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def guardar_detecciones(imagen, predicciones, clases, ruta_salida):
    imagen_con_detecciones = dibujar_cuadros_delimitadores(imagen, predicciones, clases)
    
    # Guardar la imagen con las detecciones
    cv2.imwrite(ruta_salida, imagen_con_detecciones)
