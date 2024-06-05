import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def cargar_dataset(directorio_datos, tamaño_lote):
    generador_datos = ImageDataGenerator(rescale=1./255)
    dataset = generador_datos.flow_from_directory(
        directorio_datos,
        target_size=(416, 416),
        batch_size=tamaño_lote,
        class_mode='categorical'
    )
    return dataset

def preprocesar_imagen(imagen):
    imagen = tf.image.resize(imagen, (416, 416))
    imagen = imagen / 255.0
    imagen = tf.expand_dims(imagen, axis=0)
    return imagen

def generar_anchors(anchors):
    anchors = tf.constant(anchors, dtype=tf.float32)
    return anchors

def cargar_pesos_modelo(modelo, ruta_pesos):
    modelo.load_weights(ruta_pesos)
    return modelo

def guardar_pesos_modelo(modelo, ruta_pesos):
    modelo.save_weights(ruta_pesos)

def crear_generador_datos(directorio_datos, tamaño_lote, tamaño_imagen, aumento_datos=True):
    if aumento_datos:
        generador_datos = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        generador_datos = ImageDataGenerator(rescale=1./255)
    
    dataset = generador_datos.flow_from_directory(
        directorio_datos,
        target_size=tamaño_imagen,
        batch_size=tamaño_lote,
        class_mode='categorical'
    )
    return dataset
