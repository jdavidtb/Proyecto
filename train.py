import tensorflow as tf
from Models.object_detection import YOLOModel
from Utils.data_utils import cargar_dataset, crear_generador_datos
from simulation.config import RUTA_MODELO_DETECCION, RUTA_CLASES_DETECCION

def entrenar_modelo(epocas, tamaño_lote):
    # Cargar el dataset de entrenamiento
    train_data = cargar_dataset("data/train", tamaño_lote)
    val_data = cargar_dataset("data/val", tamaño_lote)

    # Obtener el número de clases del dataset
    num_clases = len(train_data.class_indices)

    # Definir los hiperparámetros del modelo
    input_shape = (416, 416, 3)
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    learning_rate = 1e-4
    weight_decay = 5e-4

    # Crear el modelo YOLO
    modelo = YOLOModel(input_shape, num_clases, anchors, learning_rate, weight_decay)

    # Compilar el modelo
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                   loss=modelo.yolo_loss)

    # Crear los generadores de datos
    train_generator = crear_generador_datos("data/train", tamaño_lote)
    val_generator = crear_generador_datos("data/val", tamaño_lote)

    # Entrenar el modelo
    modelo.fit(train_generator,
               epochs=epocas,
               validation_data=val_generator)

    # Guardar el modelo entrenado
    modelo.save_weights(RUTA_MODELO_DETECCION)

    # Guardar las clases del modelo
    clases = list(train_data.class_indices.keys())
    with open(RUTA_CLASES_DETECCION, "w") as archivo:
        archivo.write("\n".join(clases))

if __name__ == "__main__":
    epocas = 100
    tamaño_lote = 8
    entrenar_modelo(epocas, tamaño_lote)
