import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda, Concatenate
from tensorflow.keras.regularizers import l2
from utils.data_utils import load_dataset, preprocess_image, generate_anchors
from utils.visualization_utils import draw_bounding_boxes

class YOLOModel:
    def __init__(self, input_shape, num_classes, anchors, learning_rate=1e-4, weight_decay=5e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 32, 3)
        x = self.conv_block(x, 64, 3)
        x = self.conv_block(x, 128, 3)
        x = self.conv_block(x, 256, 3)
        x = self.conv_block(x, 512, 3)

        output_layer = Conv2D(len(self.anchors) * (5 + self.num_classes), 1, kernel_regularizer=l2(self.weight_decay))(x)
        output_layer = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(self.anchors), 5 + self.num_classes)))(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self.yolo_loss)

        return model

    def conv_block(self, x, filters, kernel_size, strides=1):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def yolo_loss(self, y_true, y_pred):
        # Constantes
        lambda_coord = 5.0
        lambda_noobj = 0.5
        
        # Obtener las dimensiones relevantes
        batch_size = tf.shape(y_pred)[0]
        grid_size = tf.shape(y_pred)[1]
        num_anchors = len(self.anchors)
        
        # Reshape de las tensores y_true e y_pred
        y_true = tf.reshape(y_true, [batch_size, grid_size, grid_size, num_anchors, 5 + self.num_classes])
        y_pred = tf.reshape(y_pred, [batch_size, grid_size, grid_size, num_anchors, 5 + self.num_classes])
        
        # Separar las coordenadas, la confianza y las probabilidades de clase
        true_box_xy = y_true[..., 0:2]
        true_box_wh = y_true[..., 2:4]
        true_box_conf = y_true[..., 4]
        true_box_class = y_true[..., 5:]
        
        pred_box_xy = y_pred[..., 0:2]
        pred_box_wh = y_pred[..., 2:4]
        pred_box_conf = y_pred[..., 4]
        pred_box_class = y_pred[..., 5:]
        
        # Calcular la pérdida de las coordenadas
        coord_mask = tf.expand_dims(true_box_conf, axis=-1)
        coord_loss = lambda_coord * tf.reduce_sum(coord_mask * tf.square(true_box_xy - pred_box_xy)) + \
                    lambda_coord * tf.reduce_sum(coord_mask * tf.square(tf.sqrt(true_box_wh) - tf.sqrt(pred_box_wh)))
        
        # Calcular la pérdida de la confianza para las cajas con objeto
        obj_mask = true_box_conf
        obj_loss = tf.reduce_sum(obj_mask * tf.square(true_box_conf - pred_box_conf))
        
        # Calcular la pérdida de la confianza para las cajas sin objeto
        noobj_mask = 1 - true_box_conf
        noobj_loss = lambda_noobj * tf.reduce_sum(noobj_mask * tf.square(true_box_conf - pred_box_conf))
        
        # Calcular la pérdida de clasificación
        class_loss = tf.reduce_sum(true_box_conf * tf.square(true_box_class - pred_box_class))
        
        # Calcular la pérdida total
        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        
        return total_loss

    def train(self, train_data, val_data, epochs, batch_size):
        self.model.fit(train_data, validation_data=val_data, epochs=epochs, batch_size=batch_size)

    def predict(self, image):
        image = preprocess_image(image)
        predictions = self.model.predict(image)
        return predictions

# Función para entrenar el modelo
def train_model(input_shape, num_classes, anchors, learning_rate, weight_decay, train_data_dir, val_data_dir, epochs, batch_size):
    model = YOLOModel(input_shape, num_classes, anchors, learning_rate, weight_decay)
    train_data = load_dataset(train_data_dir, batch_size)
    val_data = load_dataset(val_data_dir, batch_size)
    model.train(train_data, val_data, epochs, batch_size)
    return model

# Función para realizar la detección de objetos en una imagen
def detect_objects(model, image_path):
    image = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
    predictions = model.predict(image)
    # Realiza el post-procesamiento de las predicciones y dibuja los cuadros delimitadores
    draw_bounding_boxes(image, predictions)
    return image