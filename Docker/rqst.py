import warnings
warnings.filterwarnings("ignore")
import os
# Configura el nivel de log a 3 (solo errores, sin advertencias o mensajes informativos)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sklearn
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import Recall
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Esto solo muestra errores y oculta los warnings
import pickle
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1, l2


class CustomRecallMetric(tf.keras.metrics.Metric):
    def __init__(self, name="custom_recall_metric", threshold=0.5, **kwargs):
        """
        Métrica personalizada para calcular un recall ponderado entre dos clases, con un umbral ajustable.

        Parameters:
        - name: Nombre de la métrica.
        - threshold: Umbral para convertir las predicciones en valores binarios.
        """
        super(CustomRecallMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        # Variables para las métricas de clase 0 y 1
        self.true_positives_0 = self.add_weight(name="tp0", initializer="zeros")
        self.false_negatives_0 = self.add_weight(name="fn0", initializer="zeros")
        self.true_positives_1 = self.add_weight(name="tp1", initializer="zeros")
        self.false_negatives_1 = self.add_weight(name="fn1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Actualiza el estado interno de la métrica con los valores reales y predichos.

        Parameters:
        - y_true: Tensor de etiquetas reales.
        - y_pred: Tensor de predicciones del modelo.
        - sample_weight: Pesos opcionales para las muestras.
        """
        # Convertir predicciones a valores binarios según el umbral
        y_pred = tf.cast(y_pred >= self.threshold, tf.int32)
        y_true = tf.cast(y_true, tf.int32)

        # Actualizar métricas para la clase 0
        self.true_positives_0.assign_add(tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 0), tf.float32)))
        self.false_negatives_0.assign_add(tf.reduce_sum(tf.cast((y_true == 0) & (y_pred != 0), tf.float32)))

        # Actualizar métricas para la clase 1
        self.true_positives_1.assign_add(tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32)))
        self.false_negatives_1.assign_add(tf.reduce_sum(tf.cast((y_true == 1) & (y_pred != 1), tf.float32)))

    def result(self):
        """
        Calcula el resultado de la métrica basada en el estado interno.

        Returns:
        - recall_conjunto: Recall ponderado entre las clases 0 y 1.
        """
        # Calcular el recall para cada clase
        recall_0 = self.true_positives_0 / (self.true_positives_0 + self.false_negatives_0 + tf.keras.backend.epsilon())
        recall_1 = self.true_positives_1 / (self.true_positives_1 + self.false_negatives_1 + tf.keras.backend.epsilon())

        # Combinar con los pesos proporcionados
        recall_conjunto = 0.3 * recall_0 + 0.7 * recall_1
        return recall_conjunto

    def reset_states(self):
        """
        Reinicia las variables internas al inicio de cada época.
        """
        self.true_positives_0.assign(0)
        self.false_negatives_0.assign(0)
        self.true_positives_1.assign(0)
        self.false_negatives_1.assign(0)

class RainNetwork:
    def __init__(self, eps: int = 50, batch_size: int = 16, learning_rate: float = 0.01, capas : int = 4, neuronas: int = 64, dropout: float = 0.0, activation : str = 'relu', type :str = 'binary_clas', l2: float = 0.0, umbral: int = 0.5):
        self.eps = eps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.capas = capas
        self.neuronas = neuronas
        self.dropout = dropout
        self.activation = activation
        self.type = type
        self.model = Sequential()
        self.l2 = l2
        self.umbral = umbral
    
    def create_model(self, X_train, num_clases):
    
        for i in range(self.capas):
            self.model.add(Dense(self.neuronas, input_shape= (X_train.shape[1],), activation = self.activation, kernel_regularizer = l2(self.l2)))
            if self.dropout != 0.0:
                self.model.add(Dropout(self.dropout))
                
        if self.type == 'binary_class':
            self.model.add(Dense(1, activation='sigmoid'))
        else:
            self.model.add(Dense(num_clases, activation='softmax'))

        self.model.compile(optimizer = Adam(learning_rate= self.learning_rate),
                      loss = 'binary_crossentropy',
                      metrics = [CustomRecallMetric(threshold=self.umbral)])
    
    def train(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(X_train,y_train, validation_data= (X_val, y_val), epochs = self.eps, batch_size = self.batch_size )
        return history.history['loss'], history.history['val_loss']

    def evaluate(self, X_test, y_test):
        loss, rec_2_clases = self.model.evaluate(X_test,y_test)
        print(f"Test recall 2 clases: {rec_2_clases:.4f}")

    def predict(self, X_new):
        prediction = self.model.predict(X_new)
        return prediction  
   