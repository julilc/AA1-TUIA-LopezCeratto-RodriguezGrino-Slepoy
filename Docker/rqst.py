import sklearn
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import Recall
import tensorflow as tf
import tensorflow as tf
import pickle
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


class CustomRecallMetric(tf.keras.metrics.Metric):
    def __init__(self, name='custom_recall_metric', **kwargs):
        super(CustomRecallMetric, self).__init__(name=name, **kwargs)
        self.true_positives_0 = self.add_weight(name="tp0", initializer="zeros")
        self.false_negatives_0 = self.add_weight(name="fn0", initializer="zeros")
        self.true_positives_1 = self.add_weight(name="tp1", initializer="zeros")
        self.false_negatives_1 = self.add_weight(name="fn1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices using a threshold
        y_pred = tf.cast(y_pred >= 0.5, tf.int32)  # Ajusta el umbral según sea necesario
        y_true = tf.cast(y_true, tf.int32)

        # Update metrics for class 0
        self.true_positives_0.assign_add(tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 0), tf.float32)))
        self.false_negatives_0.assign_add(tf.reduce_sum(tf.cast((y_true == 0) & (y_pred != 0), tf.float32)))

        # Update metrics for class 1
        self.true_positives_1.assign_add(tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32)))
        self.false_negatives_1.assign_add(tf.reduce_sum(tf.cast((y_true == 1) & (y_pred != 1), tf.float32)))

    def result(self):
        # Calculate recall for class 0
        recall_0 = self.true_positives_0 / (self.true_positives_0 + self.false_negatives_0 + tf.keras.backend.epsilon())
        # Calculate recall for class 1
        recall_1 = self.true_positives_1 / (self.true_positives_1 + self.false_negatives_1 + tf.keras.backend.epsilon())
        # Combine recalls with weighted average
        custom_metric = 0.3 * recall_0 + 0.7 * recall_1  # Ajusta los pesos según tu necesidad
        return custom_metric

    def reset_states(self):
        # Reset metric state variables
        self.true_positives_0.assign(0)
        self.false_negatives_0.assign(0)
        self.true_positives_1.assign(0)
        self.false_negatives_1.assign(0)


class RainNetwork:
    def __init__(self, eps: int = 50, batch_size: int = 16, learning_rate: float = 0.01, capas : int = 4, neuronas: int = 64, dropout: float = 0.0, activation : str = 'relu', type :str = 'binary_clas'):
        self.eps = eps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.capas = capas
        self.neuronas = neuronas
        self.dropout = dropout
        self.activation = activation
        self.type = type
        self.model = Sequential()
    
    def create_model(self, X_train, num_clases):
    
        for i in range(self.capas):
            self.model.add(Dense(self.neuronas, input_shape= (X_train.shape[1],), activation = self.activation))
            if Dropout!= 0.0:
                self.model.add(Dropout(self.dropout))

        if self.type == 'binary_class':
            self.model.add(Dense(1, activation='sigmoid'))
        else:
            self.model.add(Dense(num_clases, activation='softmax'))

        self.model.compile(optimizer = Adam(learning_rate= self.learning_rate),
                      loss = 'binary_crossentropy',
                      metrics = [CustomRecallMetric()])
    
    def train(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(X_train,y_train, validation_data= (X_val, y_val), epochs = self.eps, batch_size = self.batch_size )
        return history.history['loss'], history.history['val_loss']

    def evaluate(self, X_test, y_test):
        loss, rec_2_clases = self.model.evaluate(X_test,y_test)
        print(f"Test recall 2 clases: {rec_2_clases:.4f}")

    def predict(self, X_new):
        prediction = self.model.predict(X_new)
        return prediction  
