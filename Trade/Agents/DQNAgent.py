from collections import deque
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Input, Model

class DQNAgent:
    def __init__(self):
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        
    def _create_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = layers.Flatten()(input)
        x = layers.Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        output = layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        model = Model(input, output, name='MLP_model')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model