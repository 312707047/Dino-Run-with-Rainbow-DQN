import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Input
from tensorflow.keras import backend as K

class Actor_model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        self.model = self._create_model
        self.action_space = action_space
        self.input_shape = input_shape
        self.lr = lr
        self.optimizer = optimizer
    
    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        state = actions * y_pred
        old_state = actions * prediction_picks
        
        state = K.clip(state, 1e-10, 1.0)
        old_state = K.clip(old_state, 1e-10, 1.0)
        
        ratio = K.exp(K.log(state) - K.log(old_state))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        
        actor_loss = -K.mean(K.minimum(p1, p2))
        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        total_loss = actor_loss - entropy
        
        return total_loss
    
    def _create_model(self):
        input = Input(shape=self.input_shape)
        x = layers.Flatten()(input)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(self.action_space, activation='softmax')(x)
        model = Model(input, output)
        model.compile(loss=self.ppo_loss, optimizer=self.optimizer(learning_rate=self.lr))
        return model
    
    def predict(self, state):
        return self.model.predict(state)

class Critic_model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        self.model = self._create_model
        self.action_space = action_space
        self.input_shape = input_shape
        self.lr = lr
        self.optimizer = optimizer
    
    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true-y_pred)**2)
        return value_loss
    
    def _create_model(self):
        input = Input(shape=self.input_shape)
        x = layers.Flatten()(input)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1, activation=None)(x)
        model = Model(input, output)
        model.compile(loss=self.critic_PPO2_loss, optimizer=self.optimizer(learning_rate=self.lr))
    
    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))])