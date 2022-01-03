import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import RMSprop
from utils import huber_loss

# class ConvBlock(Model):
#     def __init__(self, batch_norm=False):
#         super().__init__()
#         self.batch_norm = batch_norm
#         self.conv_1 = layers.Conv2D(32, 8, 4, activation='relu')
#         self.conv_2 = layers.Conv2D(64, 4, 2, activation='relu')
#         self.conv_3 = layers.Conv2D(64, 3, 1, activation='relu')
        
#     def call(self, x):
#         if self.batch_norm:
#             x = layers.BatchNormalization(self.conv_1(x))
#             x = layers.BatchNormalization(self.conv_2(x))
#             x = layers.BatchNormalization(self.conv_3(x))
#         else:
#             x = self.conv_1(x)
#             x = self.conv_2(x)
#             x = self.conv_3(x)
            
#         return layers.Flatten()(x)

# class Net(ConvBlock):
#     def __init__(self, n_actions, batch_norm=False):
#         super().__init__(batch_norm=batch_norm)
#         self.dense = layers.Dense(512, activation='relu')
#         self.head = layers.Dense(n_actions, activation='linear')
    
#     def call(self, x):
#         # x = self.dense(x)
#         x = self.head(x)
        
#         return x
        
# class DuelNet(ConvBlock):
#     def __init__(self, n_actions, batch_norm=False):
#         super().__init__(batch_norm=batch_norm)
#         self.a_1 = layers.Dense(512, activation='relu')
#         self.a_2 = layers.Dense(n_actions, activation='linear')
#         self.v_1 = layers.Dense(512, activation='relu')
#         self.v_2 = layers.Dense(1, activation='linear')
    
#     def call(self, x):
#         a = self.a_1(x)
#         a = self.a_2(a)
        
#         v = self.v_1(x)
#         v = self.v_2(v)
        
#         output = v + (a - tf.reduce_max(a, axis=1, keepdims=True))
#         return output
def ConvBlock(x, batch_norm=False):
    if batch_norm:
        x = layers.Conv2D(32, 8, 4, activation='relu', kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 4, 2, activation='relu', kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, 1, activation='relu', kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.Conv2D(32, 8, 4, activation='relu', kernel_initializer='he_uniform')(x)
        x = layers.Conv2D(64, 4, 2, activation='relu', kernel_initializer='he_uniform')(x)
        x = layers.Conv2D(64, 3, 1, activation='relu', kernel_initializer='he_uniform')(x)

    return layers.Flatten()(x)

def Net(n_actions, lr, batch_norm=False):
    input = Input(shape=(80, 160, 4))
    x = ConvBlock(input, batch_norm)
    x = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    output = layers.Dense(n_actions, activation='linear', kernel_initializer='he_uniform')(x)
    model = Model(input, output)
    model.compile(optimizer=RMSprop(learning_rate=lr),
                  loss=huber_loss)
    
    return model

def DuelNet(n_actions, lr, batch_norm=False):
    input = Input(shape=(80, 160, 4))
    x = ConvBlock(input, batch_norm)
    a = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    a = layers.Dense(n_actions, activation='linear', kernel_initializer='he_uniform')(a)
    v = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    v = layers.Dense(1, activation='linear', kernel_initializer='he_uniform')(v)
    output = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
    model = Model(input, output)
    model.compile(optimizer=RMSprop(learning_rate=lr), loss=huber_loss)
    return model