import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import RMSprop
from utils import huber_loss

def ConvBlock(x, batch_norm=False):
    if batch_norm:
        x = layers.Conv2D(32, 8, 4, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 4, 2, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, 1, activation='relu')(x)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.Conv2D(32, 8, 4, activation='relu')(x)
        x = layers.Conv2D(64, 4, 2, activation='relu')(x)
        x = layers.Conv2D(64, 3, 1, activation='relu')(x)

    return layers.Flatten()(x)

def Net(n_actions, lr, batch_norm=False):
    input = Input(shape=(80, 160, 4))
    x = ConvBlock(input, batch_norm)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(n_actions, activation='linear')(x)
    model = Model(input, output)
    model.compile(optimizer=RMSprop(learning_rate=lr),
                  loss=huber_loss)
    
    return model

def DuelNet(n_actions, lr, batch_norm=False):
    input = Input(shape=(80, 160, 4))
    x = ConvBlock(input, batch_norm)
    a = layers.Dense(512, activation='relu')(x)
    a = layers.Dense(n_actions, activation='linear')(a)
    v = layers.Dense(512, activation='relu')(x)
    v = layers.Dense(1, activation='linear')(v)
    output = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
    model = Model(input, output)
    model.compile(optimizer=RMSprop(learning_rate=lr), loss=huber_loss)
    return model