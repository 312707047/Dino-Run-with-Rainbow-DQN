import random
from collections import namedtuple
import tensorflow as tf

class LinearAnneal:
    """Decay a parameter linearly"""
    def __init__(self, start_val, end_val, steps):
        self.p = start_val
        self.end_val = end_val
        self.decay_rate = (start_val - end_val) / steps

    def anneal(self):
        if self.p > self.end_val:
            self.p -= self.decay_rate
        return self.p

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """Experience replay pool"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.math.abs(error) < clip_delta

    squared_loss = 0.5 * tf.math.square(error)
    linear_loss  = clip_delta * (tf.math.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)