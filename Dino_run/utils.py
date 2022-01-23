import numpy as np
import random
import tensorflow as tf
import torch

from collections import namedtuple
from SumTree import SumSegmentTree, MinSegmentTree

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

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.math.abs(error) < clip_delta

    squared_loss = 0.5 * tf.math.square(error)
    linear_loss  = clip_delta * (tf.math.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


class StateProcessor:
    """Convert state image to tensor"""
    def to_array(self, state):
        state = np.array(state).transpose((2, 0, 1))
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        return state

    def to_tensor(self, state):
        state = self.to_array(state)
        state = torch.from_numpy(state)
        return state.unsqueeze(0)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, capacity, alpha):
        super().__init__(capacity)
        self._alpha = alpha
        
        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
    
    def push(self, *args):
        idx = self.position
        super().push(*args)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
    
    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    
    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        transitions = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
            transitions.append(self.memory[idx])

        return transitions, weights, idxes

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)