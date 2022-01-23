import random
import itertools
import tensorflow as tf
import numpy as np
import logging
import cv2
import os

from collections import deque
from tf_model import Net, DuelNet
from utils import LinearAnneal
from parameters import HyperParam

np.random.seed(87)
random.seed(87)
tf.random.set_seed(87)

class DQN(HyperParam):
    def __init__(self, n_actions, name, batch_norm=False):
        self.n_actions = n_actions
        self.batch_norm = batch_norm
        self.name = name
        
        self._init_model()
        # self.replay_memory = ReplayMemory(self.MEMORY_SIZE)
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)
        self.epsilon = LinearAnneal(self.EPS_INIT, self.EPS_END, self.EXPLORE_STEP)
        
        # initialize log
        self._init_logger()
    
    def _init_model(self):
        self.policy_model = Net(self.n_actions, self.LR, self.batch_norm)
        self.target_model = Net(self.n_actions, self.LR, self.batch_norm)
        self.target_model.set_weights(self.policy_model.get_weights())
    
    def _init_logger(self):
        formatter = logging.Formatter(r'"%(asctime)s",%(message)s')
        self.logger = logging.getLogger("dino-rl")
        self.logger.setLevel(logging.INFO)
        # fh = logging.FileHandler(f"G:/Code/Python/GitHub/Final-RL-Project/Dino_run/log/{self.name}.csv")
        fh = logging.FileHandler(f"G:/Code/Python/GitHub/Final-RL-Project/Dino_run/80-80-log/{self.name}.csv")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def _choose_action(self, state):
        if random.random() > self.epsilon.anneal():
            return np.argmax(self.policy_model.predict(np.expand_dims(state, axis=0)/255))
        return np.random.randint(0, self.n_actions)
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE * 3:
            return 0

        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])/255
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])/255
        # dones = np.array([transition[4] for transition in batch])
        
        Qs = self.policy_model.predict(states)
        Qs_next = self.target_model.predict(next_states)
        
        Qs_next_max = tf.reduce_max(Qs_next, axis=1, keepdims=True).numpy()
        Qs_target = np.copy(Qs)
        
        for i in range(self.BATCH_SIZE):
            Qs_target[i, actions[i]] = rewards[i] + self.DISCOUNT * Qs_next_max[i]
        
        # for those who wants to tune learning rate:    
        # for i in range(self.BATCHSIZE):
            # Qs_target[i, actions[i]] = (1 - lr) * Qs_target[i, actions[i]] + lr * (rewards[i] + self.DISCOUNT * Qs_next_max[i])
        
        loss = self.policy_model.train_on_batch(states, Qs_target)
        
        return loss
    
    def _save(self, filepath):
        tf.saved_model.save(self.policy_model, filepath)
    
    def train(self, env):
        optim_cnt = 0
        for episode in range(1, self.N_EPISODE+1):
            total_reward = 0
            epoch_loss = []
            env.reset()
            state = np.array(env.reset())
            for t in itertools.count():
                if env.timer.tick() % 1 == 0:
                    # show what the agent see
                    # cv2.imshow('Dino', env.frames[0])
                    # cv2.waitKey(1)
                    action = self._choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    self.new_transition = (state, action, reward, next_state)
                    
                    if done:
                        loss = self._optimize()
                        epoch_loss.append(loss)
                        break
                    else:
                        self.replay_memory.append(self.new_transition)
                        loss = self._optimize()
                        epoch_loss.append(loss)
                    
                    state = next_state
                    
            if episode % self.TARGET_UPDATE == 0:
                self.target_model.set_weights(self.policy_model.get_weights())
            
            optim_cnt += t
            avg_loss = np.mean(epoch_loss)
            score = env.unwrapped.game.get_score()
            
            self.logger.info(f"{episode},{optim_cnt},{total_reward:.1f},{avg_loss:.4f},{score},{self.epsilon.p:.6f}")
        self._save(filepath=f'G:/Code/Python/GitHub/Final-RL-Project/Dino_run/models/80-80-{self.name}')
            
class DoubleDQN(DQN):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(n_actions, batch_norm)
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE * 3:
            return
        
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])/255
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])/255
        
        # Q value
        Qs = self.policy_model.predict(states)
        
        # Expect Q value
        # predict action with policy model
        evaluated_action = np.argmax(self.policy_model.predict(next_states), axis=1)
        # Output: action of new states
        
        # evaluate action with target model
        Qs_next = np.array(self.target_model.predict(next_states))
        # Output: Q value of new states
        
        Qs_target = np.copy(Qs)
        
        # Calculate Q value
        for i in range(self.BATCH_SIZE):
            Qs_target[i, actions[i]] = rewards[i] + self.DISCOUNT * Qs_next[i, evaluated_action[i]]

        loss = self.policy_model.train_on_batch(states, Qs_target)
        
        return loss

class DuelDQN(DQN):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(n_actions, batch_norm)
    
    def _init_model(self):
        self.policy_model = DuelNet(self.n_actions, self.LR, self.batch_norm)
        self.target_model = DuelNet(self.n_actions, self.LR, self.batch_norm)
        self.target_model.set_weights(self.policy_model.get_weights())

'''
Not finished yet
class PERDQN(DQN):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(n_actions, batch_norm=batch_norm)
        self.replay_memory = Memory(self.MEMORY_SIZE)
    
    def _update_replay_memory(self, transitions):
        self.replay_memory.store(transitions)
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE * 3:
            return
        
        tree_idx, batch = self.replay_memory.sample(self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])/255
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])/255
        
        Qs = self.policy_model.predict(states)
        Qs_target = np.copy(Qs)
        
        Qs_next = self.target_model.predict(next_states)
        Qs_next_max = tf.reduce_max(Qs_next, axis=1, keepdims=True).numpy()
        
        for i in range(self.BATCH_SIZE):
            Qs_target[i, actions[i]] = rewards[i] + self.DISCOUNT * Qs_next_max[i]
        
        # self.replay_memory.batch_update(tree_idx, errors)
        
        loss = self.policy_model.train_on_batch(states, Qs_target)
        
        return loss
'''

class CERDQN(DQN):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(n_actions, batch_norm=batch_norm)
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE * 3:
            return

        batch = random.sample(self.replay_memory, self.BATCH_SIZE-1)
        batch.append(self.new_transition)
        states = np.array([transition[0] for transition in batch])/255
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])/255
        
        Qs = self.policy_model.predict(states)
        Qs_next = self.target_model.predict(next_states)
        
        Qs_next_max = tf.reduce_max(Qs_next, axis=1, keepdims=True).numpy()
        Qs_target = np.copy(Qs)
        
        for i in range(self.BATCH_SIZE):
            Qs_target[i, actions[i]] = rewards[i] + self.DISCOUNT * Qs_next_max[i]
        
        loss = self.policy_model.train_on_batch(states, Qs_target)
        
        return loss
            