from collections import deque
import random
import itertools
import tensorflow as tf
import numpy as np

from model import Net, DuelNet
from utils import LinearAnneal, Memory, huber_loss
from parameters import HyperParam

np.random.seed(87)
tf.random.set_seed(87)

class DQN(HyperParam):
    def __init__(self, n_actions, batch_norm=False):
        self.n_actions = n_actions
        self.batch_norm = batch_norm
        
        self._init_model()
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)
        self.epsilon = LinearAnneal(self.EPS_INIT, self.EPS_END, self.EXPLORE_STEP)
    
    def _init_model(self):
        self.policy_model = Net(self.n_actions, self.LR, self.batch_norm)
        self.target_model = Net(self.n_actions, self.LR, self.batch_norm)
        self.target_model.set_weights(self.policy_model.get_weights())
    
    def _update_replay_memory(self, transitions):
        self.replay_memory.append(transitions)
    
    def _choose_action(self, state):
        if random.random() > self.epsilon.anneal():
            return np.argmax(self.policy_model.predict(np.expand_dims(state, axis=0)/255))
        return np.random.randint(0, self.n_actions)
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE * 3:
            return

        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])/255
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])/255
        dones = np.array([transition[4] for transition in batch])
        
        Qs = self.policy_model.predict(states)
        Qs_next = self.target_model.predict(next_states)
        
        Qs_next_max = tf.reduce_max(Qs_next, axis=1, keepdims=True).numpy()
        Qs_target = np.copy(Qs)
        
        for i in range(self.BATCH_SIZE):
            Qs_target[i, actions[i]] = rewards[i] + self.DISCOUNT * Qs_next_max[i]
        
        # for those who wants to tune learning rate:    
        # for i in range(self.BATCHSIZE):
            # Qs_target[i, actions[i]] = (1 - lr) * Qs_target[i, actions[i]] + lr * (reward[i] + self.DISCOUNT * Qs_next_max[i])
        
        self.policy_model.train_on_batch(states, Qs_target)
        
        
    
    def _save(self, filepath):
        tf.saved_model.save(self.policy_model, filepath)
    
    def train(self, env, logger):
        optim_cnt = 0
        score_list = []
        score_list.append(0)
        for episode in range(1, self.N_EPISODE+1):
            total_reward = 0
            env.reset()
            state = np.array(env.reset())
            for t in itertools.count():
                if env.timer.tick() % 1 == 0:
                    action = self._choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    self._update_replay_memory((state, action, reward, next_state, done))
                    self._optimize()
                    state = next_state
                    if done:
                        break
                    
            if episode % self.TARGET_UPDATE == 0:
                self.target_model.set_weights(self.policy_model.get_weights())
            
            optim_cnt += t
            score = env.unwrapped.game.get_score()
            
            if (score > max(score_list)) and (score > 150):
                self._save(filepath='PERDQN_higest')
                
            score_list.append(score)
            
            logger.info(f"{episode},{optim_cnt},{total_reward:.1f},{score},{self.epsilon.p:.6f}")
        self._save(filepath='PER_lastest')
            
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
        dones = np.array([transition[4] for transition in batch])
        
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

            
        self.policy_model.train_on_batch(states, Qs_target)

class DuelDQN(DQN):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(n_actions, batch_norm)
    
    def _init_model(self):
        self.policy_model = DuelNet(self.n_actions, self.LR, self.batch_norm)
        self.target_model = DuelNet(self.n_actions, self.LR, self.batch_norm)
        self.target_model.set_weights(self.policy_model.get_weights())

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
        dones = np.array([transition[4] for transition in batch])
        
        Qs = self.policy_model.predict(states)
        Qs_target = np.copy(Qs)
        
        Qs_next = self.target_model.predict(next_states)
        Qs_next_max = tf.reduce_max(Qs_next, axis=1, keepdims=True).numpy()
        
        for i in range(self.BATCH_SIZE):
            Qs_target[i, actions[i]] = rewards[i] + self.DISCOUNT * Qs_next_max[i]
        
        errors = huber_loss(Qs, Qs_target)
        self.replay_memory.batch_update(tree_idx, errors)
        
        self.policy_model.train_on_batch(states, Qs_target)

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
        dones = np.array([transition[4] for transition in batch])
        
        Qs = self.policy_model.predict(states)
        Qs_next = self.target_model.predict(next_states)
        
        Qs_next_max = tf.reduce_max(Qs_next, axis=1, keepdims=True).numpy()
        Qs_target = np.copy(Qs)
        
        for i in range(self.BATCH_SIZE):
            Qs_target[i, actions[i]] = rewards[i] + self.DISCOUNT * Qs_next_max[i]
        
        self.policy_model.train_on_batch(states, Qs_target)
            
    def train(self, env, logger):
        optim_cnt = 0
        score_list = []
        score_list.append(0)
        for episode in range(1, self.N_EPISODE+1):
            total_reward = 0
            env.reset()
            state = np.array(env.reset())
            for t in itertools.count():
                if env.timer.tick() % 1 == 0:
                    action = self._choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    self.new_transition = (state, action, reward, next_state, done)
                    self._update_replay_memory(self.new_transition)
                    self._optimize()
                    state = next_state
                    if done:
                        break
                    
            if episode % self.TARGET_UPDATE == 0:
                self.target_model.set_weights(self.policy_model.get_weights())
            
            optim_cnt += t
            score = env.unwrapped.game.get_score()
            
            if (score > max(score_list)) and (score > 150):
                self._save(filepath='CERDQN_higest')
                
            score_list.append(score)
            
            logger.info(f"{episode},{optim_cnt},{total_reward:.1f},{score},{self.epsilon.p:.6f}")
        self._save(filepath='CERDQN_lastest')