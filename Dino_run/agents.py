from collections import deque
import random
import itertools
import tensorflow as tf
import numpy as np

from model import Net
from utils import LinearAnneal
from parameters import HyperParam

class DQNAgent(HyperParam):
    def __init__(self, n_actions, batch_norm=False):
        self.n_actions = n_actions
        self.batch_norm = batch_norm
        
        self.policy_model = Net(n_actions, self.LR, batch_norm)
        self.target_model = Net(n_actions, self.LR, batch_norm)
        self.target_model.set_weights(self.policy_model.get_weights())
        
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)
        self.epsilon = LinearAnneal(self.EPS_INIT, self.EPS_END, self.EXPLORE_STEP)
        self.target_update_counter = 0
    
    def _update_replay_memory(self, transitions):
        self.replay_memory.append(transitions)
        
    def _get_qs(self, state):
        return self.policy_model.predict(np.array(np.expand_dims(state, axis=0))/255)[0]
    
    def _choose_action(self, state):
        if random.random() > self.epsilon.anneal():
            return np.argmax(self._get_qs(state))
        else:
            return np.random.randint(0, self.n_actions)
    
    def _optimize(self, terminal):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])/255
        qs_list = self.policy_model.predict(states)
        
        new_states = np.array([transition[3] for transition in batch])/255
        new_qs_list = self.target_model.predict(new_states)
        
        X, y = [], []
        
        for index, (state, action, reward, next_state, done)in enumerate(batch):
            if not done:
                max_next_q = np.max(new_qs_list[index])
                new_q = reward + self.GAMMA * max_next_q
            else:
                new_q = reward
        
            qs = qs_list[index]
            qs[action] = new_q
            
            X.append(state)
            y.append(qs)
        
        self.policy_model.fit(np.array(X)/255, np.array(y),
                              batch_size=self.BATCH_SIZE,
                              verbose=0,
                              shuffle=False)
        if terminal:
            self.target_update_counter += 1
        
        if self.target_update_counter > self.TARGET_UPDATE:
            self.target_model.set_weights(self.policy_model.get_weights())
            self.target_update_counter = 0
    
    def save(self, filename):
        tf.keras.models.save_model(self.policy_model, filename)
    
    def train(self, env, logger):
        optim_cnt = 0
        for episode in range(self.N_EPISODE):
            total_reward = 0
            state = env.reset()
            for t in itertools.count():
                action = self._choose_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                self._update_replay_memory((state, action, reward, next_state, done))
                self._optimize(done)
                state = next_state
            optim_cnt += t
            score = env.unwrapped.game.get_score()
            logger.info(f"{episode},{optim_cnt},{total_reward:.1f},{score},{self.epsilon.p:.6f}")
            
                