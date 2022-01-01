import gym
import numpy as np
from collections import deque
import itertools
import random
import time

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import RMSprop

# Hyperparameter
LR = 1e-4
MEMORY_SIZE = 5000
EPISODES = 2000
EPSILON = 1
EPS_DECAY = 0.999
MIN_EPS = 0.01
BATCH_SIZE = 32
GAMMA = 0.99
UPDATE_EVERY = 5
SHOW_EVERY = 10

class DQN:
    def __init__(self, env):
        self.env = env
        self.n_action = env.action_space.n
        self.epsilon = EPSILON
        
        # initialize model
        self.policy_model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.policy_model.get_weights())
        
        # replay_memory
        self._replay_memory = deque(maxlen=MEMORY_SIZE)
        
        self.step = 0
    
    def _update_memory(self, state, action, next_state, reward):
        # reward shaping
        if next_state[0] >= 0.4:
            reward += 1
        self._replay_memory.append((state, action, next_state, reward))
        
    def _create_model(self):
        input = Input(shape=(env.reset().shape))
        x = layers.Dense(32, activation='relu')(input)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(self.n_action, activation='linear')(x)
        model = Model(input, output)
        model.compile(optimizer=RMSprop(learning_rate=LR),
                      loss='mse')
        return model
    
    def _choose_action(self, state):
        if np.random.random() < max(self.epsilon * EPS_DECAY ** self.step, MIN_EPS):
            return np.argmax(self.policy_model.predict(tf.expand_dims(state, axis=0))[0])
        return self.env.action_space.sample()
    
    def _optimize(self):
        if len(self._replay_memory) < BATCH_SIZE:
            return
        
        replay_batch = random.sample(self._replay_memory, BATCH_SIZE)
        states = np.array([transition[0] for transition in replay_batch])
        next_states = np.array([transition[2] for transition in replay_batch])
        
        Q = self.policy_model.predict(states)
        next_Q = self.target_model.predict(next_states)
        
        for i, (state, action, next_state, reward) in enumerate(replay_batch):
            Q[i][action] = (1 - GAMMA) * Q[i][action] + GAMMA * (reward + np.amax(next_Q[i])*0.95)
        
        self.policy_model.fit(states, Q, verbose=0)
        
        
    
    def _save_model(self, filepath='./MountainCar-v0-DQN.h5'):
        print('model saved')
        self.policy_model.save(filepath=filepath)
        
        
    
    def train(self):
        
        score_list = []
        for episode in range(1, EPISODES+1):
            self.env.reset()
            state = self.env.reset()
            score = 0
            done = False
            if episode % SHOW_EVERY == 0:
                render = True
            else:
                render = False
            
            while not done:
                if render:
                    self.env.render()
                
                # time.sleep(0.01)
                action = self._choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self._update_memory(state, action, next_state, reward)
                self._optimize()
                score += reward
                if next_state[0] >= self.env.goal_position:
                    print(f'Made it on episode: {episode}')
                state = next_state
            
            if episode % UPDATE_EVERY == 0:
                self.target_model.set_weights(self.policy_model.get_weights())
            
            score_list.append(score)
            print('episode:', episode, 'score:', score)
            if np.mean(score_list[-10:]) > -160:
                self._save_model()
                break

if __name__ == '__main__':
    env = gym.make('MountainCar-v0') # initialize environment
    env.reset() # return initial state
    DQN(env).train()
    env.close()