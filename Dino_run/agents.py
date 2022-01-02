from collections import deque
import random
import itertools
import tensorflow as tf
import numpy as np

from model import Net, DuelNet
from utils import LinearAnneal, huber_loss
from parameters import HyperParam

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
            return np.argmax(self.policy_model.predict(tf.expand_dims(state, axis=0)/255))
        return np.random.randint(0, self.n_actions)
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return

        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])/255
        next_states = np.array([transition[3] for transition in batch])/255
        
        Qs = self.policy_model.predict(states)
        Qs_next = self.target_model.predict(next_states)
        
        for i, (_, action, reward, _,) in enumerate(batch):
            Qs[i][action] = (reward + self.GAMMA * tf.reduce_max(Qs_next[i])) #(1 - self.RL_LR) * Qs[i][action] + self.RL_LR * 
        
        self.policy_model.fit(states, Qs, verbose=0)
    
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
                    self._update_replay_memory((state, action, reward, next_state))
                    self._optimize()
                    state = next_state
                    if done:
                        break
                    
            if episode % self.TARGET_UPDATE == 0:
                self.target_model.set_weights(self.policy_model.get_weights())
            
            optim_cnt += t
            score = env.unwrapped.game.get_score()
            
            if ((score > max(score_list)) and (score > 150)) or (episode == self.N_EPISODE+1):
                self._save(filepath=f'./models/DQN_ep:{episode}')
                print('saving model')
                
            score_list.append(score)
            
            logger.info(f"{episode},{optim_cnt},{total_reward:.1f},{score},{self.epsilon.p:.6f}")
            # print(f"episode:{episode} | total_reward: {total_reward:.1f} | score: {score} | epsilon:{self.epsilon.p:.6f}")
            
class DoubleDQN(DQN):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(n_actions, batch_norm)
    
    def _optimize(self):
        if len(self.replay_memory) < self.MIN_MEMORY:
            return
        
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])/255
        next_states = np.array([transition[3] for transition in batch])/255
        
        # Q value
        Qs = self.policy_model.predict(states)
        
        # Expect Q value
        # predict action with policy model
        evaluated_action = np.argmax(self.policy_model.predict(next_states), axis=1)
        # Output: action of new states
        
        # evaluate action with target model
        Qs_next_target_state = np.array(self.target_model.predict(next_states))
        # Output: Q value of new states
        
        # Calculate Q value
        for index, (state, action, reward, next_state) in enumerate(batch):
            Qs[index][action] = reward + self.GAMMA * Qs_next_target_state[index][evaluated_action[index]]
            
        self.policy_model.fit(states, Qs, verbose=0)

class DuelDQN(DQN):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(n_actions, batch_norm)
    
    def _init_model(self):
        self.policy_model = DuelNet(self.n_actions, self.LR, self.batch_norm)
        self.target_model = DuelNet(self.n_actions, self.LR, self.batch_norm)
        self.target_model.set_weights(self.policy_model.get_weights())
    