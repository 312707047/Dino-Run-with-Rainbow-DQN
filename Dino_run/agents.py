from collections import deque
import numpy as np
import random

from model import Network
from utils import ImageTensorProcessor

import torch as T
import torch.nn.functional as F
import torch.optim as optim

# Set parameters
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 30000
MIN_REPLAY_MEMORY_SIZE = 10000
MINI_BATCH_SIZE = 128
UPDATE_TARGET_EVERY = 5
MODEL_NAME = 'Dino_run'
MIN_REWARD = 200
FRAME_PER_ACTION = 1
LR = 2e-5

EPISODES = 20000

epsilon = 0.1
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.0001

AGGREGATE_STATS_EVERY = 100  # episodes

class DQNAgent:
    def __init__(self, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.target_model = self.target_model.load_state_dict(self.policy_model.state_dict())
        
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr = LR)
        
        
        self.target_update_counter = 0
    
    def _create_model(self, n_actions):
        self.policy_model = Network(n_actions).to(self.device)
        self.target_model = Network(n_actions).to(self.device)
        self.target_model = self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
    
    def _take_actions(self, state, epsilon):
        if random.random() > epsilon:
            with T.no_grad():
                return self.policy_model(state).max(1)[1].view(1, 1)
        else:
            action = random.randrange(self.n_actions)
            return T.tensor([[action]], device=self.device, dtype=T.long)
    
    def _q(self, states, actions):
        return self.policy_model(states).gather(1, actions)
    
    def _expected_q(self, next_states, rewards):
        non_final_mask = T.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self.device, dtype=T.bool)
        non_final_next_states = T.cat([s for s in next_states if s is not None])
        
        next_q = T.zeros(MINI_BATCH_SIZE, device=self.device)
        next_q[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_q = rewards + DISCOUNT * next_q
        return expected_q.unsqueeze(1)
    
    def _optimize(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        transitions = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        current_state = T.from_numpy(np.array([transition[0] for transition in transitions])/255)
        actions = T.from_numpy(np.array([transition[1] for transition in transitions]))
        rewards = T.from_numpy(np.array([transition[2] for transition in transitions]))
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    # train network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        if len(self.replay_memory) == MIN_REPLAY_MEMORY_SIZE:
            print('-----Start Training-----')

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        # print(current_qs_list) [[a, b, c, ...]] a, b, c shape=(9,)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            # print(current_qs)
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINI_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        # return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
        return self.model.predict(np.array(np.expand_dims(state, axis=0))/255)[0]