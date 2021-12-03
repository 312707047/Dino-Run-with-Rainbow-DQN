import numpy as np
from numpy.lib.function_base import average
import tensorflow.keras.backend as backend
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000 # Step that keeps for training
MIN_REPLAY_MEMORY_SIZE = 1000 # minimum number of steps in a memory to start
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5 # terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200 #for model save
MEMORY_FRACTION = 0.2

# Environment settings
EPISODES = 2000

# Exploration settings
epsilon = 1 # it will decay
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 50 # episodes
SHOW_PREVIEW = False

class Agent:
    def __init__(self, size):
        self.size = size
        # initialize Agent position
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
    
    def __str__(self):
        return f'Agent ({self.x}, {self.y})'
    
    # get the distance from other Agent
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    # Agent action
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)
    
    # Movements
    def move(self, x=False, y=False):
        # if no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        
        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        
        # fix out of bound problem
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

class Env:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3) # 4
    ACTION_SPACE_SIZE = 9
    
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}
    
    def reset(self):
        self.player = Agent(self.SIZE)
        self.food = Agent(self.SIZE)
        
        # deal with overlapping problem
        while self.food == self.player:
            self.food = Agent(self.SIZE)
        self.enemy = Agent(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Agent(self.SIZE)
        
        self.episode_step = 0
        
        # decide to use image or digit as observation
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation
    
    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        
        self.enemy.move()
        self.food.move()
        
        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)
        
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY
        
        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True
            
        return new_observation, reward, done
    
    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)
    
    # CNN
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N] # set food to green
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N] # set enemy to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N] # set player to blue
        img = Image.fromarray(env, 'RGB')
        return img

env = Env()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create model folder
if not os.path.isdir('models'):
    os.makedirs('models')

# Own Tensorboard class
class ModifiedTensorBoard(callbacks.TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
    
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

# Agent
class DQNAgent:
    def __init__(self) -> None:
        
        # Main model
        self.model = self.create_model()
        
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir=f'logs/{MODEL_NAME}-{int(time.time())}')
        
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    
    def create_model(self):
        model = Sequential([
            layers.Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(256, (3, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(env.ACTION_SPACE_SIZE, activation='linear')])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model
    
    # Add step's data to a memory replay array
    # (observation space action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    # Trains main network every step during episode
    def train(self, terminal_state, step):
        
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        y = []
        
        # Now we need to enumerate our batches
        for index, (current_states, action, reward, new_current_state, done) in enumerate(minibatch):
            
            # if not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            # And append to our training data
            X.append(current_states)
            y.append(current_qs)
        
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255,
                       np.array(y),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
        
        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    
    # Update tensorboard step every episode
    agent.tensorboard.step = episode
    
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1
    
    # Reset environment and get initial state
    current_state = env.reset()
    
    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        
        new_state, reward, done = env.step(action)
        
        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward
        
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()
            
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1
        
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode ==1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        
        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{int(time.time())}.model')
    
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)