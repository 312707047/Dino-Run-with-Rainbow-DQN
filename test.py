import numpy as np
import gym
import gym_chrome_dino
import tensorflow as tf
import time
import random
import os
import cv2
from PIL import Image
from tensorflow.python.keras import callbacks
from tqdm import tqdm
from collections import deque

# Set parameters
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINI_BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MODEL_NAME = 'Dino_run'
MIN_REWARD = -200
MEMORY_FRACTION = 0.2

EPISODES = 20000

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50  # episodes

env = gym.make('ChromeDino-v0')

ep_rewards = [-200]

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# create model folder
if not os.path.isdir('./models'):
    os.makedirs('./models')

from tensorflow.keras.callbacks import TensorBoard

class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()
from tensorflow.keras import layers, Sequential

# Create agent
class DQNAgent:
    def __init__(self):
        # main model
        self.model = self.create_model()
        # target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        
        self.target_update_counter = 0
    
    def create_model(self):
        model = Sequential([
            layers.Conv2D(256, 3, activation='relu' ,input_shape=(env.render().shape)),
            layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(env.action_space.n, activation='linear')]) # use linear because we use np.argmax
        
        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mae'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    # train network every step during episode
    def train(self, terminal_state, step):
        # start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # get a mini batch of random samples from memory replay table
        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        
        # get current states from mini batch, then throw the data into NN to get Q value
        current_states = np.array([transition[0] for transition in mini_batch])/255
        current_qs_list = self.model.predict(current_states)
        
        # get future states from mini batch, then throw the data into NN to get Q value
        new_current_states = np.array([transition[3] for transition in mini_batch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        
        X, y = [], []
        
        # enumerate data from mini batch
        for idx, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):
            
            # If not terminal state, get new Q from future state, otherwise set it to 0
            if not done:
                max_future_q = np.max(future_qs_list[idx])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[idx]
            current_qs[action] = new_q
            
            X.append(current_state)
            y.append(current_qs)
        
        # Fit on all samplse as one batch, log only on terminal state
        self.model.fit(np.array(X)/255,
                       np.array(y),
                       batch_size=MINI_BATCH_SIZE,
                       verbose=0,
                       shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):
    
    agent.tensorboard = episode
    
    episode_reward = 0
    step = 1
    
    current_state = env.reset()
    
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, _ = env.step(action)
        
        episode_reward += reward
        
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        
        current_state = new_state
        step += 1
    
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]/len(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
          
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)          