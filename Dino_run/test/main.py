import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino

import tensorflow as tf
import time
import random
import os
from tqdm import tqdm
from collections import deque
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np

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

ep_rewards = [-200]

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# create model folder
if not os.path.isdir('./models'):
    os.makedirs('./models')
if not os.path.isdir('./history'):
    os.makedirs('./history')

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
        input = Input(shape=(80, 160, 4)) # env.current_frame.shape
        # x = layers.Resizing(80, 80)(input)
        x = layers.Conv2D(32, 8, strides=4, activation='relu', padding='same')(input)
        # x = layers.MaxPool2D()(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, 4, strides=2, activation='relu', padding='same')(x)
        # x = layers.MaxPool2D()(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.Dense(32, activation='relu')(x)
        # x = layers.Dropout(0.2)(x)
        output = layers.Dense(env.action_space.n, activation='linear')(x) # use linear because we use np.argmax
        
        model = Model(input, output)
        
        model.compile(loss='mse',
                      optimizer=RMSprop(learning_rate=LR),
                      metrics=['mse'])
        return model
    
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

env = gym.make('ChromeDino-v0')
# env = gym.make('ChromeDinoNoBrowser-v0')
env = make_dino(env, timer=True, frame_stack=True)
# env = SubprocVecEnv([env_lambda for _ in range(4)])
agent = DQNAgent()


# for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):
for episode in range(1, EPISODES+1):
    
    # agent.tensorboard.step = episode
    
    episode_reward = 0
    step = 1
    
    current_state = env.reset()
    
    done = False
    while not done:
        # if env.timer.tick() % FRAME_PER_ACTION == 0:
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
    # print(f'total random walk: {randoms} | total not random: {not_random} | epsilon" {epsilon}')
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    # if not os.path.isfile(f'./history/lr:{LR}batch:{MINI_BATCH_SIZE}eps:{EPSILON_DECAY}'):
    
    print(f'episode:{episode}| episode_reward:{episode_reward:.2f} | step: {step} | score: {env.unwrapped.game.get_score()} | epsilon: {epsilon:.4f}')

env.close()