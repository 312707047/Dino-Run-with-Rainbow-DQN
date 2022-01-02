import logging
import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino

from agents import DQN, DuelDQN, DoubleDQN

# Setup logging
formatter = logging.Formatter(r'"%(asctime)s",%(message)s')
logger = logging.getLogger("dino-rl")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("G:/Code/Python/GitHub/Final-RL-Project/Dino_run/log/Batch256.csv")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Initialise the game
env = gym.make('ChromeDino-v0')
# env = gym.make('ChromeDinoNoBrowser-v0')
env = make_dino(env, timer=True, frame_stack=True)
# Get the number of actions and the dimension of input
n_actions = env.action_space.n

### Basic DQN Agent ###
agent = DQN(n_actions)
agent.train(env, logger)

### Dueling DQN Agent ###
# agent = DuelDQN(n_actions)
# agent.train(env, logger)

### Double DQN Agent ###
# agent = DoubleDQN(n_actions)
# agent.train(env, logger) 