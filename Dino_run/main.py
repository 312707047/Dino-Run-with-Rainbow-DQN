import logging
import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino

from agents import DQNAgent

# Setup logging
formatter = logging.Formatter(r'"%(asctime)s",%(message)s')
logger = logging.getLogger("dino-rl")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("./dino-log.csv")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Initialise the game
env = gym.make('ChromeDino-v0')
# env = gym.make('ChromeDinoNoBrowser-v0')
env = make_dino(env, timer=True, frame_stack=True)

# Get the number of actions and the dimension of input
n_actions = env.action_space.n

dqn = DQNAgent(n_actions)
dqn.train(env, logger)