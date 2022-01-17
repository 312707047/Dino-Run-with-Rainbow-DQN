import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino

from agents import DQN, DuelDQN, DoubleDQN, PERDQN, CERDQN

# Initialize the game
env = gym.make('ChromeDino-v0')
# env = gym.make('ChromeDinoNoBrowser-v0')
env = make_dino(env, timer=True, frame_stack=True)
# Get the number of actions and the dimension of input
n_actions = env.action_space.n

### Basic DQN Agent ###
agent = DQN(n_actions, name='DQN_Batchnorm', batch_norm=True)
agent.train(env)

### Dueling DQN Agent ###
# agent = DuelDQN(n_actions)
# agent.train(env, logger)

### Double DQN Agent ###
# agent = DoubleDQN(n_actions)
# agent.train(env, logger)

### PER DQN Agent ###
# agent = PERDQN(n_actions)
# agent.train(env, logger)

### CER DQN Agent ###
# agent = CERDQN(n_actions)
# agent.train(env, logger)