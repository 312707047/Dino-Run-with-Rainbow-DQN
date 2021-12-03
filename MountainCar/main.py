import gym
import numpy as np

env = gym.make('MountainCar-v0') # initialize environment
env.reset() # return initial state

LEARNING_RATE = 0.1
DISCOUNT = 0.95 # alpha
EPISODES = 8000
SHOW_EVERY = 2000

epsilon = 0.5 # highier, more exploration
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
# print(env.observation_space.high)
# # 0.6, 0.07
# print(env.observation_space.low)
# # -1.2, -0.07
# print(env.action_space.n)
# # 3

# Create Q-table with a managable size

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # 20*20
discrete_os_win_size = (env.observation_space.high-env.observation_space.low) / DISCRETE_OS_SIZE
# print(discrete_os_win_size) # 0.09, 0.007

# initialize Q-table with possible state and action and random value
q_table = np.random.uniform(low=-2,
                            high=0,
                            size=(DISCRETE_OS_SIZE + [env.action_space.n])) # reward is between -1 and 0
                            # total action of 20*20*3
# print(q_table.shape) # (20,20,3)   
# print(q_table)     

# convert continuous state to discrete state
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

discrete_state = get_discrete_state(env.reset())
print(discrete_state) # (7, 10)
print(q_table[discrete_state]) # Q_value: [-0.30754493 -0.30126694 -0.49713614]
print(np.argmax(q_table[discrete_state])) # 0, take action 0

for episode in range(EPISODES):
    
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action) # after environment do a step, we received new state
        new_discrete_state = get_discrete_state(new_state) # transform new continuous state to discrete state
        if render:
            env.render()
        if not done: # update Q table
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f'We made it on episode: {episode}')
            q_table[discrete_state+(action, )] = 0
        
        discrete_state = new_discrete_state
    
    # add epsilon for random action for exploration
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()