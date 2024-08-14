import random
import gymnasium as gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import Adam
import tensorflow as tf
import pickle

#per attivare metal  source ~/venv-metal/bin/activate

IS_TRAINED = True

ENV_NAME = "CartPole-v1"

GAMMA = 0.99
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 25

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.999


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
      
        for state, action, reward, state_next, terminal in batch:
            
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next, verbose=0)[0]))
            q_values = self.model.predict(state, verbose=0)  #salvare il q* 
            q_values[0][action] = q_update
            
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while run<10000:
        run += 1
        state,_ = env.reset()
        state = np.array(state).reshape(1, -1)
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal,_ ,_ = env.step(action)
            
            reward = reward if not terminal else -reward
            state_next = np.array(state_next).reshape(1, -1)
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
               
                break
            dqn_solver.experience_replay()
    with open('dqn_solver', 'wb') as f:
        pickle.dump(dqn_solver, f)
        
def cartpole_trained(is_render = False):
    dqn_solver = None
    with open('dqn_solver_v1', 'rb') as f:
        dqn_solver = pickle.load(f)
    
    if is_render:
        env = gym.make(ENV_NAME,render_mode = 'human')
    else:
        env = gym.make(ENV_NAME)
   
    state,_ = env.reset()
    
    tot_reward = 0
    
    mean_tot_reward= []
    
    terminal = False
    for _  in range(100):
        while not terminal:
            state = np.array(state).reshape(1, -1)
            action = dqn_solver.act(state)
            state, reward, terminal,_ ,_ = env.step(action)
            tot_reward += reward
        
        print(tot_reward)
        terminal = False 
        mean_tot_reward.append(tot_reward)
        state,_ = env.reset()
        tot_reward=0
    
    print(f'Mean last 100 reward: {np.mean(mean_tot_reward)}')
            


if __name__ == "__main__":
    if (IS_TRAINED):    
        cartpole_trained(is_render=True)
    else:
        cartpole()
    