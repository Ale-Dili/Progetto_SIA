from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import WarpFrame, EpisodicLifeEnv, AtariWrapper
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.logger import configure, JSONOutputFormat, Logger
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import AtariPreprocessing, FrameStack


def display_observation(observation):
    observation = observation[:, :, 0]
    plt.imshow(observation)
    plt.axis('off')
    plt.show()
'''
env = make_atari_env('BreakoutNoFrameskip-v4')
env.metadata['render_fps'] = 60
env = VecFrameStack(env, n_stack=4)

obs = env.reset()

model = A2C.load("A2C_breakout_7e6")
while True:
    action, _states = model.predict(obs)
    
    obs, rewards, dones, info = env.step(action)
    print(obs.shape)
    #env.render('human')    
    env.render('human')
    time.sleep(1/30)
'''
env = gym.make("BreakoutNoFrameskip-v4", render_mode = 'human')
env.metadata['render_fps'] = 60
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True)
env = FrameStack(env, num_stack=4)


model = A2C.load("A2C_breakout_7e6")

obs,info = env.reset()

n_lives = 5

while True:
    obs = np.array(obs)
    if (n_lives>info['lives']):
        action = 1
        n_lives=info['lives']
    else:
        action,_ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
   # print(info['lives'])

    if(terminated):
        obs, info = env.reset()
