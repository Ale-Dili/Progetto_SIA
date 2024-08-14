
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.logger import configure, JSONOutputFormat, Logger
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._log_freq = 10

    def _on_step(self) -> bool:
        #some code
        return True


now = datetime.now()
ts = now.strftime("%d-%m-%y_%H-%M")
N_STEP = 7e6


tmp_path = "log/"

#new_logger = configure(tmp_path, ["stdout", "tensorboard"])


env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=16)
#env = make_atari_env('BreakoutNoFrameskip-v4')
env.metadata['render_fps'] = 60

env = VecFrameStack(env, n_stack=4)



model = A2C("CnnPolicy", env, verbose=1,tensorboard_log='./'+tmp_path, )
#model.set_logger(new_logger)
model.learn(total_timesteps=int(N_STEP),callback=TensorboardCallback(), log_interval=300) #, callback=TensorboardCallback()

obs = env.reset()
#model = A2C.load("A2C_breakout_2e6") #uncomment to load saved model
model.save("A3C_breakout_"+str(N_STEP))


