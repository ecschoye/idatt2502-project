import gym 
from ppo import PPO

env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)