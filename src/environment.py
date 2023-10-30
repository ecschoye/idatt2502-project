from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.utils.play import play
import numpy as np

class MarioEnvironment:
    def __init__(self):
        self.env =  gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
    #Test run with random moves
    def test_run(self):
        done = True
        for step in range(5000):
            if done:
                state = self.env.reset()
            state, reward, done, info = self.env.step(self.env.action_space.sample())
            self.env.render()

        self.env.close()
    #Play with keyboard
    def play(self):
        print(self.env._action_meanings)
        print(self.env.action_space)
        play(gym_super_mario_bros.make("SuperMarioBros-v0"), self.env.get_keys_to_action())

mario = MarioEnvironment()

