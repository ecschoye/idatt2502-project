from gym.core import Env
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.utils.play import play
import gym
import cv2
import collections
import numpy as np


class FrameSkipWrapper(gym.Wrapper):
    """
        Wrapper for environment which repeats action a given amount of frames (default = 4).
    """
    def __init__(self, env, skip=4):
        super(FrameSkipWrapper,self).__init__(env)
        self.buffer = collections.deque(maxlen=2)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.skip):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.buffer.append(observation)
            if(done):
                break
        obs = np.max(np.stack(self.buffer), axis=0)
        return obs, total_reward, done, info

class DownsampleAndGreyscale(gym.ObservationWrapper):
    """
    Downsamples and greyscales and image to 84x84. 
    """
    def __init__(self, env):
        super(DownsampleAndGreyscale, self).__init__(env)
        self.observation_space =  gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
    
    def observation(self, observation):
        return FrameSkipWrapper.process(observation)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


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

