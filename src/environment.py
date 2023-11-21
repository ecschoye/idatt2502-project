import collections

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class FrameSkipWrapper(gym.Wrapper):
    """
    Wrapper for environment which repeats action
    at a given amount of frames (default = 4).
    """

    def __init__(self, env, skip=4):
        super(FrameSkipWrapper, self).__init__(env)
        self.buffer = collections.deque(maxlen=2)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.skip):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.buffer.append(observation)
            if done:
                break
        obs = np.max(np.stack(self.buffer), axis=0)
        self.frame = obs.copy()
        return obs, total_reward, done, info


class DownsampleAndGreyscale(gym.ObservationWrapper):
    """
    Downsamples and greyscales and image to 84x84.
    """

    def __init__(self, env):
        super(DownsampleAndGreyscale, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, observation):
        return DownsampleAndGreyscale.process(observation)

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


class FrameToTensor(gym.ObservationWrapper):
    """Converts frame to pytorch tensors"""

    def __init__(self, env):
        super(FrameToTensor, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    """
    Sliding window of observations
    """

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype,
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class PixelNormalize(gym.ObservationWrapper):
    """
    Normalize pixel values in frame from rgb to 0 or 1.
    """

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def create_mario_env(map="SuperMarioBros-v0", skip=4):
    """Creates a Super Mario Bros environment with wrappers"""
    env = FrameSkipWrapper(gym_super_mario_bros.make(map), skip)
    env = DownsampleAndGreyscale(env)
    env = FrameToTensor(env)
    env = BufferWrapper(env, 4)
    env = PixelNormalize(env)
    return JoypadSpace(env, SIMPLE_MOVEMENT)
