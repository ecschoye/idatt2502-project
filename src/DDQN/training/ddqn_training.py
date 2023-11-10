import time

import numpy as np
import torch
from tqdm import tqdm

from DDQN.agent.ddqn_agent import DDQNAgent
from environment import create_mario_env
from neptune_wrapper import NeptuneModels, NeptuneRun
from DDQN.utils.config import DDQNTrainingParameters

NUM_EPISODES = DDQNTrainingParameters.NUM_EPISODES.value
ENV_NAME = DDQNTrainingParameters.ENV_NAME.value


class DDQNTrainer:
    """
    Class for training a DDQN agent
    """
    def __init__(self):
        self.num_episodes = NUM_EPISODES
        self.env = create_mario_env(ENV_NAME)
        self.agent = DDQNAgent(self.env, self.env.observation_space.shape, self.env.action_space.n)
        self.flag_count = 0

    def train(self, log=False):
        """
        Train the agent for a number of episodes
        :param log: Whether to log to Neptune.ai
        """
        print("Training for {} episodes".format(self.num_episodes))
        total_rewards = []
        max_episode_reward = 0
        interval_reward = 0
        lowest_frame_count = float('inf')
        logger = None
        frames = []

        # init logger with proper params
        if log:
            logger = self.setup_neptune_logger(self.agent, NUM_EPISODES)

        self.env.reset()
        for episode in tqdm(range(self.num_episodes)):
            state = self.env.reset()
            state = torch.tensor(np.array([state]))
            total_reward = 0
            steps = 0
            while True:
                action = self.agent.act(state)
                steps += 1

                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                interval_reward += reward
                next_state = torch.tensor(np.array([next_state]))
                reward = torch.tensor([reward], dtype=torch.float).unsqueeze(0)
                done = torch.tensor([done], dtype=torch.bool).unsqueeze(0)
                action = torch.tensor([action], dtype=torch.long).unsqueeze(0)

                self.agent.add_experience_to_memory(state, action, reward, next_state, done)
                self.agent.learn_from_memory_batch()

                state = next_state
                if log:
                    frames.append(self.env.frame)
                if done:
                    if info["flag_get"]:
                        self.flag_count += 1

                        if len(frames) < lowest_frame_count:
                            lowest_frame_count = len(frames)
                            print(f"New lowest frame count: {lowest_frame_count}")

                            if episode >= self.num_episodes * 0.65:
                                logger.log_frames(frames, episode)

                    if total_reward > max_episode_reward:
                        max_episode_reward = total_reward
                    break

            total_rewards.append(reward)
            if episode % 10 == 0:
                flag_percentage = (self.flag_count / (episode + 1)) * 100
                tqdm.write(
                    "Episode: {}, "
                    "Reward: {}, "
                    "Max Reward: {}, "
                    "Epsilon: {}, "
                    "Steps: {}, "
                    "Flags: {}, "
                    "Flag Percentage:{},".format(
                        episode,
                        total_reward,
                        max_episode_reward,
                        self.agent.epsilon,
                        steps,
                        self.flag_count,
                        flag_percentage,
                    )
                )
                if log and NUM_EPISODES >= NUM_EPISODES * 0.02:
                    logger.log_epoch(
                        {"train/avg_reward_per_10_episodes": interval_reward / 10}
                    )
                interval_reward = 0
                self.agent.save()
            if log:
                logger.log_epoch(
                    {
                        "train/reward": total_reward,
                        "train/max_episode_reward": max_episode_reward,
                        "train/epsilon": self.agent.epsilon,
                        "train/steps": steps,
                        "train/reward_per_step": total_reward / steps,
                        "train/flags": self.flag_count,
                    }
                )
            self.agent.update_epsilon()
        if log:
            logger.finish()
        self.agent.save()
        self.env.close()

    def setup_neptune_logger(self, agent, num_episodes):
        """Sets up Neptune logger with proper parameters."""
        logger = NeptuneRun(
            params={
                "gamma": agent.gamma,
                "epsilon": agent.epsilon,
                "epsilon_min": agent.epsilon_min,
                "epsilon_decay": agent.epsilon_decay_rate,
                "learning_rate": agent.lr,
                "batch_size": agent.batch_size,
                "memory_size": agent.memory_size,
                "copy": agent.target_update_frequency,
                "num_episodes": num_episodes,
            },
            description="DDQN training run",
            tags=["DDQN"],
        )
        return logger


class DDQNRenderer:
    """
    Class for rendering a trained DDQN agent
    """
    def __init__(self):
        self.env = create_mario_env(ENV_NAME)
        self.agent = DDQNAgent(self.env, self.env.observation_space.shape, self.env.action_space.n)

    def render(self):
        """
        Render the agent playing the game
        """
        self.agent.load()
        state = self.env.reset()
        state = torch.tensor(np.array([state]), dtype=torch.float32)

        total_reward = 0
        done = False

        while not done:
            self.env.render()
            time.sleep(0.05)

            action = self.agent.select_greedy_action(state).item()

            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            state = torch.tensor(np.array([next_state]), dtype=torch.float32)

        print("Total reward achieved:", total_reward)


class DDQNLogger:
    """
    Logs DDQN model to Neptune.ai
    """
    def __init__(self):
        self.env = create_mario_env(ENV_NAME)
        self.agent = DDQNAgent(self.env, self.env.observation_space.shape, self.env.action_space.n)

    def log(self):
        """
        Logs DDQN model to Neptune.ai
        """
        self.agent.load()
        logger = NeptuneModels()
        logger.model_version(
            "MARIO-DDQN",
            {
                "gamma": self.agent.gamma,
                "epsilon": self.agent.epsilon,
                "epsilon_min": self.agent.epsilon_min,
                "epsilon_decay": self.agent.epsilon_decay_rate,
                "learning_rate": self.agent.lr,
                "batch_size": self.agent.batch_size,
                "memory_size": self.agent.memory_size,
                "copy": self.agent.target_update_frequency,
                "action_space": self.agent.action_space,
                "state_space": self.agent.state_space,
                "num_episodes": NUM_EPISODES,
            },
            ["../trained_model", "../experience_replay_buffer_data"],
        )
