import time

import numpy as np
import torch
from tqdm import tqdm

from agent.ddqn_agent import DDQNAgent
from environment import create_mario_env
from neptune_wrapper import NeptuneModels, NeptuneRun
from utils.config import DDQNTrainingParameters

NUM_EPISODES = DDQNTrainingParameters.NUM_EPISODES
ENV_NAME = DDQNTrainingParameters.ENV_NAME


def train_mario(log=False):
    """Trains the agent with DDQN algorithm and logs the results to Neptune."""
    print("Creating environment")
    env = create_mario_env(ENV_NAME)
    state_space = env.observation_space.shape
    action_space = env.action_space.n

    # Initialize DDQN
    agent = DDQNAgent(env, state_space, action_space)

    print("Training for {} episodes".format(NUM_EPISODES))
    total_rewards = []
    max_episode_reward = 0
    interval_reward = 0
    logged_flags = 0

    # init logger with proper params
    if log:
        logger = setup_neptune_logger(agent, NUM_EPISODES)

    env.reset()
    for episode in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        state = torch.tensor(np.array([state]))
        total_reward = 0
        steps = 0
        frames = []
        while True:
            if episode % 10 == 0:
                env.render()
            action = agent.act(state)
            steps += 1

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            interval_reward += reward
            next_state = torch.tensor(np.array([next_state]))
            reward = torch.tensor([reward], dtype=torch.float).unsqueeze(0)
            done = torch.tensor([done], dtype=torch.bool).unsqueeze(0)
            action = torch.tensor([action], dtype=torch.long).unsqueeze(0)

            agent.add_experience_to_memory(state, action, reward, next_state, done)
            agent.learn_from_memory_batch()

            state = next_state
            if log:
                frames.append(env.frame)
            if done:
                if info["flag_get"]:
                    logged_flags += 1

                    if log and episode >= 3000:
                        logger.log_frames(frames, episode)

                if total_reward > max_episode_reward:
                    max_episode_reward = total_reward
                break

        total_rewards.append(reward)
        if episode % 10 == 0:
            tqdm.write(
                "Episode: {}, "
                "Reward: {}, "
                "Max Reward: {}, "
                "Epsilon: {}, "
                "Steps: {}, "
                "Flags: {}".format(
                    episode,
                    total_reward,
                    max_episode_reward,
                    agent.epsilon,
                    steps,
                    logged_flags,
                )
            )
            if log and NUM_EPISODES >= 100:
                logger.log_epoch(
                    {"train/avg_reward_per_10_episodes": interval_reward / 10}
                )
            interval_reward = 0
            agent.save()
        if log:
            logger.log_epoch(
                {
                    "train/reward": total_reward,
                    "train/max_episode_reward": max_episode_reward,
                    "train/epsilon": agent.epsilon,
                    "train/steps": steps,
                    "train/reward_per_step": total_reward / steps,
                    "train/flags": logged_flags,
                }
            )
        agent.update_epsilon()
    if log:
        logger.log_frames(frames, episode)
        logger.finish()
    agent.save()
    env.close()


def setup_neptune_logger(agent, num_episodes):
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
            "action_space": agent.action_space,
            "state_space": agent.state_space,
            "num_episodes": num_episodes,
        },
        description="DDQN training run",
        tags=["DDQN"],
    )
    return logger


def render_mario():
    """Render the ddqn agent playing the game of Mario."""
    env = create_mario_env(ENV_NAME)
    state_space = env.observation_space.shape
    action_space = env.action_space.n

    agent = DDQNAgent(env, state_space, action_space)

    agent.load()

    try:
        while True:
            state = env.reset()
            state = torch.tensor(np.array([state]), dtype=torch.float32)

            total_reward = 0
            done = False

            while not done:
                env.render()
                time.sleep(0.05)

                action = agent.select_greedy_action(state).item()

                next_state, reward, done, info = env.step(action)
                total_reward += reward

                state = torch.tensor(np.array([next_state]), dtype=torch.float32)

            print("Total reward achieved:", total_reward)
    except KeyboardInterrupt:
        print("Rendering stopped by the user.")
        env.close()


def log_model_version():
    """Log the model version to Neptune."""
    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DDQNAgent(env, state_space, action_space)
    agent.load()
    logger = NeptuneModels()
    logger.model_version(
        "MARIO-DDQN",
        {
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay_rate,
            "learning_rate": agent.lr,
            "batch_size": agent.batch_size,
            "memory_size": agent.memory_size,
            "copy": agent.target_update_frequency,
            "action_space": agent.action_space,
            "state_space": agent.state_space,
            "num_episodes": NUM_EPISODES,
        },
        ["../trained_model", "../experience_replay_buffer_data"],
    )
