import time
import numpy as np
import torch
from tqdm import tqdm
from src.agent.ddqn_agent import DDQNAgent
from src.environment import create_mario_env
from src.utils.experience_replay_buffer import ExperienceReplayBuffer
from src.neptune_wrapper import NeptuneModels, NeptuneRun


def train_mario(pretrained=False, log=False):
    print("Creating environment")
    env = create_mario_env("SuperMarioBros-1-1-v0")
    state_space = env.observation_space.shape
    action_space = env.action_space.n

    agent = DDQNAgent(env, state_space, action_space)

    if pretrained:
        agent.load()

    num_episodes = 5000
    print("Training for {} episodes".format(num_episodes))
    total_rewards = []
    max_episode_reward = 0
    interval_reward = 0
    logged_flags = 0
    done = False
    # init logger with proper params
    if log:
        logger = NeptuneRun(
            params={
                "gamma": agent.gamma,
                "epsilon": agent.epsilon,
                "epsilon_min": agent.epsilon_min,
                "epsilon_decay": agent.epsilon_decay,
                "learning_rate": agent.lr,
                "batch_size": agent.batch_size,
                "memory_size": agent.memory_size,
                "copy": agent.copy,
                "action_space": agent.action_space,
                "state_space": agent.state_space,
                "num_episodes": num_episodes,
            },
            description="DDQN training run",
            tags=["DDQN"],
        )

    env.reset()
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.tensor(np.array([state]))
        # print("State shape in train_mario:", state.shape)
        total_reward = 0
        steps = 0
        frames = []
        while True:
            if episode % 10 == 0:
                env.render()
            # print("State shape before agent.act:", state.shape)
            action = agent.act(state)
            steps += 1

            next_state, reward, done, info = env.step(action)
            # reward = get_reward(done, steps, reward, env)
            total_reward += reward
            interval_reward += reward
            next_state = torch.tensor(np.array([next_state]))
            # print("Next state shape:", next_state.shape)
            reward = torch.tensor([reward], dtype=torch.float).unsqueeze(0)
            # print("Reward shape:", reward.shape)
            done = torch.tensor([done], dtype=torch.bool).unsqueeze(0)
            # print("Done shape:", done.shape)
            action = torch.tensor([action], dtype=torch.long).unsqueeze(0)
            # print("Action shape:", action.shape)

            agent.store_experience(state, action, reward, next_state, done)
            agent.experience_replay()

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
            if log and num_episodes >= 100:
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


def get_reward(done, step, reward, env):
    if not done or step == 10000:
        return reward
    return -100


def render_mario():
    env = create_mario_env("SuperMarioBros-6-4-v0")
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

                action = agent.best_action(state).item()

                next_state, reward, done, info = env.step(action)
                total_reward += reward

                state = torch.tensor(np.array([next_state]), dtype=torch.float32)

            print("Total reward achieved:", total_reward)
    except KeyboardInterrupt:
        print("Rendering stopped by the user.")
        env.close()




def log_model_version():
    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DDQNAgent(env, state_space, action_space)
    agent.load()
    num_episodes = 1000
    logger = NeptuneModels()
    logger.model_version(
        "MARIO-DDQN",
        {
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay,
            "learning_rate": agent.lr,
            "batch_size": agent.batch_size,
            "memory_size": agent.memory_size,
            "copy": agent.copy,
            "action_space": agent.action_space,
            "state_space": agent.state_space,
            "num_episodes": num_episodes,
        },
        ["../trained_model", "../replay_buffer_data"],
    )


if __name__ == "__main__":
    print("Starting training")
    train_mario(log=True)
    # train_mario(log="true")
    # render_mario()
    # log_model_version()
