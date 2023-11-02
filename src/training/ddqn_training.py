import math
import time
import numpy as np
import torch
from tqdm import tqdm

from src.agent.ddqn_agent import DDQNAgent
from src.environment import create_mario_env
from src.utils.replay_buffer import ReplayBuffer
from src.neptune_wrapper import NeptuneModels, NeptuneRun

def train_mario(log = False):
    print("Creating environment")
    env = create_mario_env("SuperMarioBros-1-1-v0")
    state_space = env.observation_space.shape
    action_space = env.action_space.n

    agent = DDQNAgent(env, state_space, action_space)


    num_episodes = 200
    print("Training for {} episodes".format(num_episodes))
    total_rewards = []
    max_episode_reward = 0
    flags = 0
    done = False
    prev_reward = 0
    #init logger with proper params
    if log:
        logger = NeptuneRun(params = {
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
            tags = ["DDQN"]
            )
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.tensor(np.array([state]))
        # print("State shape in train_mario:", state.shape)
        total_reward = 0
        steps = 0
        reward_per_step = 0
        while True:
            if episode % 10 == 0:
                env.render()
            # print("State shape before agent.act:", state.shape)
            action = agent.act(state)
            steps += 1

            next_state, reward, done, info = env.step(action)
            #reward = get_reward(done, steps, reward, env)
            total_reward += reward
            next_state = torch.tensor(np.array([next_state]))
            # print("Next state shape:", next_state.shape)
            reward = torch.tensor([reward], dtype=torch.float).unsqueeze(0)
            # print("Reward shape:", reward.shape)
            done = torch.tensor([done], dtype=torch.bool).unsqueeze(0)
            # print("Done shape:", done.shape)
            action = torch.tensor([action], dtype=torch.long).unsqueeze(0)
            # print("Action shape:", action.shape)

            agent.remember(state, action, reward, next_state, done)
            agent.experience_replay()

            state = next_state

            if done:
                if info['flag_get']:
                    flags += 1
                if total_reward > max_episode_reward:
                    max_episode_reward = total_reward
                break

            if agent.epsilon > 0.05:
                if total_reward > prev_reward and episode > int(0.1 * num_episodes):
                    agent.epsilon = math.pow(agent.epsilon_decay,
                                             episode - int(0.05 * num_episodes))
            else:
                agent.update_epsilon()

            prev_reward = total_reward

        total_rewards.append(reward)
        if episode % 10 == 0:
            tqdm.write("Episode: {}, Reward: {}, Max Reward: {}, Epsilon: {}, Steps: {}, Flags: {}".format(episode,
                                                                                                           total_reward,
                                                                                                           max_episode_reward,
                                                                                                           agent.epsilon,
                                                                                                           agent.steps,
                                                                                                           flags)
            )
            agent.save()
        if log:
            logger.log_epoch({
                "train/reward" : total_reward,
                "train/max_episode_reward" : max_episode_reward,
                "train/epsilon" : agent.epsilon,
                "train/steps" : steps,
                "train/reward_per_step" : total_reward / steps,
            })
    if log:
        logger.finish()
    agent.save()
    env.close()


def get_reward(done, step, reward, env):
    if not done or step == 10000:
        return reward
    return -100

def render_mario():
    your_state_space = (4, 84, 84)
    your_memory_size = 20000
    replay_buffer = ReplayBuffer(your_state_space, your_memory_size)
    replay_buffer.load()
    env = create_mario_env()
    env.reset()
    for state, action, reward, next_state, done_flag in zip(
            replay_buffer.states,
            replay_buffer.actions,
            replay_buffer.rewards,
            replay_buffer.next_states,
            replay_buffer.done_flags
    ):
        env.render()
        action = action.item()
        next_state, reward, done, info = env.step(action)
        time.sleep(0.05)

        if done:
            env.reset()


    env.close()

def log_model_version():
    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DDQNAgent(env, state_space, action_space)
    agent.load()
    num_episodes = 100
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
        ["../trained_model", "../replay_buffer_data"]
    )

if __name__ == '__main__':
    print("Starting training")
    train_mario(log = True)
    #render_mario()
    #log()
