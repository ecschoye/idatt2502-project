import time

import numpy as np
import torch
from tqdm import tqdm

from src.agent.ddqn_agent import DDQNAgent
from src.environment import create_mario_env
from src.utils.replay_buffer import ReplayBuffer


def train_mario():
    print("Creating environment")
    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n

    agent = DDQNAgent(env, state_space, action_space)

    num_episodes = 1000
    print("Training for {} episodes".format(num_episodes))
    total_rewards = []
    max_episode_reward = 0
    flags = 0
    done = False
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.tensor(np.array([state]))
        # print("State shape in train_mario:", state.shape)
        total_reward = 0
        steps = 0
        while True:
            # print("State shape before agent.act:", state.shape)
            action = agent.act(state)
            steps += 1

            next_state, reward, done, info = env.step(action)
            reward = get_reward(done, steps, reward, evn)
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

        total_rewards.append(reward)
        if episode % 10 == 0:
            tqdm.write("Episode: {}, Reward: {}, Max Reward: {}, Epsilon: {}, Steps: {}, Flags: {}".format(episode,
                                                                                                           total_reward,
                                                                                                           max_episode_reward,
                                                                                                           agent.epsilon,
                                                                                                           steps,
                                                                                                           flags))
            agent.save()

    agent.save()
    env.close()


def get_reward(done, step, reward, evn):
    if not done or step == env._max_episode_steps-1:
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

        if done:
            env.reset()


    env.close()


if __name__ == '__main__':
    print("Starting training")
    #train_mario()
    render_mario()
