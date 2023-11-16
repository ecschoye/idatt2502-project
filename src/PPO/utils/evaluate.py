import time

import torch
import torch.nn.functional as F


def evaluate(policy, env, render=False, render_delay=0.04):
    if torch.cuda.is_available():
        policy = policy.cuda()
        # Rollout with the policy and environment, and log each episode's data
    for ep_num, (ep_len, ep_ret) in enumerate(
        rollout(policy, env, render, render_delay)
    ):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)


def _log_summary(ep_len, ep_ret, ep_num):
    """
    Print to stdout what we've logged so far in the most recent episode.

    Parameters:
            None

    Return:
            None
    """
    # Round decimal places for more aesthetic logging messages
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print("------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy, env, render, delay):
    """
    Returns a generator to roll out each episode given a trained policy and
    environment to test on.
    """
    # Rollout until user kills process
    while True:
        obs = env.reset()
        done = False

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0  # episodic length
        ep_ret = 0  # episodic return

        while not done:
            t += 1

            # Render environment if specified, off by default
            if render:
                env.render()
                time.sleep(delay)

            # Query deterministic action from policy and run it
            obs = torch.tensor(obs, dtype=torch.float)
            action_probs = F.softmax(policy(obs.unsqueeze(0)), dim=-1)
            action = torch.multinomial(action_probs, num_samples=1)
            obs, rew, done, _ = env.step(action.item())

            # Sum all episodic rewards as we go along
            ep_ret += rew

        # Track episodic length
        ep_len = t

        # returns episodic length and return in this iteration
        yield ep_len, ep_ret
