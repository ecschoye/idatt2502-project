import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from neptune_wrapper import NeptuneRun
from PPO.network.network import DiscreteActorCriticNN
from PPO.utils.custom_deque import CustomDeque


class PPO:
    def __init__(self, env, hyperparameters=None):
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor and critic networks
        self.actor = DiscreteActorCriticNN(self.obs_dim, self.act_dim).to(self.device)
        self.critic = DiscreteActorCriticNN(self.obs_dim, 1).to(self.device)

        # Define optimizer for our actor parameters
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Logger
        if hyperparameters.get("log", False):
            self.neptune_logger = NeptuneRun(
                params={
                    "action_space": self.act_dim,
                    "clip": self.clip,
                    "ent_coef": self.ent_coef,
                    "gamma": self.gamma,
                    "lambda": self.lam,
                    "lr": self.lr,
                    "max_grad_norm": self.max_grad_norm,
                    "max_timesteps_per_episode": self.max_timesteps_per_episode,
                    "n_updates_per_iteration": self.n_updates_per_iteration,
                    "num_minibatches": self.num_minibatches,
                    "target_kl": self.target_kl,
                    "timesteps_per_batch": self.timesteps_per_batch,
                },
                description="PPO Train Run: " + hyperparameters["run_notes"],
                tags=["PPO"],
            )
        self.logger = {
            "delta_t": time.time_ns(),
            # Per Run
            "lr": [],  # learning rates
            "i_so_far": 0,
            "t_so_far": 0,  # timesteps so far
            "n_episodes": 0,  # Number of episodes
            "kl_divergence_breaks": 0,  # Number of times KL-divergence breaks
            "flags": 0,  # Flags caught
            "flags_last_n": CustomDeque(),  # Flags caught in last 100 episodes
            "best_reward": 0.0,  # Best episodic returns
            # Per Batch
            "batch_lens": [],  # Episodic lengths in batch
            "batch_rews": [],  # Episodic returns in batch
            "batch_best_frames": [],  # Best frames in batch
            "batch_actor_loss": [],  # Losses of actor network in batch
        }

    def _init_hyperparameters(self, hyperparameters):
        """
        Start by setting default values for all parameters
        """
        # Run
        self.timesteps_per_batch = 2000  # timesteps per batch
        self.max_timesteps_per_episode = 400  # timesteps per episode

        # Algorithm
        self.gamma = 0.90  # Discount factor
        self.n_updates_per_iteration = 5  # number of updates per iteration
        self.clip = 0.2  # Recommended
        self.lr = 0.005  # Learning rate
        self.num_minibatches = 5  # K in the paper
        self.lam = 0.98  # Lambda for GAE-Lambda
        self.ent_coef = (
            0.01  # Entropy coefficient, higher penalizes overdeterministic policies
        )
        self.max_grad_norm = 0.5  # Gradient clipping threshold
        self.target_kl = 0.02  # Target KL-divergence

        # Misc
        self.save_freq = 1  # How often we save in number of iterations
        self.render = True  # If we should render during rollout
        self.full_render = False  # Watch full training
        self.render_every_i = 1  # Only render every i iterations
        self.log = False  # If we push logs to Neptune
        self.capture_frames = False  # If we capture frames
        self.run_notes = ""  # Notes for the run

        """
        Update passed hyperparameters
        """
        if hyperparameters is not None:
            for param, val in hyperparameters.items():
                setattr(self, param, val)

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks.
        """
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations so far

        while t_so_far < total_timesteps:
            # Gather data from rollout of batch
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rews,
                batch_lens,
                batch_vals,
                batch_dones,
            ) = self.rollout()

            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far
            t_so_far += np.sum(
                batch_lens
            )  # Calculate how many timesteps we collected this batch
            i_so_far += 1  # Increment the number of iterations

            # Calculate V_{phi, k}
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()
            # Normalize advantages to decrease the variance of our advantages and
            # convergence more stable and faster
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update actor and critic networks using mini-batches
            step = batch_obs.size(0)  # Number of samples in batch
            inds = np.arange(step)  # Indices to shuffle array
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):
                # Introducing dynamic learining rate
                # that decreases as the training advances
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.000001)

                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                self.logger["lr"] = new_lr

                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]  # Indices for batch samples

                    # Gather data from minibatch
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_probs = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    V, curr_log_probs, entropy = self.evaluate(
                        mini_obs, mini_acts
                    )  # Calculate pi_theta(a_t | s_t)

                    logratios = curr_log_probs - mini_log_probs
                    ratios = torch.exp(logratios)  # Calculate ratios
                    approx_kl = (
                        (ratios - 1) ** logratios
                    ).mean()  # Calculate approximated KL-divergence

                    surr1 = ratios * mini_advantage  # Calculate surrogate losses
                    surr2 = (
                        torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                        * mini_advantage
                    )

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    # Calculate gradients and
                    # perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    self.actor_optim.step()

                    # Calculate gradients
                    # and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                # Approximating KL divergence
                if approx_kl > self.target_kl:
                    self.logger["kl_divergence_breaks"] += 1
                    break

            self.logger["batch_actor_loss"] = loss
            self.logger["n_episodes"] += len(batch_lens)

            if self.log:
                self.log_epoch()
            self._print_summary()
            self.reset_batch_log()

            del (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rews,
                batch_lens,
                batch_vals,
                batch_dones,
            )

            if i_so_far % self.save_freq == 0:
                print("Saving")
                torch.save(self.actor.state_dict(), "PPO/network/ppo_actor.pth")
                torch.save(self.critic.state_dict(), "PPO/network/ppo_critic.pth")

        self.neptune_logger.finish()

    def rollout(self):
        gc.collect()
        """
        Rollout the policy and collect data for a single batch.
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []
        batch_best_frames = []
        batch_best_rews = 0
        batch_flags = 0

        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_vals = []
            ep_dones = []
            ep_frames = []

            # Generic gym rollout on one episode
            obs = self.env.reset()
            done = False
            flag = False
            """
                Run an episode for a maximum of max_timesteps_per_episode timesteps
            """
            for ep_t in range(self.max_timesteps_per_episode):
                if self.full_render or (  # Render conditions
                    self.render
                    and (self.logger["i_so_far"] % self.render_every_i == 0)
                    and len(batch_lens) == 0
                ):
                    self.env.render()

                t += 1  # Increment timesteps ran this batch so far
                ep_dones.append(done)
                batch_obs.append(obs)  # Collect observation

                # Calculate action and make step

                action, log_prob = self.get_action(obs)
                obs1 = torch.tensor(obs, dtype=torch.float).to(self.device)
                val = self.critic(obs1.unsqueeze(0))
                obs, rew, done, info = self.env.step(action)

                if info["flag_get"]:
                    flag = True

                if self.capture_frames:
                    ep_frames.append(self.env.frame)
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # Plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

            if flag:
                batch_flags += 1
                self.logger["flags_last_n"].add_to_front(1)
            else:
                self.logger["flags_last_n"].add_to_front(0)

            sum_rewards = np.sum(ep_rews)
            if sum_rewards > batch_best_rews:
                batch_best_rews = sum_rewards
                if self.capture_frames:
                    batch_best_frames = ep_frames
            self.logger["batch_rews"].append(sum_rewards)

        # Reshape data as tensors in the shape specified before returning
        # batch_obs = torch.tensor(batch_obs, dtype=torch.float32).to(self.device)
        batch_obs_np = np.array(batch_obs)
        batch_obs = torch.tensor(batch_obs_np, dtype=torch.float32).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.long).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32).to(
            self.device
        )  # TODO maybe use flatten()

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_lens"] = batch_lens
        if batch_best_rews > self.logger["best_reward"]:
            self.logger["best_reward"] = batch_best_rews
        self.logger["batch_best_frames"] = batch_best_frames
        self.logger["flags"] += batch_flags

        # Return batch data
        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rews,
            batch_lens,
            batch_vals,
            batch_dones,
        )

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = (
                        ep_rews[t]
                        + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t])
                        - ep_vals[t]
                    )
                else:
                    delta = ep_rews[t] - ep_vals[t]
                advantage = (
                    delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                )
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float32).to(self.device)

    def get_action(self, obs):
        # Query the actor network for action probabilities.
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        action_prob = F.softmax(
            self.actor(obs.unsqueeze(0).to(self.device)), dim=-1
        ).to(self.device)[0]

        # Create a categorical distribution from action probabilities.
        action_dist = Categorical(action_prob)

        # Sample an action from the distribution and get its log probability.
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.detach().to(self.device)

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()
        # Process batch_obs for evaluation
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.device)
        action_prob = F.softmax(self.actor(batch_obs), dim=-1).to(self.device)
        action_dist = Categorical(action_prob)

        # Calculate log probabilities of batch actions
        log_probs = action_dist.log_prob(batch_acts).to(self.device)
        return V, log_probs, action_dist.entropy()

    def _print_summary(self):
        """
        Print a summary of our training so far.
        """
        iteration = self.logger["i_so_far"]
        trainings_run = self.logger["i_so_far"]
        timesteps_so_far = self.logger["t_so_far"]
        avg_batch_length = np.mean(self.logger["batch_lens"])
        avg_batch_reward = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]]
        )
        avg_actor_loss = np.mean(
            [
                losses.float().mean().cpu()
                if isinstance(losses, torch.Tensor)
                else losses
                for losses in self.logger["batch_actor_loss"]
            ]
        )
        best_batch_reward = self.logger["best_reward"]
        num_episodes = self.logger["n_episodes"]
        kl_divergence_breaks = self.logger["kl_divergence_breaks"]
        flags_caught = self.logger["flags"]
        flag_percentage = (
            self.logger["flags_last_n"].get_sum()
            / self.logger["flags_last_n"].get_length()
        )
        print(
            f"""
            -------------------- Iteration #{iteration} --------------------\n
            Number of trainings run: {trainings_run}\n
            Timesteps so far: {timesteps_so_far}\n
            Average length of batched episodes: {avg_batch_length}\n
            Average reward of batched episodes: {avg_batch_reward}\n
            Average actor loss: {avg_actor_loss}\n
            Best batch reward: {best_batch_reward}\n
            Number of episodes: {num_episodes}\n
            KL divergence breaks: {kl_divergence_breaks}\n
            Flags caught: {flags_caught}\n
            Flag percentage: {flag_percentage}\n
        """
        )

    def log_epoch(self):
        flag_percentage = (
            self.logger["flags_last_n"].get_sum()
            / self.logger["flags_last_n"].get_length()
        )

        self.logger["batch_actor_loss"] = [
            tensor.item() for tensor in self.logger["batch_actor_loss"]
        ]
        self.neptune_logger.log_lists(
            {
                "train/reward": self.logger["batch_rews"],
                "train/episode_lengths": self.logger["batch_lens"],
                "train/actor_loss": self.logger["batch_actor_loss"],
            }
        )
        self.neptune_logger.log_epoch(
            {
                "train/best_reward": self.logger["best_reward"],
                "train/lr": self.logger["lr"],
                "train/n_episodes": self.logger["n_episodes"],
                "train/kl_divergence_breaks": self.logger["kl_divergence_breaks"],
                "train/flag_total": self.logger["flags"],
                "train/flag_average": flag_percentage,
            }
        )
        self.neptune_logger.log_frames(
            self.logger["batch_best_frames"], self.logger["n_episodes"]
        )

    def reset_batch_log(self):
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["batch_best_frames"] = []
        self.logger["batch_actor_loss"] = []
