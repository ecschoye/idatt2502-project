import gc
import time
import gym
import numpy as np
import torch
from network import DiscreteActorCriticNN
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam


class PPO:
    def __init__(self, env):
        # Make sure the environment is compatible with our code
        assert type(env.observation_space) is gym.spaces.Box
        assert type(env.action_space) is gym.spaces.Discrete

        self._init_hyperparameters()
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, running on CPU.")

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = DiscreteActorCriticNN(self.obs_dim, self.act_dim).to(self.device)
        self.critic = DiscreteActorCriticNN(self.obs_dim, 1).to(self.device)

        # Define optimizer for our actor parameters
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
        }

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 10000  # timesteps per batch
        self.max_timesteps_per_episode = 1000  # timesteps per episode

        self.gamma = 0.99
        self.n_updates_per_iteration = 10  # number of updates per iteration
        self.clip = 0.2  # Recommended
        self.lr = 0.001  # Learning rate

        self.save_freq = 4  # How often we save in number of iterations
        self.render = True
        self.render_every_i = 1

    def learn(self, total_timesteps):
        print("Starting learning process")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations so far
        # ALG STEP #2
        while t_so_far < total_timesteps:
            # ALG STEP #3
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rtgs,
                batch_lens,
            ) = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Introducing dynamic learining rate that decreases as the training advances
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.000001)
                self.actor_optim.param_groups[0]["lr"] = new_lr 
                self.critic_optim.param_groups[0]["lr"] = new_lr   

                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients 
                # and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger["actor_losses"].append(actor_loss.detach())

            self._log_summary()
            del batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

            if i_so_far % self.save_freq == 0:
                print("Saving")
                torch.save(self.actor.state_dict(), "./src/PPO/ppo_actor.pth")
                torch.save(self.critic.state_dict(), "./src/PPO/ppo_critic.pth")

    def rollout(self):
        gc.collect()
        # Batch data
        batch_obs = (
            []
        )  # batch operations: (number of timesteps per batch, dimension of observation)
        batch_acts = (
            []
        )  # batch actions: (number of timesteps per batch, dimension of action)
        batch_log_probs = (
            []
        )  # log probabilities of each action: (number of timesteps per batch)
        batch_rews = (
            []
        )  # batch rewards: (number of episodes, number of timesteps per episode)
        batch_rtgs = []  # batch rewards-to-go: (number of timesteps per batch)
        batch_lens = []  # episodic lengths in batch: (number of episodes)
        # Number of timesteps run so far this batch
        t = 0

        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []

            # Generic gym rollout on one episode
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if (
                    self.render
                    and (self.logger["i_so_far"] % self.render_every_i == 0)
                    and len(batch_lens) == 0
                ):
                    self.env.render()

                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action and log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # Plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.long).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32).to(
            self.device
        )

        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews).to(self.device).view(-1)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        # Return batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return
        # The shape will ne (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each ep backwards to maintain same order
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same as calling self.actor.forward(obs)
        action_probs = self.actor(obs)

        # Multivariate Normal Distribution
        action_dist = Categorical(action_probs)

        # Sample an action from the distribution and get its log probability
        action = action_dist.sample()

        return action.item(), action_dist.log_prob(action)

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs[0]).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network
        # This segment of code is similar to that in get_action()
        action_prob = self.actor(batch_obs[0])
        action_dist = Categorical(action_prob)
        log_probs = action_dist.log_prob(batch_acts)

        return V, log_probs

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.

        Parameters:
          None

        Return:
          None
        """
        # Calculate logging values. 
        # I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO;
        # feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]]
        )
        print(self.logger["actor_losses"])
        avg_actor_loss = np.mean(
            [
                losses.float().mean().cpu().item()
                for losses in self.logger["actor_losses"]
            ]
        )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print("------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
