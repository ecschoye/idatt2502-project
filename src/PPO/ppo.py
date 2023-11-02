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
    def __init__(self, env, hyperparameters=None):
        # Environment compatability
        assert type(env.observation_space) is gym.spaces.Box
        assert type(env.action_space) is gym.spaces.Discrete

        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device.type}")

        # Initialize actor and critic networks
        self.actor = DiscreteActorCriticNN(self.obs_dim, self.act_dim).to(self.device)
        self.critic = DiscreteActorCriticNN(self.obs_dim, 1).to(self.device)

        # Define optimizer for our actor parameters
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Logger
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
            "lr" : [], # learning rates 
        }

    def _init_hyperparameters(self, hyperparameters):
        """ 
            Start by setting default values for all parameters
        """
        # Run
        self.timesteps_per_batch = 2000   # timesteps per batch
        self.max_timesteps_per_episode = 400  # timesteps per episode
        
        # Algorithm
        self.gamma = 0.95                 # Discount factor    
        self.n_updates_per_iteration = 5  # number of updates per iteration
        self.clip = 0.2                   # Recommended
        self.lr = 0.005                   # Learning rate
        self.num_minibatches = 5          # K in the paper
        self.lam = 0.98                   # Lambda for GAE-Lambda
        self.ent_coef = 0.01              # Entropy coefficient, higher penalizes overdeterministic policies 
        self.max_grad_norm = 0.5          # Gradient clipping threshold
        self.target_kl = 0.02             # Target KL-divergence

        # Misc
        self.save_freq = 1  # How often we save in number of iterations
        self.render = True  # If we should render during rollout    
        self.render_every_i = 1 # Only render every i iterations

        """ 
            Update passed hyperparameters 
        """
        if hyperparameters is not None:
            for param, val in hyperparameters.items():
                exec ("self." + param + " = " + str(val))

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks.

            Parameters:   
                total_timesteps - The total number of timesteps to train for
    
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
                batch_dones
            ) = self.rollout()

            t_so_far += np.sum(batch_lens) # Calculate how many timesteps we collected this batch
            i_so_far += 1 # Increment the number of iterations
            self.logger["t_so_far"] = t_so_far # Logging timesteps so far and iterations so far
            self.logger["i_so_far"] = i_so_far

            # Calculate V_{phi, k}
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs[0]).squeeze()
            batch_rtgs = A_k + V.detach()

            # Normalize advantages to decrease the variance of our advantages and 
            # convergence more stable and faster
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update actor and critic networks using mini-batches
            step = batch_obs.size(0) # Number of samples in batch
            inds = np.arange(step)   # Indices to shuffle array
            minibatch_size = step // self.num_minibatches 
            loss = []

            for _ in range(self.n_updates_per_iteration):
                # Introducing dynamic learining rate that decreases as the training advances
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Learning rate never below zero
                new_lr = max(new_lr, 0.000001)
                self.actor_optim.param_groups[0]["lr"] = new_lr 
                self.critic_optim.param_groups[0]["lr"] = new_lr  
                self.logger['lr'] = new_lr 

                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end] # Indices for batch samples

                    # Gather data from minibatch
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_probs = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts) # Calculate pi_theta(a_t | s_t)

                    logratios = curr_log_probs - mini_log_probs
                    ratios = torch.exp(logratios) # Calculate ratios
                    approx_kl = ((ratios - 1) ** logratios).mean() # Calculate approximated KL-divergence

                    surr1 = ratios * mini_advantage # Calculate surrogate losses
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients 
                    # and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                
                # Approximating KL divergence
                if approx_kl > self.target_kl: 
                    break

            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            self.logger["actor_losses"].append(avg_loss)

            self._log_summary()
            del batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

            if i_so_far % self.save_freq == 0:
                print("Saving")
                torch.save(self.actor.state_dict(), "./src/PPO/ppo_actor.pth")
                torch.save(self.critic.state_dict(), "./src/PPO/ppo_critic.pth")

    def rollout(self):
        gc.collect()
        """
        Batch data and sizes: 
            
            batch operations: (number of timesteps per batch, dimension of observation)
            batch actions: (number of timesteps per batch, dimension of action)
            log probabilities of each action: (number of timesteps per batch)
            batch rewards: (number of episodes, number of timesteps per episode)
            batch rewards-to-go: (number of timesteps per batch)
            episodic lengths in batch: (number of episodes)
            batch state values: (number of timesteps per batch)
            batch dones: (number of timesteps per batch)
            
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []  
        batch_vals = []   
        batch_dones= []  
        ep_rews = []
        ep_vals = []
        ep_dones = []

        # Number of timesteps run so far this batch
        t = 0

        while t < self.timesteps_per_batch:
            # Rewards, vals, dones this episode
            ep_rews = []
            ep_vals = []
            ep_dones = []

            # Generic gym rollout on one episode
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if ( # Render conditions 
                    self.render
                    and (self.logger["i_so_far"] % self.render_every_i == 0)
                    and len(batch_lens) == 0
                ):self.env.render()

                ep_dones.append(done)
                
                t += 1 # Increment timesteps ran this batch so far
                batch_obs.append(obs) # Collect observation

                # Calculate action and make step
                action, log_prob = self.get_action(obs)
                val = self.critic(obs)

                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action and log probability
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

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.long).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32).to(
            self.device
        ) # TODO maybe use flatten()

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        # Return batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

    def calculate_gae(self, rewards, values, dones): 
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))): 
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1-ep_dones[t]) - ep_vals[t]
                else: 
                    delta = ep_rews[t] - ep_vals[t]
                advantage = delta + self.gamma * self.lam * (1-ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)
            
            batch_advantages.extend(advantages)
        
        return torch.tensor(batch_advantages, dtype=torch.float32).to(self.device)

    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same as calling self.actor.forward(obs)
        obs = torch.tensor(obs,dtype=torch.float).to(self.device)
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

        return V, log_probs, action_dist.entropy()

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        lr = str(round(lr, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
