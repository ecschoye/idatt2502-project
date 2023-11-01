import torch 
import time
import gym
import numpy as np
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from network import FeedForwardNN

class PPO: 
  def __init__(self, env):
    # Make sure the environment is compatible with our code
    assert(type(env.observation_space) == gym.spaces.Box)
    assert(type(env.action_space) == gym.spaces.Box)

    self._init_hyperparameters()
    # Extract environment information
    self.env = env
    self.obs_dim = env.observation_space.shape[0]
    self.act_dim = env.action_space.shape[0]

    # ALG STEP 1
    # Initialize actor and critic networks 
    self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
    self.critic = FeedForwardNN(self.obs_dim, 1)

    # Create our variable for the matrix 
    # Chose 0.5 for stdev arbitrarily 
    self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)

    # Create the covariance matrix 
    self.cov_mat = torch.diag(self.cov_var)

    # Define optimizer for our actor parameters
    self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
    self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    # This logger will help us with printing out summaries of each iteration
    self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

  def _init_hyperparameters(self):
    #Default values for hyperparameters, will need to change later. 
    self.timesteps_per_batch = 4800        # timesteps per batch
    self.max_timesteps_per_episode = 1600  # timesteps per episode

    self.gamma = 0.95
    self.n_updates_per_iteration = 5       # number of updates per iteration
    self.clip = 0.2                        # Recommended
    self.lr = 0.005                        # Learning rate

    self.save_freq = 10                    # How often we save in number of iterations

  def learn(self, total_timesteps):  
    print("Starting learning process")
    t_so_far = 0 # Timesteps simulated so far
    i_so_far = 0 # Iterations so far 
    #ALG STEP #2
    while t_so_far < total_timesteps: 
      # ALG STEP #3
      batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

      # Calculate how many timesteps we collected this batch
      t_so_far += np.sum(batch_lens)
      # Increment the number of iterations 
      i_so_far += 1

      # Logging timesteps so far and iterations so far
      self.logger['t_so_far'] = t_so_far
      self.logger['i_so_far'] = i_so_far

      # Calculate V_{phi, k}
      V, _ = self.evaluate(batch_obs, batch_acts)
      A_k = batch_rtgs - V.detach()

      # Normalize advantages 
      A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

      for _ in range(self.n_updates_per_iteration):
        # Calculate pi_theta(a_t | s_t)
        V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

        # Calculate ratios 
        ratios = torch.exp(curr_log_probs - batch_log_probs)

        # Calculate surrogate losses
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, batch_rtgs)

        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        
        # Calculate gradients and perform backward propagation for critic network
        self.critic_optim.zero_grad()    
        critic_loss.backward()    
        self.critic_optim.step()

        # Log actor loss
        self.logger['actor_losses'].append(actor_loss.detach())

      self._log_summary()

      if i_so_far % self.save_freq == 0:
        print("Saving")
        torch.save(self.actor.state_dict(), './ppo_actor.pth')
        torch.save(self.critic.state_dict(), './ppo_critic.pth')

  def rollout(self):
    # Batch data
    batch_obs = []       # batch operations: (number of timesteps per batch, dimension of observation)
    batch_acts = []      # batch actions: (number of timesteps per batch, dimension of action)
    batch_log_probs = [] # log probabilities of each action: (number of timesteps per batch)
    batch_rews = []      # batch rewards: (number of episodes, number of timesteps per episode)
    batch_rtgs = []      # batch rewards-to-go: (number of timesteps per batch)
    batch_lens = []      # episodic lengths in batch: (number of episodes)

    # Number of timesteps run so far this batch
    t = 0

    while t < self.timesteps_per_batch: 
      # Rewards this episode
      ep_rews = []
    
      #Generic gym rollout on one episode 
      obs = self.env.reset()
      done = False 

      for ep_t in range(self.max_timesteps_per_episode):
        #Increment timesteps ran this batch so far
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
      batch_lens.append(ep_t + 1) #Plus 1 because timestep starts at 0
      batch_rews.append(ep_rews)

    # Reshape data as tensors in the shape specified before returning
    batch_obs = torch.tensor(batch_obs, dtype=torch.float)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

    # ALG STEP #4
    batch_rtgs = self.compute_rtgs(batch_rews)

    # Log the episodic returns and episodic lengths in this batch.
    self.logger['batch_rews'] = batch_rews
    self.logger['batch_lens'] = batch_lens

    #Return batch data
    return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
  
  def compute_rtgs(self, batch_rews): 
    # The rewards-to-go (rtg) per episode per batch to return
    # The shape will ne (num timesteps per episode)
    batch_rtgs = []

    #Iterate through each ep backwards to maintain same order 
    for ep_rews in reversed(batch_rews): 
      discounted_reward = 0 #The discounted reward so far

      for rew in reversed(ep_rews): 
        discounted_reward = rew + discounted_reward * self.gamma
        batch_rtgs.insert(0, discounted_reward)

    #Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs

  def get_action(self, obs): 
    # Query the actor network for a mean action. 
    # Same as calling self.actor.forward(obs)
    mean = self.actor(obs)
  
    # Multivariate Normal Distribution
    dist = MultivariateNormal(mean, self.cov_mat)

    # Sample an action from the distribution and get its log probability
    action = dist.sample()
    log_prob = dist.log_prob(action)

    # Return the sampled action and the log prob of that action
    # Note that I'm calling detach() since the action and log_prob  
    # are tensors with computation graphs, so I want to get rid
    # of the graph and just convert the action to numpy array.
    # log prob as tensor is fine. Our computation graph will
    # start later down the line.
    return action.detach().numpy(), log_prob.detach()
  
  def evaluate(self, batch_obs, batch_acts): 
    # Query critic network for a value V for each obs in batch_obs
    V = self.critic(batch_obs).squeeze()

    # Calculate the log probabilities of batch actions using most
    # recent actor network 
    # This segment of code is similar to that in get_action()
    mean = self.actor(batch_obs)
    dist = MultivariateNormal(mean, self.cov_mat)
    log_probs = dist.log_prob(batch_acts)

    return V, log_probs
  
  def _log_summary(self):
    """
      Print to stdout what we've logged so far in the most recent batch.

      Parameters:
        None

      Return:
        None
    """
    # Calculate logging values. I use a few python shortcuts to calculate each value
    # without explaining since it's not too important to PPO; feel free to look it over,
    # and if you have any questions you can email me (look at bottom of README)
    delta_t = self.logger['delta_t']
    self.logger['delta_t'] = time.time_ns()
    delta_t = (self.logger['delta_t'] - delta_t) / 1e9
    delta_t = str(round(delta_t, 2))

    t_so_far = self.logger['t_so_far']
    i_so_far = self.logger['i_so_far']
    avg_ep_lens = np.mean(self.logger['batch_lens'])
    avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
    avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

    # Round decimal places for more aesthetic logging messages
    avg_ep_lens = str(round(avg_ep_lens, 2))
    avg_ep_rews = str(round(avg_ep_rews, 2))
    avg_actor_loss = str(round(avg_actor_loss, 5))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
    print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
    print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
    print(f"Average Loss: {avg_actor_loss}", flush=True)
    print(f"Timesteps So Far: {t_so_far}", flush=True)
    print(f"Iteration took: {delta_t} secs", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)

    # Reset batch-specific logging data
    self.logger['batch_lens'] = []
    self.logger['batch_rews'] = []
    self.logger['actor_losses'] = []