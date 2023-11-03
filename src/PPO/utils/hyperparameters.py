# Hyperparameters for PPO 

hyperparameters = {
    # Run
    "timesteps_per_batch": 10000,   # timesteps per batch
    "max_timesteps_per_episode": 1000,  # timesteps per episode
        
    # Algorithm
    "gamma": 0.90,                 # Discount factor    
    "n_updates_per_iteration": 18,  # number of updates per iteration
    "clip": 0.2,                   # Recommended
    "lr": 0.005,                   # Learning rate
    "num_minibatches": 6,          # K in the paper
    "lam": 0.99,                   # Lambda for GAE-Lambda
    "ent_coef": 0.05,              # Entropy coefficient, higher penalizes overdeterministic policies 
    "max_grad_norm": 0.5,          # Gradient clipping threshold
    "target_kl": 0.02,             # Target KL-divergence

    # Misc
    "save_freq": 2,  # How often we save in number of iterations
    "render": False,  # If we should render during rollout 
    "full_render": False, # Watch full training   
    "render_every_i": 1,  # Only render every i iterations
    "log": True, # If we push logs to neptune
    "capture_frames": True
}
