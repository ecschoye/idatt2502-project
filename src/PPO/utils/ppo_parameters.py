# Hyperparameters for PPO
class PPOHyperparameters:
    def __init__(self):
        self.hyperparameters = {
            # Run
            "timesteps_per_batch": 5000,  # timesteps per batch
            "max_timesteps_per_episode": 800,  # timesteps per episode
            # Algorithm
            "gamma": 0.80,  # Discount factor
            "n_updates_per_iteration": 20,  # number of updates per iteration
            "clip": 0.2,  # Recommended
            "lr": 0.005,  # Learning rate
            "num_minibatches": 4,  # K in the paper
            "lam": 0.99,  # Lambda for GAE-Lambda
            "ent_coef": 0.03,  # Entropy coefficient
            "max_grad_norm": 0.2,  # Gradient clipping threshold
            "target_kl": 0.05,  # Target KL-divergence
            # Misc
            "save_freq": 1,  # How often we save in number of iterations
            "render": False,  # If we should render during rollout
            "full_render": False,  # Watch full training
            "render_every_i": 1,  # Only render every i iterations
            "log": True,  # If we push logs to neptune
            "capture_frames": True,
            "run_notes": "",
        }

    def get_hyperparameters(self):
        return self.hyperparameters

    def set_logging(self, log):
        self.hyperparameters["log"] = log
