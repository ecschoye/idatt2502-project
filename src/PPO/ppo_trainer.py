import torch
import os
from src.environment import create_mario_env
from src.PPO.network.network import DiscreteActorCriticNN
from src.PPO.model.ppo import PPO
from src.PPO.utils.evaluate import evaluate
from PPO.utils.ppo_parameters import PPOHyperparameters

ACTOR_PATH = "./src/PPO/network/ppo_actor.pth"
CRITIC_PATH = "./src/PPO/network/ppo_critic.pth"

TIMESTEPS = 500_000
MAP = "SuperMarioBros-1-1-v0"

class PPOTrainer:
    def __init__(self):
        self.env = create_mario_env(MAP)
        self.env.metadata['render-modes']="human"
        self.parameters = PPOHyperparameters()


    def train(self, log=False, load_models=False):
        """
        Main Function to run training or testing
        """
        self.parameters.set_logging(log)

        print(f"""
            - - - - - - Starting loop - - - - - - \n
            Timesteps to run: {TIMESTEPS} \n
            Environment: {MAP} \n
            Logging {"is turned on." if log else "is not turned on."} \n
            {"Loading models from file." if load_models else "Training from scratch."} \n
            """, flush=True)
        
        self.model = PPO(self.env, self.parameters.get_hyperparameters())

        if load_models:
            print("Loading actor and critic models from file...", flush=True)
            result = self.__load_models()
            if result:
                print("Actor and critic models successfully loaded", flush=True)
            else: 
                print("No models found, training from scratch", flush=True)
        
        self.model.learn(total_timesteps=TIMESTEPS)
    
    def test(self):
        """
        Function to run the actor model and test it
        """
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.n
        self.env.metadata['render-modes']="human"
        policy = DiscreteActorCriticNN(obs_dim, act_dim)
        if os.path.exists(ACTOR_PATH):
            policy.load_state_dict(torch.load(ACTOR_PATH, map_location=("cuda" if torch.cuda.is_available() else "cpu")))
            evaluate(policy, self.env, render=True)
        else: 
            print("No actor model found, please train the model first")

    def __load_models(self) -> bool:
        if os.path.exists(ACTOR_PATH) and os.path.exists(CRITIC_PATH):
            self.model.actor.load_state_dict(torch.load(ACTOR_PATH, map_location=("cuda" if torch.cuda.is_available() else "cpu")))
            self.model.critic.load_state_dict(torch.load(CRITIC_PATH, map_location=("cuda" if torch.cuda.is_available() else "cpu")))
            return True
        return False
            