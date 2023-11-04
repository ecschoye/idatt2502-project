import torch
from src.environment import create_mario_env
from src.PPO.network.network import DiscreteActorCriticNN
from src.PPO.model.ppo import PPO
from src.PPO.utils.evaluate import evaluate
from src.PPO.utils.hyperparameters import hyperparameters

ACTOR_PATH = "./src/PPO/network/ppo_actor.pth"
CRITIC_PATH = "./src/PPO/network/ppo_critic.pth"
STRINGS = {
    "menu": """
    Welcome to Proximal Policy Main
    [1] Run Single Training
    [2] Run Training Loop
    [3] Test
  """,
    "notes": "Would you like to add any notes?",
    "loading_agents": "Loading actor and critic models from file...",
    "loading_success": "Actor and critic models successfully loaded",
    "from_scratch": "Training from scratch",
    "testing": "Testing trained model",
    "run_starting" : "Training run started",
    "run_finished" : "Training run finished",
    "training_loop": "Starting training loop"
}
TIMESTEPS = 300_000

def main():
    """
    Main Function to run training or testing
    """

    env = create_mario_env("SuperMarioBros-1-1-v0")
    env.metadata['render-modes']="human"

    print(STRINGS["menu"], flush=True)
    choice = int(input("Enter a number:"))

    if choice != 3: 
        print(STRINGS["notes"], flush=True)
        notes = input("Enter notes:")

    if choice == 1:
        train(env, ACTOR_PATH, CRITIC_PATH, hyperparameters, notes)
    elif choice == 2:
        train_loop(env, hyperparameters, notes)
    elif choice == 3:
        test(env, ACTOR_PATH, CRITIC_PATH)
    
    env.close()

def train_loop(env, parameters, notes):
    parameters['run_notes'] = str("\"" + notes + "\"")
    ITERATIONS = 8
    print(STRINGS["training_loop"], flush=True)
    for i in range(ITERATIONS): 
        print(STRINGS["run_starting"], flush=True)
        model = PPO(env, parameters)
        if i != 0:
            model.actor.load_state_dict(torch.load(ACTOR_PATH, map_location=("cuda" if torch.cuda.is_available() else "cpu")))
            model.critic.load_state_dict(torch.load(CRITIC_PATH, map_location=("cuda" if torch.cuda.is_available() else "cpu")))
        model.learn(TIMESTEPS)
        print(STRINGS["run_finished"], flush=True)
        

def train(env, actor_model, critic_model, parameters, notes):
    parameters['run_notes'] = str("\"" + notes + "\"")
    model = PPO(env, parameters)
    if actor_model != "" and critic_model != "":
        print(STRINGS["loading_agents"], flush=True)
        model.actor.load_state_dict(torch.load(actor_model, map_location=("cuda" if torch.cuda.is_available() else "cpu")))
        model.critic.load_state_dict(torch.load(critic_model, map_location=("cuda" if torch.cuda.is_available() else "cpu")))
        print(STRINGS["loading_success"], flush=True)
    else:
        print(STRINGS["from_scratch"], flush=True)

    model.learn(total_timesteps=TIMESTEPS)


def test(env, actor_model, critic_model):
    print(STRINGS["testing"], flush=True)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    env.metadata["render_modes"] = None
    # Build our policy the same way we build our actor model in PPO
    # Load in the actor model saved by the PPO algorithm
    policy = DiscreteActorCriticNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(actor_model))

    evaluate(policy, env, render=True)


main()
