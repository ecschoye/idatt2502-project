import torch
from environment import create_mario_env
from network import DiscreteActorCriticNN
from ppo import PPO
import evaluate

STRINGS = {
  'menu': """
    Welcome to Proximal Policy Main
    [1] Train New Agents
    [2] Continue Training Previous Agents
    [3] Test
  """,
  'loading_agents' : "Loading actor and critic models from file...",
  'loading_success': "Actor and critic models successfully loaded",
  'from_scratch': "Training from scratch",
  'testing': "Testing trained model"
}

def main(): 
  """
    Main Function to run training or testing 
  """

  env = create_mario_env()

  print(STRINGS['menu'], flush=True)
  choice = int(input("Enter a number:"))

  if choice == 1:
    train(env, '', '')
  elif choice == 2:
    train(env, 'ppo_actor.pth', 'ppo_critic.pth')
  elif choice == 3: 
    test(env, 'ppo_actor.pth', 'ppo_critic.pth')  

def train(env, actor_model, critic_model): 
  model = PPO(env)

  if actor_model != '' and critic_model != '': 
    print(STRINGS['loading_agents'], flush=True)
    model.actor.load_state_dict(torch.load(actor_model))
    model.critic.load_state_dict(torch.load(critic_model))
    print(STRINGS['loading_success'], flush=True)
  else:
    print(STRINGS['from_scratch'], flush=True)

  model.learn(total_timesteps=30_000)

def test(env, actor_model, critic_model):
  print(STRINGS['testing'], flush=True)

  # Extract out dimensions of observation and action spaces
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  # Build our policy the same way we build our actor model in PPO
  # Load in the actor model saved by the PPO algorithm
  policy = DiscreteActorCriticNN(obs_dim, act_dim)
  policy.load_state_dict(torch.load(actor_model))

  evaluate(policy, env, render=True)

main()