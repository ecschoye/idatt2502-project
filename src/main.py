import time

import torch

from environment import create_mario_env
from model.dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize DQN
input_shape = (4, 84, 84)  # Replace this with the correct input shape
n_actions = 7  # Replace this with the correct number of actions
model = DQN(input_shape, n_actions).to(device)

# Load the model
model.load(device, target=False)  # Set target=True if you saved a target model


env = create_mario_env()
state = env.reset()
done = False

while not done:
    # Prepare the state for the model (e.g., reshape, normalize, etc.)
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    # Get the action from the model
    with torch.no_grad():
        q_values = model(state)
    action = torch.argmax(q_values).item()

    # Take the action in the environment
    next_state, reward, done, _ = env.step(action)
    print(reward)
    print(done)
    print(_)
    # Update the current state
    state = next_state

    env.render()
    time.sleep(0.05)

    # Your code for rendering or logging goes here

env.close()
