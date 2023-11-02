from src.agent.ddqn_agent import DDQNAgent
from src.environment import create_mario_env


def run_mario():
    print("Creating environment")
    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n

    agent = DDQNAgent(env, state_space, action_space)

    num_episodes = 10
    print("Training for {} episodes".format(num_episodes))

    agent.train(num_episodes)

    agent.save()


if __name__ == "__main__":
    print("Starting training")
    run_mario()
