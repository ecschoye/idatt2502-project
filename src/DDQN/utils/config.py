from enum import Enum


class DDQNParameters(Enum):
    MEMORY_SIZE = 20000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00025
    GAMMA = 0.90
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY_RATE = 10**5
    TARGET_UPDATE_FREQUENCY = 5000


class DDQNTrainingParameters(Enum):
    NUM_EPISODES = 5000
    ENV_NAME = "SuperMarioBros-1-1-v0"
