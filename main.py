from utils import graph_training_rewards, graph_test_rewards
from args import get_config
from train import train

config = get_config()
train(config)
graph_training_rewards(config['save path'])
graph_test_rewards(config['save path'])