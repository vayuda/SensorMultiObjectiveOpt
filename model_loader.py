from user import CircleUser 
from agent import SimpleAgent, SensorQAgent, SensorDQAgent

def create_CircleUser(cfg):
    config = {
        'move probability': cfg['user move probability'],
        'wait time': cfg['user wait time'],
        'env': cfg['env']     
    }
    return CircleUser(config)

def create_SimpleAgent():
    config = {
        'name': "Simple Agent",
        'savable net': False
    }
    return SimpleAgent(config)

def create_SensorQAgent(config):
    agent_config = {
        'name': "Tabular Q Agent",
        'savable net': False,
        'ds': "tuple",

        'alpha': 0.01,
        'gamma': 0.9,

        'epsilon': 1,
        'epsilon_decay': 0.99,
        'epsilon_min': 0.01,
    }
    agent_config['epsilon_decay'] = agent_config['epsilon_min'] ** (1/config['ep len'])
    return SensorQAgent(agent_config)

def create_SensorDQAgent(config):
    agent_config = {
        'model': "SensorDQAgent",
        'name': 'DQN Agent',
        'savable net': True,
        'ds': "lambda x: torch.tensor(x,dtype=torch.float32)",

        'buffer_size': 10000,
        'buffer_alpha': 1,  
        'buffer_beta': 1,

        'target_update_interval': 100,
        'batch_size': 64,
        'nhidden': 64,

        'alpha': 0.01,
        'gamma': 0.9,

        'epsilon': 1,
        'epsilon_decay': 0.99,
        'epsilon_min': 0.01,
    }
    agent_config['epsilon_decay'] = agent_config['epsilon_min'] ** (1/config['ep len'])
    return SensorDQAgent(agent_config)

def create_agent(agent_name, config):
    if agent_name == 'SimpleAgent':
        return create_SimpleAgent()
    elif agent_name == 'SensorQAgent':
        return create_SensorQAgent(config)
    elif agent_name == 'SensorDQAgent':
        return create_SensorDQAgent(config)
    else:
        raise Exception("Invalid agent name")

def create_user(user_name, config):
    if user_name == 'CircleUser':
        return create_CircleUser(config)
    else:
        raise Exception("Invalid user model")