from random import choice, random
import numpy as np
import torch
import torch.nn as nn

from utils import PERBuffer

# see model_loader.py for the functions that create these agents with the required parameters
class SimpleAgent():
    """
    Simple Agent that always turns on the sensor
    """
    def __init__(self, cfg):
        self.cfg = cfg
        
    def policy(self, trajectory):
        return (0,1)
    def max_action(self, trajectory):
        return (0,1)
    def update(self, s, a, r, s_, not_done):
        pass


class SensorQAgent:
    """
    Tabular Q Agent
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.epsilon = cfg['epsilon']
        self.q_values = {} # trajectory is a tuple of ((state, action) for the last n steps)
        self.update_count = {}
        self.action_space = [(1,0), (0,1)] # (0,1) -> turn on sensor, (1,0) -> turn off sensor
        self.ds = eval(cfg['ds'])

    # epsilon greedy policy
    def policy(self, trajectory):
        trajectory = self.ds(trajectory)
        if trajectory in self.q_values and random() > self.epsilon:
            return self.max_action(trajectory)
        else:
            return choice(self.action_space)
    
    # greedy action
    def max_action(self, trajectory):
        trajectory = self.ds(trajectory)
        if trajectory in self.q_values:
            return self.action_space[np.argmax(self.q_values[trajectory])]
        else:
            return(1,0)
    
    def update(self, s, a, r, s_, not_done):
        s = self.ds(s)
        s_ = self.ds(s_)
        if s not in self.q_values:
            self.q_values[s] = [0,0]
            self.update_count[s] = [0,0]
        if s_ not in self.q_values:
            self.q_values[s_] = [0,0]
            self.update_count[s_] = [0,0]
        self.q_values[s][a] +=   self.cfg['alpha'] * (r + not_done * self.cfg['gamma'] * max(self.q_values[s_]) - self.q_values[s][a])
        self.q_values[s][a] +=   self.cfg['alpha'] * (r + not_done * self.cfg['gamma'] * max(self.q_values[s_]) - self.q_values[s][a])
        self.epsilon = max(self.epsilon * self.cfg['epsilon_decay'], self.cfg['epsilon_min'])
        self.update_count[s][a] += 1


class SensorDQAgent:
    """
    Double DQN based agent
    """
    def __init__(self,cfg):
        self.rbuffer = PERBuffer(cfg['buffer_size'], cfg['buffer_alpha'], cfg['buffer_beta'], cfg['input size'])
        self.batch_size = cfg['batch_size']
        self.gamma = cfg['gamma']
        self.ds = eval(cfg['ds'])# convert inputs into appropriate datastructure for this model
        self.action_space = ((1,0),(0,1))
        
        self.target_update_interval = cfg['target_update_interval']
        self.t = 0

        self.epsilon = cfg['epsilon']
        self.epsilon_decay = cfg['epsilon_decay']
        self.epsilon_min = cfg['epsilon_min']

        self.net = torch.nn.Sequential(
            torch.nn.Linear(cfg['input size'], cfg['nhidden']),
            torch.nn.BatchNorm1d(cfg['nhidden']),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg['nhidden'], cfg['nhidden']),
            torch.nn.BatchNorm1d(cfg['nhidden']),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg['nhidden'], 2)
        )
        self.target = torch.nn.Sequential(
            torch.nn.Linear(cfg['input size'], cfg['nhidden']),
            torch.nn.BatchNorm1d(cfg['nhidden']),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg['nhidden'], cfg['nhidden']),
            torch.nn.BatchNorm1d(cfg['nhidden']),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg['nhidden'], 2)
        )

        self.target.load_state_dict(self.net.state_dict())

        self.lossf = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg['alpha'])
    
    def policy(self, state):
        state = self.ds(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if np.random.random() < self.epsilon:
            return choice(self.action_space)
        else:
            self.net.eval()
            return self.action_space[torch.argmax(self.net(state))]

    def max_action(self, state):
        state = self.ds(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        self.net.eval()
        return self.action_space[torch.argmax(self.net(state))]
    
    def update(self, s,a,r,s_,done):
        s = self.ds(s)
        s_ = self.ds(s_)
        self.net.train()
        self.rbuffer.add(s, a, r, s_)

        transitions = self.rbuffer.sample(self.batch_size)
        q_values = self.net(transitions['obs'])
        next_q_values = self.target(transitions['next_obs'])
        target_q_values = [reward + self.gamma * torch.max(next_q_values[i]) for i, reward in enumerate(transitions['reward'])]

        td_error = [self.lossf(q_values[i][action[1]], target_q_values[i]) for i, action in enumerate(transitions['action'])]        
        loss = torch.stack(td_error).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.t % self.target_update_interval:
            self.target.load_state_dict(self.net.state_dict())
        self.t += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DuelingDQN(nn.Module):
    def __init__(self, input_size, nhidden=128, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.action_space = [(1,0), (0,1)] # (0,1) -> turn on sensor, (1,0) -> turn off sensor
        self.inputLinear = nn.Sequential(
            nn.Linear(input_size, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.ReLU()
        )
        self.advantageLinear = nn.Sequential(
            nn.Linear(nhidden+2, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, 2)
        )
        self.valueLinear = nn.Sequential(
            nn.Linear(nhidden+2, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, 1)
        )
    
    def forward(self, x, w):
        x = self.inputLinear(x)
        w = torch.tile(w,(x.shape[0],1))
        x = torch.cat((x, w), dim=1)
        advantage = self.advantageLinear(x)
        value = self.valueLinear(x)
        return value + advantage - advantage.mean()

    def policy(self, trajectory, weight):
        if random() < self.epsilon:
            return choice(self.action_space)
        else:
            self.eval()
            return self.action_space[torch.argmax(self.forward(trajectory, weight))]
