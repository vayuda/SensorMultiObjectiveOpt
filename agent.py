from random import choice, random, sample
from enum import Enum
import numpy as np
import torch
import torch.nn as nn


class Dir(Enum):
    U = 0
    D = 1
    L = 2
    R = 3


'''
class to define agent which moves in a circle around the grid
the agent moves in a circle around the grid, and stops at the corners for a certain number of steps
'''
class CircleUser:
    def __init__(self,cfg):
        self.start = cfg['start']
        self.goal = cfg['goal']
        self.grid = cfg['grid']
        self.rows= len(self.grid)
        self.cols = len(self.grid[0])
        self.x = self.start[0]
        self.y = self.start[1]
        self.direction = Dir.R  # Initial direction
        self.wait_time = cfg['wait_time']
        self.corner_timer = self.wait_time  # Internal timer for waiting in a corner
        self.first_move = True
        self.move_probability = cfg['move_probability']

    def reset(self):
        self.x = self.start[0]
        self.y = self.start[1]
        self.direction = Dir.R
        self.first_move = True

    def move(self):
        match self.direction:
            case Dir.R:
                if self.x < self.cols - 1:
                    self.x+=1
                    return (1,0)
                else:
                    # print('turning down')
                    self.direction = Dir.D
                    return self.move()
            case Dir.D:
                if self.y < self.rows-1:
                    self.y+=1
                    return (0,1)
                else:
                    # print('turning left')
                    self.direction = Dir.L
                    return self.move()
            case Dir.L:
                if self.x > 0: 
                    self.x-=1
                    return (-1,0)
                else:
                    # print('turning up')
                    self.direction = Dir.U
                    return self.move()
            case Dir.U:
                if self.y > 0:
                    self.y -= 1
                    return (0,-1)
                else:
                    # print('turning right')
                    self.direction = Dir.R
                    return self.move()

    def corner_policy(self):
        # Check if the agent is at one of the four corners
        if (self.x, self.y) in {(0, 0), (0, self.rows-1), (self.cols-1, 0), (self.cols-1, self.rows-1)}:
            if random() < self.move_probability:
                return self.move()
            else:
                return (0,0)


    def should_move_middle(self, prob = 0.5):
        # Check if the agent is at the midpoint between two corners
        return (self.x, self.y) in {(0, self.cols//2), 
                                (self.rows//2, 0), 
                                (self.rows -1, self.cols//2),
                                (self.rows//2, self.cols -1)} and random() < prob

    def policy(self):
        if self.first_move:
            self.first_move = False
            self.x+=1
            return (1,0)
        move = self.corner_policy()
        if move:
            return move
            
        # elif self.should_move_middle():
        #     # print('moving to middle')
        #     match self.direction:
        #         case Dir.R:
        #             self.direction = Dir.D
        #         case Dir.D:
        #             self.direction = Dir.L
        #         case Dir.L:
        #             self.direction = Dir.U
        #         case Dir.U:
        #             self.direction = Dir.R
        return self.move()



class CircleUser:
    def __init__(self,cfg):
        self.start = cfg['start']
        self.goal = cfg['goal']
        self.grid = cfg['grid']
        self.rows= len(self.grid)
        self.cols = len(self.grid[0])
        self.x = self.start[0]
        self.y = self.start[1]
        self.direction = Dir.R  # Initial direction
        self.wait_time = cfg['wait_time']
        self.corner_timer = self.wait_time  # Internal timer for waiting in a corner
        self.first_move = True
        self.move_probability = cfg['move_probability']

    def reset(self):
        self.x = self.start[0]
        self.y = self.start[1]
        self.direction = Dir.R
        self.first_move = True

    def move(self):
        match self.direction:
            case Dir.R:
                if self.x < self.cols - 1:
                    self.x+=1
                    return (1,0)
                else:
                    # print('turning down')
                    self.direction = Dir.D
                    return self.move()
            case Dir.D:
                if self.y < self.rows-1:
                    self.y+=1
                    return (0,1)
                else:
                    # print('turning left')
                    self.direction = Dir.L
                    return self.move()
            case Dir.L:
                if self.x > 0: 
                    self.x-=1
                    return (-1,0)
                else:
                    # print('turning up')
                    self.direction = Dir.U
                    return self.move()
            case Dir.U:
                if self.y > 0:
                    self.y -= 1
                    return (0,-1)
                else:
                    # print('turning right')
                    self.direction = Dir.R
                    return self.move()

    def corner_policy(self):
        # Check if the agent is at one of the four corners
        if (self.x, self.y) in {(0, 0), (0, self.rows-1), (self.cols-1, 0), (self.cols-1, self.rows-1)}:
            if random() < self.move_probability:
                return self.move()
            else:
                return (0,0)


    def should_move_middle(self, prob = 0.5):
        # Check if the agent is at the midpoint between two corners
        return (self.x, self.y) in {(0, self.cols//2), 
                                (self.rows//2, 0), 
                                (self.rows -1, self.cols//2),
                                (self.rows//2, self.cols -1)} and random() < prob

    def policy(self):
        if self.first_move:
            self.first_move = False
            self.x+=1
            return (1,0)
        move = self.corner_policy()
        if move:
            return move
            
        elif self.should_move_middle():
            # print('moving to middle')
            match self.direction:
                case Dir.R:
                    self.direction = Dir.D
                case Dir.D:
                    self.direction = Dir.L
                case Dir.L:
                    self.direction = Dir.U
                case Dir.U:
                    self.direction = Dir.R
        return self.move()

class SimpleAgent:
    def policy(self, trajectory):
        return (0,1)
    def max_action(self, trajectory):
        return (0,1)
    def update(self, s, a, r, s_, not_done):
        pass


class OptimalAgent: 
    def __init__(self):
        self.visited = set()
    def policy(self,trajectory):
        if trajectory in self.visited:
            return (1,0)
        else:
            self.visited.add(trajectory)
            return (0,1)
        
    def max_action(self,trajectory):
        return self.policy(trajectory)
    
    def update(self, s, a, r, s_, not_done):
        pass


class SensorQAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.epsilon = cfg['epsilon']
        self.q_values = {} # trajectory is a tuple of ((state, action) for the last n steps)
        self.update_count = {}
        self.action_space = [(1,0), (0,1)] # (0,1) -> turn on sensor, (1,0) -> turn off sensor

    def policy(self, trajectory):
        if trajectory in self.q_values and random() > self.epsilon:
            return self.max_action(trajectory)
        else:
            return choice(self.action_space)
    
    def max_action(self, trajectory):
        if trajectory in self.q_values:
            return self.action_space[np.argmax(self.q_values[trajectory])]
        else:
            return(1,0)
    
    def update(self, s, a, r, s_, not_done):
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


class DQN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.epsilon = cfg['epsilon']
        self.action_space = [(1,0), (0,1)] # (0,1) -> turn on sensor, (1,0) -> turn off sensor
        self.rewards = []
        self.network = nn.Sequential(
            nn.Linear(cfg['trajectory_length'], cfg['nhidden']),
            nn.BatchNorm1d(cfg['nhidden']),
            nn.ReLU(),
            nn.Linear(cfg['nhidden'], cfg['nhidden']),
            nn.BatchNorm1d(cfg['nhidden']),
            nn.ReLU(),
            nn.Linear(cfg['nhidden'], 2)
        )
    
    def forward(self, x):
        return self.network(x)

    def policy(self, trajectory):
        if random() < self.epsilon:
            return choice(self.action_space)
        else:
            self.eval()
            return self.action_space[torch.argmax(self.network(trajectory))]
    
    def max_action(self, trajectory):
        self.eval()
        return self.action_space[torch.argmax(self.network(trajectory))]



    


class DuelingDQN(nn.Module):
    def __init__(self, trajectory_length, nhidden=128, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.action_space = [(1,0), (0,1)] # (0,1) -> turn on sensor, (1,0) -> turn off sensor
        self.inputLinear = nn.Sequential(
            nn.Linear(trajectory_length * 4, nhidden),
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
