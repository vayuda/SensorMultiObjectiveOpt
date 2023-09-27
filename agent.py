from random import choice, random
from enum import Enum
import numpy as np


class Dir(Enum):
    U = 0
    D = 1
    L = 2
    R = 3

class CircleAgent:
    def __init__(self,env,wait_time):
        self.env = env
        self.start = env.start
        self.goal = env.goal
        self.grid = env.grid
        self.rows= len(self.grid)
        self.cols = len(self.grid[0])
        self.x = self.start[0]
        self.y = self.start[1]
        self.direction = Dir.R  # Initial direction
        self.wait_time = wait_time
        self.corner_timer = wait_time  # Internal timer for waiting in a corner

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

    def should_wait(self):
        # Check if the agent is at one of the four corners
        return (self.x, self.y) in {(0, 0), (0, self.rows-1), (self.cols-1, 0), (self.cols-1, self.rows-1)}

    def should_move_middle(self):
        # Check if the agent is at the midpoint between two corners
        return (self.x, self.y) in {(0, self.cols//2), 
                                (self.rows//2, 0), 
                                (self.rows -1, self.cols//2),
                                (self.rows//2, self.cols -1)} and random() < 0.5

    def policy(self):
        if self.should_wait():
            
            if self.corner_timer == 0:
                self.corner_timer = self.wait_time
                # print('finished waiting')
            else:
                self.corner_timer -= 1
                # print('waiting')
                return (0,0)
            
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


class SensorQAgent:
    def __init__(self, env, alpha, epsilon, w1=1, w2=1):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_values = {} # trajectory is a tuple of ((state, action) for the last n steps)
        self.action_space = [(1,0), (0,1)] # (0,1) -> turn on sensor, (1,0) -> turn off sensor
        self.visited = set()
        self.sensed = set()
        self.w1 = w1
        self.w2 = w2
        self.rewards = []

    def policy(self, trajectory):
        if trajectory in self.q_values:
            if random() < self.epsilon:
                return choice(self.action_space)
            else:
                return self.action_space[np.argmax(self.q_values[trajectory])]
        else:
            return choice(self.action_space)
    
    def update(self, trajectory, next_trajectory):
        self.visited.add(trajectory[-1][0])
        if trajectory[-1][1] == (0,1):
            self.sensed.add(trajectory[-1][0])
        reward = self.w1 * len(self.sensed)/len(self.visited) - self.w2 * (next_trajectory[-1][1] == (0,1))
        self.rewards.append(reward)
        if trajectory not in self.q_values:
            self.q_values[trajectory] = [0,0]
        if next_trajectory not in self.q_values:
            self.q_values[next_trajectory] = [0,0]
        self.q_values[trajectory][0] += self.alpha * (reward + max(self.q_values[next_trajectory]) - self.q_values[trajectory][0])
        self.q_values[trajectory][1] += self.alpha * (reward + max(self.q_values[next_trajectory]) - self.q_values[trajectory][1])
 


'''
q learing agent that can turnon and off a sensor
the agent policy is given the last three state action pairs, decide whether to activate the sensor.
R = W1 * battery + W2 * quality
battery is -1 whenever action is taken
quality is proportion of states sensed

first handcode an agent to traverse the edge of the map, and stop at the corners, 
but sometimes go to the middle before returning to the edge

use this to "train" a p(s)->s'
'''    
