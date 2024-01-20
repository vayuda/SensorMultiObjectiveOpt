from random import random
from enum import Enum

class Dir(Enum):
    U = 0
    D = 1
    L = 2
    R = 3


class CircleUser:
    """
    User which moves in a circle around the grid 
    and at corners, has some probability of staying at each timestep
    """
    def __init__(self, cfg):
        self.env = cfg['env']
        self.x = self.env.start[0]
        self.y = self.env.start[1]
        self.direction = Dir.R  # Initial direction
        self.wait_time = cfg['wait time']
        self.corner_timer = self.wait_time  # Internal timer for waiting in a corner
        self.first_move = True
        self.move_probability = cfg['move probability']

    def reset(self):
        self.x = self.env.start[0]
        self.y = self.env.start[1]
        self.direction = Dir.R
        self.first_move = True

    def move(self):
        match self.direction:
            case Dir.R:
                if self.x < self.env.size - 1:
                    self.x+=1
                    return (1,0)
                else:
                    # print('turning down')
                    self.direction = Dir.D
                    return self.move()
            case Dir.D:
                if self.y < self.env.size-1:
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
        if (self.x, self.y) in {
            (0, 0), 
            (0, self.env.size-1), 
            (self.env.size-1, 0), 
            (self.env.size-1, self.env.size-1)}:
                if random() < self.move_probability:
                    return self.move()
                else:
                    return (0,0)


    def should_move_middle(self, prob = 0.5):
        # Check if the agent is at the midpoint between two corners
        return (self.x, self.y) in {(0, self.env.size//2), 
                                (self.env.size//2, 0), 
                                (self.env.size -1, self.env.size//2),
                                (self.env.size//2, self.env.size -1)} and random() < prob

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