import tkinter as tk


class GridWorld:
    def __init__(self, start,goal, grid, cell_size=50):
        self.rows =len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.goal = goal
        self.goal_reached = False
        self.cell_size = cell_size
        self.agent_location = start
        self.grid = grid

    
    def move_agent(self, newX, newY):
        self.agent_location = (newX, newY)
        if self.agent_location == self.goal:
            self.goal_reached = True

    def action_space(self):
        x, y = self.agent_location
        actions = [(0,0)]
        
        if x > 0 and self.grid[y][x-1] != 'w':
            actions.append((-1,0))
        if x < self.cols - 1 and self.grid[y][x+1] != 'w':
            actions.append((1,0))
        if y > 0 and self.grid[y-1][x] != 'w':
            actions.append((0,-1))
        if y < self.rows - 1 and self.grid[y+1][x] != 'w':
            actions.append((0,1))
        return actions





