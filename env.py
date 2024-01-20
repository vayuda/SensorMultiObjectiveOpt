from collections import deque
from utils import square_print
class GridWorld:
    def __init__(self, config):
        self.cfg = config
        self.grid = [['-' for j in range(config['grid size'])] for i in range(config['grid size'])],
        self.agent_location = self.cfg['start']
        self.size = config['grid size']
        self.start = config['start']
        self.goal = config['goal']
    def move_agent(self, newX, newY):
        self.agent_location = (newX, newY)
        self.is_terminal()

    def is_terminal(self):
        self.goal_reached = True
        return self.agent_location == self.goal
    
    def reset(self):
        self.agent_location = self.start
        self.goal_reached = False

    def generate_user_trajectories(self, user, n_episodes, max_ep_len):
        data = []
        
        for i in range(n_episodes):
            trajectory = [self.start]
            self.reset()
            user.reset()
            for j in range(max_ep_len):
                action = user.policy()
                newX = self.agent_location[0] + action[0]
                newY = self.agent_location[1] + action[1]
                trajectory.append((newX,newY))
                self.move_agent(newX, newY)
                if self.is_terminal():
                    break
            data.append(trajectory)

        return data

    def sim_episode(self, episode, agent, train, show_actions = False):
        trajectory = deque([0]*self.cfg['trajectory length'],maxlen=self.cfg['trajectory length'])
        coverage_window = deque(maxlen=self.cfg['coverage window'])
        covered_states = deque(maxlen=self.cfg['coverage window'])
        sensor_usage = deque(maxlen=self.cfg['coverage window'])
        sensed = set()
        visited = set()
        activation_count = 0
        ep_reward = 0
        viz = ""

        for t,position in enumerate(episode):
            visited.add(position)
            s = trajectory
            if train:
                action = agent.policy(s)
            else:
                action = agent.max_action(s)

            if action[1]: #give agent knowledge of position if sensor was activated
                viz += "1"
                sensed.add(position)
                sensor_usage.append(1)
                if position not in covered_states:
                    coverage_window.append(1)
                else:
                    coverage_window.append(0)
                covered_states.append(position)
                coverage_window.append(1)
                activation_count +=1
                trajectory.append(position[0])
                trajectory.append(position[1])
            elif self.cfg['exact position']:
                trajectory.append(position[0])
                trajectory.append(position[1])
                coverage_window.append(0)
            else:
                viz+= "0"
                trajectory.append(-1)
                trajectory.append(-1)
                coverage_window.append(0)
                sensor_usage.append(0)

            r = self.cfg['w1'] * sum(coverage_window)/len(coverage_window) - self.cfg['w2']* action[1]
            s_ = trajectory
            ep_reward += r
            if train:
                agent.update(s, action[1], r, s_, self.is_terminal())
        if show_actions:
            square_print(viz, self.size)
        return ep_reward, activation_count





