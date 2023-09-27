This repository contains code to simulate and solve the following problem:
Phones have multiple sensors which provide information at the expense of depleting the phones battery.
We intend to frame whether to activate a sensor as a multi-objective reinforcement problem.

Currently this code contains a toy example just to show proof of concept.
In this 5x5 grid world with walls in the middle of each quadrant, a hand crafted agent 
which simulates a person walking through various locations in the day. 
The agent tries to visit the four corners of the grid world and stay there for some period of time before moving onto the next corner in clockwise fashion.
However, at the midpoint of this transition between corners, the agent has 50% chance to instead travel towards the opposite side of the world, passing through the middle.

A tabular Q-agent was created to determine whether to activate a gps sensor, which reveals the positions of the agent. 
The Q-agent recieves as input, the previous {trajectory_length} positions of the agent. 
If the gps was not activated, the position is recorded as (-1,-1)
The reward for the agent is a simple linear combination as follows:
reward = w1 * (number of unique positions sensed / number of unique positions visited) - self.w2 * (1 if sensor was activated, else 0)

Directory Structure:
gridworld.py contains code for simulating the environment
agent.py contains code for the handcrafted agent and the Q-agent
sim.ipynb contains code for running the simulation described above


