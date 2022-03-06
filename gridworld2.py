import gym 						#type: ignore
from gym.core import GoalEnv	#type: ignore
from gym import error			#type: ignore

import numpy as np 				#type: ignore
import random
import typing
import pdb
# import constants
from constants import *
# from obstacles


noise_samples = [(1,0), (-1, 0), (0, 1), (0,1)]

def state_noise(k):
	return random.sample(noise_samples + [(0,0)]*k)


is_blocked = { 
	EMPTY: lambda state: False,			
	BLOCK: lambda state: True,
	WIND:  lambda state: False,
	RANDOM_DOOR: lambda state: False if random.random() < 0.1 else True,
}




class GridworldEnv(GoalEnv):
	"""
	The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

	"""
	def __init__(self, size, start, new_goal):
		self.size = size
		self.start = start
		self.new_goal = new_goal
		self.grid = np.zeros((size, size))

		self.reward_range = (0,1)

	def reset(self):
		# self.state = self.start()
		# self.goal = self.new_goal()
		self.state = self.start
		self.goal = self.new_goal
		# self.goal = self.rand_state()
		return self.get_obs()

	def step(self, action):
		state = self.state 
		proposed_next_state = state + action
		next_state_type = self.grid[tuple(proposed_next_state)]
		next_state = transitions[next_state_type](state, proposed_next_state)

		# print(next_state_type)
		# pdb.set_trace()

		if np.abs(state - next_state).sum() > 1.05:
			pdb.set_trace()

		reward = self.compute_reward(next_state, self.goal)
		self.state = next_state
		return self.get_obs(), self.compute_reward(next_state, self.goal), False, {}

	def compute_reward(self, ag, dg):
		return 1 if (ag == dg).all() else 0

	def rand_state(self):
		return np.array([np.random.randint(0, size), np.random.randint(0, size)])

	def set_state(self, state): 
		self.state = state

	def get_state(self): 
		return self.state

	def get_obs(self):
		return {
			"state": self.state,
			"observation": self.state,
			"achieved_goal": self.state,
			"desired_goal": self.goal,
		}

def create_map_1():
	size = 8
	start  = np.array([1,size//2 -1])
	new_goal  = np.array([1, size//2 +1])
	gridworld = GridworldEnv(size, start, new_goal)
	for i in range(size):
		#Borders
		gridworld.grid[0,i] = BLOCK
		gridworld.grid[size-1,i] = BLOCK
		gridworld.grid[i,0] = BLOCK
		gridworld.grid[i, size-1] = BLOCK

		#Wall through the middle
		gridworld.grid[i,size//2 ] = BLOCK


	gridworld.grid[1,size//2] = RANDOM_DOOR
	gridworld.grid[size-2, size//2] = EMPTY

	return gridworld
