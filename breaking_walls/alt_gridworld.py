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


SUCCESS_CHANCE = .1

transitions = { 
	EMPTY: lambda last_state, state: (state, False),			#Just move
	BLOCK: lambda last_state, state: (last_state, False),	#Prevent agent from moving
	WIND:  lambda last_state, state: (state + state_noise(4), False),
			#Currently does not work because state may be blocked
			#To fix later
	BREAKING_DOOR: lambda last_state, state: (state, False) if random.random() < SUCCESS_CHANCE \
		else (last_state, True),
		# else (last_state, False),
	NONBREAKING_DOOR: lambda last_state, state: (state, False) if random.random() < SUCCESS_CHANCE \
		else (last_state, False),
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
		self.broken = False
		# self.goal = self.rand_state()
		return self.get_obs()

	def step(self, action):
		state = self.state 
		if self.broken: 
			# assert False
			return self.get_obs(), self.compute_reward(state, self.goal), False, {}
		else:
			proposed_next_state = state + action
			next_state_type = self.grid[tuple(proposed_next_state)]
			next_state, broken = transitions[next_state_type](state, proposed_next_state)

			self.broken = broken
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
		assert False, "Not updated yet!"
		self.state = state[0]
		self.broken = state[1]

	def get_state(self): 
		return np.append(self.state, 1 if self.broken else 0)

	def get_obs(self):
		return {
			"state": self.get_state(),
			"observation": self.get_state(),
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


	gridworld.grid[1,size//2] = BREAKING_DOOR
	gridworld.grid[size-2, size//2] = EMPTY

	return gridworld
