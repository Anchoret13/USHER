import gym 						#type: ignore
from gym.core import GoalEnv	#type: ignore
from gym import error			#type: ignore

import numpy as np 				#type: ignore
import random
import typing 
from typing import Tuple
import pdb
# import constants
from constants import *
# from obstacles


noise_samples = [(1,0), (-1, 0), (0, 1), (0,1)]

def state_noise(k: int) -> State:
	return random.sample(noise_samples + [(0,0)]*k, 1)[0]



transitions = { 
	EMPTY: lambda last_state, state: state,			#Just move
	BLOCK: lambda last_state, state: last_state,	#Prevent agent from moving
	WIND:  lambda last_state, state: state + state_noise(4),
			#Currently does not work because state may be blocked
			#To fix later
	RANDOM_DOOR: lambda last_state, state: state if random.random() < 0.1 else last_state,
}




class GridworldEnv(GoalEnv):
	def __init__(self, size: int, start: State, new_goal: Goal):
		self.size = size
		self.start = start
		self.new_goal = new_goal
		self.grid = np.zeros((size, size))

		self.reward_range = (0,1)

	def reset(self) -> ObservationDict:
		return self.get_obs(self.start, self.new_goal)

	def step(self, state: State, action: Action) -> Tuple[ObservationDict, float, bool, dict]:
		proposed_next_state = add(state, action)
		next_state_type = self.grid[proposed_next_state]
		next_state = transitions[next_state_type](state, proposed_next_state)

		# print(next_state_type)
		# pdb.set_trace()

		# if np.abs(sub(state, next_state)).sum() > 1.05:
		# 	pdb.set_trace()

		reward = self.compute_reward(next_state, self.new_goal)
		return self.get_obs(next_state, self.new_goal), self.compute_reward(next_state, self.new_goal), False, {}

	def compute_reward(self, ag: Goal, dg: Goal) -> float:
		return 1 if all([a == d for (a, d) in zip(ag, dg)]) else 0

	def rand_state(self) -> State:
		return (np.random.randint(0, self.size), np.random.randint(0, self.size))

	def get_obs(self, state: State, goal: Goal) -> ObservationDict:
		return {
			"state": state,
			"observation": state,
			"achieved_goal": state,
			"desired_goal": goal,
		}

def create_map_1():
	size = 8
	start  = tuple([1,size//2 -1])
	new_goal  = tuple([1, size//2 +1])
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
