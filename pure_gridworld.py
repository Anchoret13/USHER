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

SUCCESS_CHANCE = .1
ADD_ZERO = True

BLOCK_ALT_PATH = False
# BLOCK_ALT_PATH = True

# transitions = { 
# 	EMPTY: lambda last_state, state: state,			#Just move
# 	BLOCK: lambda last_state, state: last_state,	#Prevent agent from moving
# 	WIND:  lambda last_state, state: state + state_noise(4),
# 			#Currently does not work because state may be blocked
# 			#To fix later
# 	RANDOM_DOOR: lambda last_state, state: state if random.random() < SUCCESS_CHANCE else last_state,
# }

transitions = { 
	EMPTY: lambda last_state, state: (state, False),			#Just move
	BLOCK: lambda last_state, state: (last_state, False),	#Prevent agent from moving
	WIND:  lambda last_state, state: (state + state_noise(4), False),
			#Currently does not work because state may be blocked
			#To fix later
	RANDOM_DOOR: lambda last_state, state: (state, False) if random.random() < SUCCESS_CHANCE \
		else (last_state, True),
}



# is_unblocked = { 
# 	EMPTY: lambda : True,			
# 	BLOCK: lambda : False,
# 	WIND:  lambda : True,
# 	RANDOM_DOOR: lambda : True if random.random() < SUCCESS_CHANCE else False
# }

def state_noise(k):
	return random.sample(noise_samples + [(0,0)]*k)

class GridworldEnv(GoalEnv):
	def __init__(self, size, start, new_goal):
		self.dim = 2
		self.size = size
		self.start = start
		self.new_goal = new_goal
		self.grid = np.zeros((size, size))

		if ADD_ZERO: 
			self.obs_scope = (size, size, 2)
		else:
			self.obs_scope = (size, size)

		self.goal_scope = (size, size)

		self.reward_range = (0,1)

	def reset(self):
		# self.state = self.start()
		# self.goal = self.new_goal()
		self.state = self.start
		self.goal = self.new_goal
		self.broken = False
		# self.goal = self.rand_state()
		obs = {
			"state": np.append(self.state, 0),
			"observation": np.append(self.state, 0),
			"achieved_goal": self.state,
			"desired_goal": self.goal,
		}
		return obs

	def step(self, input_state, action):
		assert input_state.shape == (3,)
		assert input_state[-1] == 1 or input_state[-1] == 0
		state = input_state[:2]
		was_broken = True if input_state[-1] == 1 else False
		if was_broken: 	
			obs = {
				"state": input_state,
				"observation": input_state,
				"achieved_goal": state,
				"desired_goal": self.goal,
			}
			reward = self.compute_reward(state, self.goal)
		else:

			proposed_next_state = state + action
			next_state_type = self.grid[tuple(proposed_next_state)]
			next_state, broken = transitions[next_state_type](state, proposed_next_state)

			if broken:
				ret_next_state = input_state
			else:
				ret_next_state = np.append(state, broken)

			obs = {
				"state": ret_next_state,
				"observation": ret_next_state,
				"achieved_goal": ret_next_state[:2],
				"desired_goal": self.goal,
			}

			reward = self.compute_reward(ret_next_state[:2], self.goal)
		
		return obs, reward, False, {}

	def compute_reward(self, ag, dg):
		return 1 if (ag == dg).all() else 0

	def rand_state(self):
		return np.array([np.random.randint(0, size), np.random.randint(0, size)])

	def set_state(self, state): 
		self.state = state

	def get_state(self): 
		if ADD_ZERO:
			return np.append(self.state, self.broken)
		else: 
			return self.state

	def get_goal(self): 
		return self.state

	# def get_obs(self):
	# 	return {
	# 		"state": self.get_state(),
	# 		"observation": self.get_state(),
	# 		"achieved_goal": self.get_goal(),
	# 		"desired_goal": self.goal,
	# 	}

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
	
	if not BLOCK_ALT_PATH:
		#Hole in the right side of the wall
		gridworld.grid[size-2, size//2] = EMPTY


	return gridworld
