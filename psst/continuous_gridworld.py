import gym 						#type: ignore
from gym.core import GoalEnv	#type: ignore
from gym import error			#type: ignore
from gym.spaces import Box		#type: ignore


import numpy as np 				#type: ignore
import random
import typing
import pdb
# import constants
from constants import *
# from obstacles


noise_samples = [(1,0), (-1, 0), (0, 1), (0,1)]

ADD_ZERO = True


force_long_road = False
force_short_road = False
assert not (force_long_road and force_short_road)
if force_long_road:
	BLOCK_ALT_PATH = False
	SUCCESS_CHANCE = 0
else: 
	if force_short_road:
		BLOCK_ALT_PATH = True
	else: 
		BLOCK_ALT_PATH = False
	# SUCCESS_CHANCE = .5
	SUCCESS_CHANCE = .25
	# SUCCESS_CHANCE = .15
	# SUCCESS_CHANCE = .1

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
	BREAKING_DOOR: lambda last_state, state: (state, False) if random.random() < SUCCESS_CHANCE \
		else (last_state, True),
	NONBREAKING_DOOR: lambda last_state, state: (state, False) if random.random() < SUCCESS_CHANCE \
		else (last_state, False),
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
		self.action_space = Box(np.array([-1,-1]), np.array([1,1]))

		self.reward_range = (0,1)

	def reset(self):
		c = 0
		# self.state = self.start()
		# self.goal = self.new_goal()
		rand_displacement = lambda c : (1-c)/2 + c*np.random.rand(2)
		assert (rand_displacement(c) > c/2).all() and (rand_displacement(c) < (1-c/2)).all()
		self.state = self.start + rand_displacement(c)
		self.goal = self.new_goal + rand_displacement(1)
		self.broken = False
		# self.goal = self.rand_state()
		return self.get_obs()

	def dynamics(self, state, action):
		return (state + action)

	def state_to_goal(self, state):
		return state

	def step(self, action):
		# action = action/np.linalg.norm(action)
		l1_norm = np.abs(action).sum()
		if l1_norm > 1: 
			action = action/l1_norm
		began_broken = self.broken
		state = self.state
		last_state = self.state.copy()
		proposed_next_state = self.dynamics(state, action)
		next_state_type = self.grid[tuple(proposed_next_state.astype(int))]
		next_state, broken = transitions[next_state_type](state, proposed_next_state)

		# if is_unblocked[next_state_type](): and not self.broken:
		# 	next_state = proposed_next_state
		# else: 
		# 	next_state = state
		# 	self.broken = True
		if broken: 
			self.broken = True
			# next_state = self.start
		if self.broken:
			next_state = state.copy()
		# next_state = transitions[next_state_type](state, proposed_next_state)

		# assert np.abs(state - next_state).sum() < 1.01
		assert (np.abs(state - next_state) < 1.01).all()

		# reward = self.compute_reward(next_state, self.goal)
		self.state = next_state.copy()
		# obs = self.get_obs()
		reward = self.compute_reward(self.state_to_goal(next_state), self.goal)
		assert type(reward) == int or type(reward) == np.int64
		return self.get_obs(), reward, False, {"is_success": reward == 0}

		# return rv


	def compute_reward(self, ag, dg, info=None):
		return (ag.astype(int) == dg.astype(int)).all(axis=-1) - 1

		# return 1  else 0

	def rand_state(self):
		return np.array([np.random.randint(0, size), np.random.randint(0, size)])

	def set_state(self, state): 
		self.state = state

	def get_state(self): 
		rv = np.append(self.state, self.broken)
		
		return rv

	def get_goal(self): 
		return self.state

	def get_obs(self):
		return {
			"state": self.get_state(),
			"observation": self.get_state(),
			"achieved_goal": self.get_goal(),
			"desired_goal": self.goal,
		}


class RotationGridworldEnv(GridworldEnv):
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
		self.action_space = Box(np.array([-1,-1]), np.array([1,1]))

		self.reward_range = (0,1)

		self.velocity = np.zeros(2)
		self.rotation = 0


	def reset(self):
		assert False #not reimplemented yet
		c = 0
		# self.state = self.start()
		# self.goal = self.new_goal()
		rand_displacement = lambda c : (1-c)/2 + c*np.random.rand(2)
		assert (rand_displacement(c) > c/2).all() and (rand_displacement(c) < (1-c/2)).all()
		self.state = self.start + rand_displacement(c)
		self.goal = self.new_goal + rand_displacement(1)
		self.broken = False
		# self.goal = self.rand_state()
		return self.get_obs()

	def dynamics(self, state, action):
		assert False #not reimplemented yet
		return (state + action)

	def state_to_goal(self, state):
		assert False #not reimplemented yet
		return state

	def step(self, action):
		forward_acc = action[0]
		assert False #not reimplemented yet
		l1_norm = np.abs(action).sum()
		if l1_norm > 1: 
			action = action/l1_norm
		began_broken = self.broken
		state = self.state
		last_state = self.state.copy()
		proposed_next_state = self.dynamics(state, action)
		next_state_type = self.grid[tuple(proposed_next_state.astype(int))]
		next_state, broken = transitions[next_state_type](state, proposed_next_state)

		if broken: 
			self.broken = True
			# next_state = self.start
		if self.broken:
			next_state = state.copy()
		# next_state = transitions[next_state_type](state, proposed_next_state)

		# assert np.abs(state - next_state).sum() < 1.01
		assert (np.abs(state - next_state) < 1.01).all()

		# reward = self.compute_reward(next_state, self.goal)
		self.state = next_state.copy()
		# obs = self.get_obs()
		reward = self.compute_reward(self.state_to_goal(next_state), self.goal)
		assert type(reward) == int or type(reward) == np.int64
		return self.get_obs(), reward, False, {"is_success": reward == 0}

		# return rv


	def compute_reward(self, ag, dg, info=None):
		return (ag.astype(int) == dg.astype(int)).all(axis=-1) - 1

		# return 1  else 0

	def rand_state(self):
		return np.array([np.random.randint(0, size), np.random.randint(0, size)])

	def get_state(self): 
		rv = np.append(self.state, self.broken, self.velocity, self.rotation)
		
		return rv

	def get_goal(self): 
		return self.state

	def get_obs(self):
		return {
			"state": self.get_state(),
			"observation": self.get_state(),
			"achieved_goal": self.get_goal(),
			"desired_goal": self.goal,
		}

# def create_map_1():
# 	size = 8
# 	start  = np.array([1,size//2 -1])
# 	new_goal  = np.array([1, size//2 +1])
# 	gridworld = GridworldEnv(size, start, new_goal)
# 	for i in range(size):
# 		#Borders
# 		gridworld.grid[0,i] = BLOCK
# 		gridworld.grid[size-1,i] = BLOCK
# 		gridworld.grid[i,0] = BLOCK
# 		gridworld.grid[i, size-1] = BLOCK

# 		#Wall through the middle
# 		gridworld.grid[i,size//2 ] = BLOCK


# 	gridworld.grid[1,size//2] = RANDOM_DOOR
	
# 	if not BLOCK_ALT_PATH:
# 		#Hole in the right side of the wall
# 		gridworld.grid[size-2, size//2] = EMPTY


# 	return gridworld

def create_map_1(block_start=False):
	size = 6
	mid = size//2
	start  = np.array([1,mid -2])
	new_goal  = np.array([1, mid +1])
	gridworld = GridworldEnv(size, start, new_goal)
	if block_start:
		gridworld.grid[tuple(start)] = BLOCK
		gridworld.grid[tuple(start + np.array([1, 1]))] = BLOCK
	for i in range(size):
		#Borders
		gridworld.grid[0,i] = BLOCK
		gridworld.grid[size-1,i] = BLOCK
		gridworld.grid[size-2,i] = BLOCK
		gridworld.grid[i,0] = BLOCK
		gridworld.grid[i, size-1] = BLOCK

		#Wall through the middle
		gridworld.grid[i,mid ] = BLOCK

	gridworld.grid[1,mid] = BREAKING_DOOR
	
	if not BLOCK_ALT_PATH:
		#Hole in the right side of the wall
		# gridworld.grid[size-2,mid] = EMPTY
		gridworld.grid[size-3,mid] = EMPTY
		# gridworld.grid[size-4,mid] = EMPTY
		# gridworld.grid[size-5,mid] = EMPTY

	return gridworld



def two_door_environment(block_start=False):
	size = 5
	mid = 2
	start  = np.array([mid,mid -1])
	new_goal  = np.array([mid, mid +1])
	gridworld = GridworldEnv(size, start, new_goal)
	if block_start:
		gridworld.grid[tuple(start)] = BLOCK
	for i in range(size):
		#Borders
		gridworld.grid[0,i] = BLOCK
		gridworld.grid[size-1,i] = BLOCK
		gridworld.grid[i,0] = BLOCK
		gridworld.grid[i, size-1] = BLOCK

		#Wall through the middle
		gridworld.grid[i,mid] = BLOCK


	gridworld.grid[1,mid] = BREAKING_DOOR
	gridworld.grid[-2,mid] = NONBREAKING_DOOR
	


	return gridworld
