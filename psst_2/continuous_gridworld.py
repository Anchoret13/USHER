
import gym 						#type: ignore
from gym.core import GoalEnv	#type: ignore
from gym import error			#type: ignore
from gym.spaces import Box		#type: ignore
from gym import spaces


import numpy as np 				#type: ignore
from numpy.linalg import norm
import random
import typing
import pdb
# import constants
from constants import *
# from obstacles
import math


noise_samples = [(1,0), (-1, 0), (0, 1), (0,1)]

ADD_ZERO = True

DISPLAY = True
# DISPLAY = False

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
NONBREAKING_FAILURE_CHANCE = .6
HIGH_FAILURE_CHANCE = .7#.3
LOW_FAILURE_CHANCE = HIGH_FAILURE_CHANCE/3

break_chance = 0.0#.6
# BREAKING = False
BREAKING = True
obstacle_density = 0#.2#15

transitions = { 
	EMPTY: lambda last_state, state, dt=1: (state, False),			#Just move
	# BLOCK: lambda last_state, state: (last_state, False),	#Prevent agent from moving
	# BLOCK: lambda last_state, state: (last_state, True),	#Prevent agent from moving
	BLOCK: lambda last_state, state, dt=1: (last_state, True if random.random() < break_chance*dt else False),	#Prevent agent from moving
	WIND:  lambda last_state, state, dt=1: (state + state_noise(4), False),
	BREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > HIGH_FAILURE_CHANCE*dt \
		else (last_state, BREAKING),
	LOWCHANCE_BREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > LOW_FAILURE_CHANCE*dt \
		else (last_state, BREAKING),
	NONBREAKING_DOOR: lambda last_state, state, dt=1: (state, False) if random.random() > NONBREAKING_FAILURE_CHANCE*dt \
		else (last_state, False),
}


stopping = { 
	EMPTY: lambda dt=1: (False, False),			#Just move
	# BLOCK: lambda last_state, state: (last_state, False),	#Prevent agent from moving
	# BLOCK: lambda last_state, state: (last_state, True),	#Prevent agent from moving
	BLOCK: lambda dt=1: (True, True) if random.random() < break_chance*dt else (True, False),	#Prevent agent from moving
	BREAKING_DOOR: lambda dt=1: (False, False) if random.random() > HIGH_FAILURE_CHANCE*dt \
		else (True, BREAKING),
	LOWCHANCE_BREAKING_DOOR: lambda dt=1: (False, False) if random.random() > LOW_FAILURE_CHANCE*dt \
		else (True, BREAKING),
	NONBREAKING_DOOR: lambda dt=1: (False, False) if random.random() > NONBREAKING_FAILURE_CHANCE*dt \
		else (True, False),
}

# is_unblocked = { 
# 	EMPTY: lambda : True,			
# 	BLOCK: lambda : False,
# 	WIND:  lambda : True,
# 	RANDOM_DOOR: lambda : True if random.random() < SUCCESS_CHANCE else False
# }

def state_noise(k):
	return random.sample(noise_samples + [(0,0)]*k, 1)

def state_normalize(s, size):
	return s*2/size - 1

def rotate(s, theta):
	rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	return rot@s


# Version that the positive results were gathered with
class OriginalGridworldEnv(GoalEnv):
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
		return self.get_obs(), reward, False, {"is_success": reward == 1}

		# return rv


	def compute_reward(self, ag, dg, info=None):
		return (ag.astype(int) == dg.astype(int)).all(axis=-1) - 0#1

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


class OldGridworldEnv(GoalEnv):
	#Most recent version
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
		next_state_type = self.grid[tuple(self.state_to_goal(proposed_next_state).astype(int))]
		next_state, broken = transitions[next_state_type](state, proposed_next_state)

		recover_chance = 0
		if broken: 
			self.broken = True
		if self.broken:
			next_state = state.copy()
			if np.random.rand() < recover_chance: 
				self.broken = False


		assert (np.abs(state - next_state) < 1.01).all()

		self.state = next_state.copy()
		reward = self.compute_reward(self.state_to_goal(next_state), self.goal)
		assert type(reward) == int or type(reward) == np.int64
		return self.get_obs(), reward, False, {"is_success": reward == 0}

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
		vals = [-1, 0, 1]
		moves = [np.array([i,j]) for i in vals for j in vals if (i,j) != (0,0)]
		# moves = [np.array([i,j]) for i in vals for j in vals if abs(i) + abs(j) == 1]
		# moves = [move/np.abs(move).sum() for move in moves]
		moves = moves + [move/2 for move in moves] #+ [move/4 for move in moves]
		# moves = [move/len(moves) for move in moves]
		# moves = [move*.1 for move in moves]
		# moves = []
		state = self.get_state()
		# surroundings  = [self.grid[tuple((state[:len(move)] + move).astype(int))] for move in moves]
		surroundings  = [self.grid[tuple((state[:len(move)] + move).astype(int))] == BLOCK for move in moves]
		# surroundings += [self.grid[tuple((state[:len(move)] + move).astype(int))] == BREAKING_DOOR for move in moves]
		surroundings = [x*.1 for x in surroundings]
		# surroundings = []
		# next_state_type = self.grid[tuple(proposed_next_state.astype(int))]
		return {
			"state": state,
			# "observation": state,
			"observation": np.append(state, surroundings),
			"achieved_goal": self.get_goal(),
			"desired_goal": self.goal,
		}


rand_displacement = lambda c : (1-c)/2 + c*np.random.rand(2)

class SimpleDynamicsEnv:#(gym.GoalEnv):
	def __init__(self, size):
		self.action_dim = 2
		self.state_dim = 2
		self.goal_dim = 2

		self.size = size
		self.obs_low = np.array([0, 0])
		self.obs_high = np.array([self.size - 1, self.size-1])
		self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype='float32')

	def dynamics(self, state, action, dt):
		l1_norm = np.abs(action).sum()
		if l1_norm > 1: 
			action = action/(l1_norm + .0001)
		return (state + action*dt)
		# pos =  (state['pos'] + action*dt)
		# return {"pos": pos, "rot": 0}

	def reset(self, initial_position): 
		return initial_position
		# return {"pos": initial_position, "rot": 0}

	def state_to_obs(self, state) -> np.ndarray:
		return state
		# return state['pos']

	def state_to_goal(self, state) -> np.ndarray:
		return state
		# return state['pos']

	def stop(self, proposed_invalid_state, prev_state): return prev_state


class AsteroidsDynamicsEnv:#(gym.GoalEnv):
	def __init__(self, size):
		self.action_dim = 2
		self.state_dim = 5
		self.goal_dim = 2

		self.acc_speed = 2
		self.rot_speed = 5

		self.size = size
		self.translation_speed= 1#1##self.size/2
		self.obs_low = np.array([0, 0, -1, -1, -1])
		self.obs_high = np.array([self.size - 1, self.size-1, 1, 1, 1])
		self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype='float32')

	def dynamics(self, state, action, dt):
		action = np.clip(action, -1, 1)

		new_rotation = (state['rot'] + action[1]*dt*self.rot_speed)%(2*math.pi)
		# new_rotation = action[1]*(2*math.pi)
		new_acceleration = np.array([action[0]*math.cos(new_rotation), action[0]*math.sin(new_rotation)])
		new_velocity = state['vel']*(1-self.acc_speed*dt) + new_acceleration*self.acc_speed*dt
		# new_velocity = new_acceleration

		# norm = np.linalg.norm(new_velocity, p=2)
		norm = np.linalg.norm(new_velocity, ord=2)
		new_velocity = new_velocity if norm <= 1 else new_velocity/(norm + .0001)
		new_position = state['pos'] + new_velocity*dt*self.translation_speed
		assert ((new_position - state['pos'])**2).sum()**.5 <= self.translation_speed*1.0001

		new_state= {
				'pos': new_position, 
				'vel': new_velocity, 
				'rot': new_rotation
			}
		return new_state

	def reset(self, initial_position):
		# state= {'pos': initial_position, 
		# 		'vel': np.random.rand(2)*2 - 1, #np.zeros(2),
		# 		'rot': np.random.rand()*2*math.pi}
		state= {'pos': initial_position, 
				'vel': np.zeros(2),
				'rot': np.random.rand()*2*math.pi}
		return state

	def state_to_obs(self, state) -> np.ndarray:
		# return np.concatenate([state['pos'], state['vel'], np.array([state['rot']])/math.pi - 1])
		return np.concatenate([state_normalize(state['pos'], self.size), state['vel'], np.array([math.cos(state['rot']), math.sin(state['rot'])])])

	def state_to_goal(self, state) -> np.ndarray:
		return state['pos']

	def stop(self, proposed_invalid_state, prev_state):
		new_state ={'pos': prev_state['pos'],
					# 'vel': np.zeros(2), 
					'vel': prev_state['pos'], 
					'rot': proposed_invalid_state['rot']}
					#allow it to turn when it's run up against a wall, rather than just sticking there
		return new_state


class CarDynamicsEnv(AsteroidsDynamicsEnv):
	def __init__(self, size):
		super().__init__(size)
		self.rot_speed=10

	def dynamics(self, state, action, dt):
		turn = action[1]
		heading = np.array([math.cos(state['rot']), math.sin(state['rot'])])
		new_rotation = (state['rot'] + norm(state['vel'])*turn*dt*self.rot_speed)%(2*math.pi)
		new_acceleration = action[0]*heading
		# new_velocity = state['vel']*(1-self.acc_speed*dt) + new_acceleration*self.acc_speed*dt
		# new_velocity = (new_velocity@heading)*heading
		new_velocity = new_acceleration*heading
		# new_velocity = np.clip(new_velocity, -1, 1)
		vel_norm = norm(new_velocity, ord=2)
		new_velocity = new_velocity if vel_norm <= 1 else new_velocity/(vel_norm + .0001)
		new_position = state['pos'] + new_velocity*dt*self.translation_speed
		assert ((new_position - state['pos'])**2).sum()**.5 <= self.translation_speed*1.00001

		new_state= {
				'pos': new_position, 
				'vel': new_velocity, 
				'rot': new_rotation
			}
		return new_state

class GridworldEnv(GoalEnv):
	def __init__(self, size, grid, randomize_start, dynamics, start=None, goal=None):
		self.dim = 2
		self.size = size
		self.start = start
		self.grid = grid
		self.env = dynamics

		if ADD_ZERO: 
			self.obs_scope = (size, size, 2)
		else:
			self.obs_scope = (size, size)

		self.goal_scope = (size, size)
		self.action_space = Box(np.array([-1,-1]), np.array([1,1]))

		self.reward_range = (0,1)
		self.steps = 1
		self.high_speed_pretraining = True
		# self.env = SimpleDynamicsEnv(size, start)
		self.pretrain_iters = 0#250
		# self.env.translation_speed = 1#self.size/10
		self.base_dt = 1#.5#.25


		self.randomize_start = randomize_start
		self.start = start
		self.goal = goal
		self.size = self.grid.shape[0] - 1
		self.observation_space = spaces.Dict(dict(
		    desired_goal	=spaces.Box(0, self.size, shape= (2,), dtype='float32'),
		    achieved_goal	=spaces.Box(0, self.size, shape= (2,), dtype='float32'),
		    observation 	=self.env.observation_space,
		))

	def reset_start_and_goal(self):
		if self.randomize_start:
			self.start = self.observation_space['desired_goal'].sample()
			while self.grid[tuple(self.start.astype(int))] != EMPTY:
				self.start = self.observation_space['desired_goal'].sample()
			self.goal = self.observation_space['desired_goal'].sample()
			while self.grid[tuple(self.start.astype(int))] != EMPTY:
				self.goal = self.observation_space['desired_goal'].sample()

	def reset(self):
		self.reset_start_and_goal()
		self.state = self.env.reset(self.start)
		# self.goal = self.new_goal + rand_displacement(1)
		self.broken = False
		self.size = self.grid.shape[0]
		if self.pretrain_iters > 0:
			self.pretrain_iters -= 1
			if self.pretrain_iters == 0:
				self.high_speed_pretraining = False
				# self.env.translation_speed = 1#/2
				if DISPLAY: print("Pre-training finished")
		# else:
		# 	self.env.translation_speed = 1
		return self.get_obs()

	def step(self, action):
		action = np.clip(action, -1, 1)
		began_broken = self.broken
		state = self.state
		last_state = self.state.copy()
		proposed_next_state = state
		next_state = state
		broken = False
		dt = 1/self.steps*self.base_dt
		for _ in range(self.steps):
			proposed_next_state = self.env.dynamics(next_state, action, dt)
			proposed_ag = np.clip(self.env.state_to_goal(proposed_next_state), .001, self.size - 1.001)
			next_state_type = self.grid[tuple(proposed_ag.astype(int))]
			next_state, next_broken = transitions[next_state_type](state, proposed_next_state, dt)
			broken = broken or next_broken
			if broken: break

			# proposed_next_state = self.env.dynamics(next_state, action, dt)
			# proposed_ag = np.clip(self.env.state_to_goal(proposed_next_state), .001, self.size - 1.001)
			# next_state_type = self.grid[tuple(proposed_ag.astype(int))]
			# next_stopped, next_broken = stopping[next_state_type](dt)
			# assert next_stopped or not next_broken
			# if next_stopped or next_broken: 
			# 	next_state = self.env.stop(proposed_next_state, next_state)
			# else: 
			# 	next_state = proposed_next_state
			# broken = broken or next_broken
			# if broken: break

		# broken = False
		# next_state = proposed_next_state if (proposed_ag <= self.size-1).all() and (proposed_ag >= 0).all() else state

		recover_chance = 0
		if broken: 
			self.broken = True
		if self.broken:
			next_state = state.copy()
			if np.random.rand() < recover_chance: 
				self.broken = False

		self.state = next_state.copy()
		reward = self.compute_reward(state_normalize(self.env.state_to_goal(next_state), self.size), state_normalize(self.goal, self.size))
		assert type(reward) == int or type(reward) == np.int64
		observation = self.get_obs()
		# if self.high_speed_pretraining: self.broken = False
		return observation, reward, False, {"is_success": reward == 0}

	# def compute_reward(self, ag, dg, info=None):
	# 	return (ag.astype(int) == dg.astype(int)).all(axis=-1) - 1
	def compute_reward(self, ag, dg, info=None):
		true_threshold = 2**(-.5)#1#2
		threshold = 2/self.size*true_threshold
		reward = (((ag - dg)**2).sum(axis=-1) < threshold) - 1
		# if len(ag.shape) > 1: pdb.set_trace()
		return reward

		# return 1  else 0

	def get_state_obs(self): 
		rv = np.append(self.env.state_to_obs(self.state), self.broken)		
		return rv

	def get_goal(self): 
		return self.env.state_to_goal(self.state)

	def get_obs(self):
		vals = [-1, 0, 1]
		moves = [np.array([i,j]) for i in vals for j in vals if (i,j) != (0,0)]
		try: 
			moves = [rotate(move, state['rot']) for move in moves]
		except: 
			pass
		# n_angles = 16
		# try:  rot = state['rot']
		# else: rot = 0
		# moves = [rotate(np.array([1,0]), state['rot'] + i/(2*math.pi)) for i in range(n_angles)]
		moves = moves + [move*2 for move in moves]  + [move/2 for move in moves] + [np.zeros(2)]

		state_obs = self.get_state_obs()
		ag = self.env.state_to_goal(self.state)
		def fail_chance(move):
			blocktype = self.grid[tuple((np.clip(ag + move, .01, self.size-1.01)).astype(int))]
			if blocktype == BLOCK: 
				chance = 1
			elif blocktype == BREAKING_DOOR:
				chance = HIGH_FAILURE_CHANCE
			elif blocktype == LOWCHANCE_BREAKING_DOOR:
				chance = LOW_FAILURE_CHANCE
			elif blocktype == NONBREAKING_DOOR:
				chance = NONBREAKING_FAILURE_CHANCE
			elif blocktype == EMPTY:
				chance = 0
			return chance


		surroundings  = [fail_chance(move) == BLOCK for move in moves]
		# surroundings  = [self.grid[tuple((np.clip(ag + move, .01, self.size-1.01)).astype(int))] == BREAKING_DOOR for move in moves]
		# surroundings  = [self.grid[tuple((np.clip(ag + move, .01, self.size-1.01)).astype(int))] == LOWCHANCE_BREAKING_DOOR for move in moves]
		# surroundings = [x/len(surroundings) for x in surroundings]
		# surroundings = []
		# next_state_type = self.grid[tuple(proposed_next_state.astype(int))]
		return {
			"state": state_obs,
			# "observation": state,
			"observation": np.append(state_obs, surroundings),
			"achieved_goal": state_normalize(self.get_goal(), self.size),
			"desired_goal": state_normalize(self.goal, self.size)
		}


class RandomResetGridworldEnv(GridworldEnv):
	def __init__(self, size, grid, randomize_start, dynamics, start=None, goal=None):
		super().__init__(size, grid, randomize_start, dynamics, start, goal)
		self.reset_grid = reset_grid

	def reset(self):
		self.grid = self.reset_grid()
		return super().reset()

# class SimpleDynamicsGridworldEnv(GridworldEnv):
# 	def __init__(self, size, grid, randomize_start, start=None, goal=None):
# 		super().__init__(size, grid, randomize_start, start, goal)
# 		self.env = SimpleDynamicsEnv(size, start)

# class AsteroidsGridworldEnv(GridworldEnv):
# 	def __init__(self, size, grid, randomize_start, start=None, goal=None):
# 		super().__init__(size, grid, randomize_start, start, goal)
# 		self.env = AsteroidsDynamicsEnv(size, start)

# class CarGridworldEnv(GridworldEnv):
# 	def __init__(self, size, grid, randomize_start, start=None, goal=None):
# 		super().__init__(size, grid, randomize_start, CarDynamicsEnv(size, start) ,start, goal)

def generate_random_map(size):
	if DISPLAY: print("Generating random map")
	size = size
	mid = size//2
	offset = 2
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	grid = np.zeros((size, size))

	for i in range(size):
		block_chance = obstacle_density
		for j in range(size):
			# if np.random.rand() < block_chance:
			# 	# grid[i, j] = BLOCK
			# 	grid[i, j] = LOWCHANCE_BREAKING_DOOR
			if np.random.rand() < block_chance:
				grid[i, j] = BREAKING_DOOR
				# grid[i, j] = NONBREAKING_DOOR
				# grid[i, j] = LOWCHANCE_BREAKING_DOOR
			if np.random.rand() < block_chance:
				grid[i, j] = LOWCHANCE_BREAKING_DOOR
				# grid[i, j] = NONBREAKING_DOOR

	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

	adj = [-1, 0, 1]
	# for i in adj:
	# 	for j in adj:
	# 		grid[start_pos + i, start_pos + j] = EMPTY
	# 		grid[goal_pos + i, goal_pos + j] = EMPTY
	if DISPLAY: print(grid)
	return grid



def generate_blocky_random_map(size):
	if DISPLAY: print("Generating blocky random map")
	size = size
	mid = size//2
	offset = 2
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	grid = np.zeros((size, size))

	mean_length = 2
	num_squares = (size-1)**2
	block_fraction = obstacle_density

	def assign_blocks(block_type, block_fraction=block_fraction):
		for _ in range(int(num_squares*block_fraction/mean_length)):
			loc = np.random.randint(offset, size-offset, size=2).squeeze()
			# pdb.set_trace()
			grid[tuple(loc)] = block_type
			while not np.random.rand() < 1/(mean_length+1):
				step_loc = lambda loc : (loc + np.array(random.sample(noise_samples, 1)).squeeze())%size
				# step_loc = lambda loc : (loc + np.array(random.sample(noise_samples, 1)))%size
				# step_loc = lambda loc: random.sample([(loc[0]+1, loc[1]),(loc[0]-1, loc[1]),(loc[0], loc[1]+1),(loc[0], loc[1]),-1], 1)
				loc = step_loc(loc)
				grid[tuple(loc)] = block_type
	# assign_blocks(LOWCHANCE_BREAKING_DOOR)
	# assign_blocks(NONBREAKING_DOOR, block_fraction)
	assign_blocks(BREAKING_DOOR, block_fraction)
	assign_blocks(LOWCHANCE_BREAKING_DOOR)
	# assign_blocks(BLOCK, block_fraction=block_fraction/2)

	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

	# adj = [-1, 0, 1]
	# for i in adj:
	# 	for j in adj:
	# 		grid[start_pos + i, start_pos + j] = EMPTY
	# 		grid[goal_pos + i, goal_pos + j] = EMPTY

	if DISPLAY: print(grid)

	return grid

def random_map_environment():
	size = 14
	mid = size//2
	offset = 3
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)

	return RandomResetGridworldEnv(size, start, goal, lambda : generate_random_map(size))

def create_map_1_grid(size, block_start=False):
	grid = np.zeros((size, size))
	mid = size//2
	if block_start:
		grid[tuple(start)] = BLOCK
		grid[tuple(start + np.array([1, 1]))] = BLOCK
	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[size-2,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

		#Wall through the middle
		grid[i,mid ] = BLOCK

	grid[1,mid] = BREAKING_DOOR
	
	if not BLOCK_ALT_PATH:
		grid[size-3,mid] = EMPTY

	if DISPLAY: print(grid)
	return grid


def create_test_map_grid(size, block_start=False):
	grid = np.zeros((size, size))
	mid = size//2
	if block_start:
		grid[tuple(start)] = BLOCK
		grid[tuple(start + np.array([1, 1]))] = BLOCK
	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[size-2,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

		#Wall through the middle
		grid[i,mid ] = BLOCK

	grid[1,mid] = BREAKING_DOOR
	
	if not BLOCK_ALT_PATH:
		grid[size-3,mid] = EMPTY

	if DISPLAY: print(grid)
	return grid



def two_door_environment(block_start=False):
	size = 5
	mid = 2
	start  = np.array([mid,mid -1])
	new_goal  = np.array([mid, mid +1])
	gridworld = OldGridworldEnv(size, start, new_goal)
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
	


def get_class_constructor(env_type, size, grid, randomize_start, start, goal):
	if env_type == "linear":
		env = GridworldEnv(size, grid, randomize_start, SimpleDynamicsEnv(size), start, goal)
	elif env_type == "asteroids": 
		env = GridworldEnv(size, grid, randomize_start, AsteroidsDynamicsEnv(size), start, goal)
	elif env_type == "car": 
		env = GridworldEnv(size, grid, randomize_start, CarDynamicsEnv(size), start, goal)
	else: 
		print(f"No dynamics environment matches name {env_type}")
		raise Exception
	return env

def random_map(env_type="linear"): 
	size = 6
	grid = generate_random_map(size)
	mid = size//2
	offset = 3
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	randomize_start = True
	# env = get_class_constructor(env_type)(size, grid, randomize_start, start, goal)
	env = get_class_constructor(env_type, size, grid, randomize_start, start, goal)
	# env.set_grid(grid, True, start=start, goal=goal)
	# env.env.rot_speed *= 2
	# env.env.translation_speed = size
	# env.steps = size*5
	return env

def random_blocky_map(env_type="linear"): 
	size = 8
	grid = generate_blocky_random_map(size)
	mid = size//2
	offset = 3
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	randomize_start = True
	env = get_class_constructor(env_type, size, grid, randomize_start, start, goal)
	return env


def create_map_1(env_type="linear", block_start=False):
	size = 6
	grid = create_map_1_grid(size, block_start)
	mid = size//2
	start  = np.array([1,mid -2])
	goal  = np.array([1, mid +1])
	randomize_start = False
	env = get_class_constructor(env_type, size, grid, randomize_start, start, goal)
	env = OriginalGridworldEnv(size, start, goal)
	env.grid = grid
	return env




def create_test_map(env_type="linear", block_start=False):
	size = 6
	grid = create_test_map_grid(size, block_start)
	mid = size//2
	start  = np.array([1,mid -2])
	goal  = np.array([1, mid +1])
	randomize_start = False
	env = get_class_constructor(env_type, size, grid, randomize_start, start, goal)
	# env = OriginalGridworldEnv(size, start, goal)
	# env.grid = grid
	return env

# def random_asteroids_map():

