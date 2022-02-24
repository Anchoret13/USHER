
import gym 						#type: ignore
from gym.core import GoalEnv	#type: ignore
from gym import error			#type: ignore
from gym.spaces import Box		#type: ignore


import numpy as np 				#type: ignore
from numpy.linalg import norm
import random
import typing
import pdb
from gym import spaces
# import constants
from constants import *
# from obstacles
import math


noise_samples = [(1,0), (-1, 0), (0, 1), (0,1)]

ADD_ZERO = True

SUCCESS_CHANCE = .21
HIGH_SUCCESS_CHANCE = .8

break_chance = 0.0#.6

transitions = { 
	EMPTY: lambda last_state, state: (state, False),			#Just move
	BLOCK: lambda last_state, state: (last_state, True if random.random() < break_chance else False),	#Prevent agent from moving
	WIND:  lambda last_state, state: (state + state_noise(4), False),
	BREAKING_DOOR: lambda last_state, state: (state, False) if random.random() < SUCCESS_CHANCE \
		else (last_state, True),
	LOWCHANCE_BREAKING_DOOR: lambda last_state, state: (state, False) if random.random() < HIGH_SUCCESS_CHANCE \
		else (last_state, True),
	NONBREAKING_DOOR: lambda last_state, state: (state, False) if random.random() < SUCCESS_CHANCE \
		else (last_state, False),
}


def square_to_radian(square):
	return ((square + 1)*math.pi)%(2*math.pi)

def radian_to_square(radian):
	return radian%(2*math.pi)/math.pi - 1

def is_approx(a, b): 
	thresh= .000001
	return abs(a-b) < thresh

def state_normalize(s, size):	return s*2/size - 1

def state_noise(k):				return random.sample(noise_samples + [(0,0)]*k, 1)

def rand_displacement(c): 		return (1-c)/2 + c*np.random.rand(2)

class SimpleDynamicsEnv:#(gym.GoalEnv):
	def __init__(self, size, start):
		self.action_dim = 2
		self.state_dim = 2
		self.goal_dim = 2

		self.start = start
		self.size = size

	def dynamics(self, state, action, dt):
		l1_norm = np.abs(action).sum()
		if l1_norm > 1: 
			action = action/(l1_norm + .0001)
		return (state + action*dt)

	def reset(self): 
		c = 0
		return self.start + rand_displacement(c)

	def state_to_obs(self, state) -> np.ndarray:
		return state

	def state_to_goal(self, state) -> np.ndarray:
		return state


class AsteroidsDynamicsEnv:#(gym.GoalEnv):
	def __init__(self, size, start):
		self.action_dim = 2
		self.state_dim = 5
		self.goal_dim = 2

		self.acc_speed = 2
		self.rot_speed = 5

		self.start = start
		self.size = size

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
		new_position = state['pos'] + new_velocity*dt
		assert ((new_position - state['pos'])**2).sum() <= 1

		new_state= {
				'pos': new_position, 
				'vel': new_velocity, 
				'rot': new_rotation
			}
		return new_state

	def reset(self): 
		state= {'pos': self.start + rand_displacement(0), 
				'vel': np.zeros(2), 
				'rot': np.random.rand()*2*math.pi}
		return state

	def state_to_obs(self, state) -> np.ndarray:
		# return np.concatenate([state['pos'], state['vel'], np.array([state['rot']])/math.pi - 1])
		return np.concatenate([state_normalize(state['pos'], self.size), state['vel'], np.array([math.cos(state['rot']), math.sin(state['rot'])])])

	def state_to_goal(self, state) -> np.ndarray:
		return state['pos']


class CarDynamicsEnv(AsteroidsDynamicsEnv):
	def dynamics(self, state, action, dt):
		turn = action[1]
		heading = np.array([math.cos(state['rot']), math.sin(state['rot'])])
		new_rotation = (state['rot'] + norm(state['vel'])*turn*dt*self.rot_speed)%(2*math.pi)
		new_acceleration = action[0]*heading
		new_velocity = state['vel']*(1-self.acc_speed*dt) + new_acceleration*self.acc_speed*dt
		new_velocity = (new_velocity@heading)*heading
		# new_velocity = np.clip(new_velocity, -1, 1)
		vel_norm = norm(new_velocity, ord=2)
		new_velocity = new_velocity if vel_norm <= 1 else new_velocity/(vel_norm + .0001)
		new_position = state['pos'] + new_velocity*dt
		assert ((new_position - state['pos'])**2).sum() <= 1

		new_state= {
				'pos': new_position, 
				'vel': new_velocity, 
				'rot': new_rotation
			}
		return new_state

class GridworldEnv(GoalEnv):
	def __init__(self, size, start, new_goal, steps=1):
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
		self.steps = 1
		# self.env = SimpleDynamicsEnv(size, start)

	def reset(self):
		self.state = self.env.reset()
		self.goal = self.new_goal + rand_displacement(1)
		self.broken = False
		return self.get_obs()

	def step(self, action):
		action = np.clip(action, -1, 1)
		began_broken = self.broken
		state = self.state
		last_state = self.state.copy()
		proposed_next_state = state
		for _ in range(self.steps):
			proposed_next_state = self.env.dynamics(proposed_next_state, action, 1/self.steps)
		# next_state_type = self.grid[tuple(self.env.state_to_goal(proposed_next_state).astype(int))]
		# next_state, broken = transitions[next_state_type](state, proposed_next_state)
		broken = False
		proposed_ag = self.env.state_to_goal(proposed_next_state)
		next_state = proposed_next_state if (proposed_ag <= self.size-1).all() and (proposed_ag >= 0).all() else state

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
		return self.get_obs(), reward, False, {"is_success": reward == 0}

	# def compute_reward(self, ag, dg, info=None):
	# 	return (ag.astype(int) == dg.astype(int)).all(axis=-1) - 1
	def compute_reward(self, ag, dg, info=None):
		true_threshold = 2
		threshold = .5#2/self.size*true_threshold
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
		moves = moves + [move/2 for move in moves] 
		state_obs = self.get_state_obs()
		ag = self.env.state_to_goal(self.state)
		surroundings = []
		# next_state_type = self.grid[tuple(proposed_next_state.astype(int))]
		return {
			"state": state_obs,
			# "observation": state,
			"observation": np.append(state_obs, surroundings),
			"achieved_goal": state_normalize(self.get_goal(), self.size),
			"desired_goal": state_normalize(self.goal, self.size)
		}


class RandomResetGridworldEnv(GridworldEnv):
	def __init__(self, size, start, goal, reset_grid):
		super().__init__(size, start, goal)
		self.reset_grid = reset_grid

	def reset(self):
		self.grid = self.reset_grid()
		return super().reset()


class SimpleDynamicsGridworldEnv(GridworldEnv):
	def __init__(self, size, start, goal):
		super().__init__(size, start, goal)
		self.env = SimpleDynamicsEnv(size, start)

class AsteroidsGridworldEnv(GridworldEnv):
	def __init__(self, size, start, goal):
		super().__init__(size, start, goal)
		self.env = AsteroidsDynamicsEnv(size, start)

class CarGridworldEnv(GridworldEnv):
	def __init__(self, size, start, goal):
		super().__init__(size, start, goal)
		self.env = CarDynamicsEnv(size, start)


class RotationEnv(gym.GoalEnv):
	def __init__(self, vel_goal= False, shift=False, shift_scale=1):
		self.size=10
		self.translation_speed = self.size/2
		self.dt = .2
		self.acc_speed = 2
		self.rot_speed = 5

		self.drag = 0
		self.action_dim = 2
		self.state_dim = 5
		self.goal_dim = 2

		self.low = 0
		self.high = self.size
		self.obs_low = np.array([self.low, self.low, -1, -1, -1])
		self.obs_high = np.array([self.high, self.high, 1, 1, 1])
		self.action_space 	=spaces.Box(-1, 1, shape=(self.action_dim,), dtype='float32')
		self.observation_space = spaces.Dict(dict(
		    desired_goal	=spaces.Box(self.low, self.high, shape= (self.goal_dim,), dtype='float32'),
		    achieved_goal	=spaces.Box(self.low, self.high, shape= (self.goal_dim,), dtype='float32'),
		    observation 	=spaces.Box(self.obs_low, self.obs_high, dtype='float32'),
		))

		# self.scale = (self.high-self.low)/2
		self.threshold = .1*(self.high-self.low)/2
		self.vel_goal = vel_goal
		self.shift = shift
		self.shift_scale = 1

	def scale(self, s):
		return s * 2/(self.high-self.low)

	def reset(self): 
		self.goal  = self.observation_space['desired_goal'].sample()
		state = self.observation_space['observation' ].sample()
		self.position = state[:2]
		# self.velocity = state[2:4]
		self.velocity = np.zeros(2)#state[2:4]
		self.rotation = np.array([0])#square_to_radian(state[-1:])
		# self.state['pos'] = state[:2]
		# self.state['vel'] = state[2:4]
		# self.state['rot'] = square_to_radian(state[-1:])
		return self._get_obs()



	def _get_obs(self):
		observation = np.concatenate([self.scale(self.position), self.velocity, radian_to_square(self.rotation)], axis=-1)
		assert type(observation) == np.ndarray
		assert observation.shape == (5,)
		assert self.position.shape == (2,)
		assert self.velocity.shape == (2,)
		assert self.rotation.shape == (1,)
		ag = observation[:self.goal_dim]
		obs = {'observation': observation, 'achieved_goal': ag, 'desired_goal': self.scale(self.goal)}
		return obs

	def _goal_distance(self, a,b): 
		return ((a-b)**2).sum(axis=-1)**.5

	def compute_reward(self, state1, state2, info=None):
		reward = (self._goal_distance(state1, state2) < self.threshold) - 1
		return reward

	def step(self, action):
		new_rotation = (self.rotation + action[1]*self.dt*self.rot_speed)%(2*math.pi)
		new_acceleration = np.array([action[0]*math.cos(new_rotation), action[0]*math.sin(new_rotation)])
		new_velocity = self.velocity*(1-self.acc_speed*self.dt) + new_acceleration*self.acc_speed*self.dt
		# new_velocity = np.clip(new_velocity, -1, 1)
		norm = np.linalg.norm(new_velocity, ord=2)
		new_velocity = new_velocity if norm <= 1 else new_velocity/(norm + .0001)
		new_position = np.clip(self.position + new_velocity*self.dt*self.translation_speed, self.low, self.high)

		self.rotation = new_rotation
		self.velocity = new_velocity
		self.position = new_position

		obs = self._get_obs()
		ag = obs['achieved_goal']
		dg = obs['desired_goal']
		reward = self.compute_reward(ag, dg)
		is_success = reward > -.5
		info = {'is_success': is_success}

		return obs,reward, False, info


def generate_random_map(size):
	size = 10
	mid = size//2
	offset = 2
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	grid = np.zeros((size, size))

	for i in range(size):
		block_chance = .0
		for j in range(size):
			if np.random.rand() < block_chance:
				grid[i, j] = BLOCK
			if np.random.rand() < block_chance:
				# grid[i, j] = BREAKING_DOOR
				grid[i, j] = NONBREAKING_DOOR
			if np.random.rand() < block_chance:
				grid[i, j] = LOWCHANCE_BREAKING_DOOR

	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

	adj = [-1, 0, 1]
	for i in adj:
		for j in adj:
			grid[start_pos + i, start_pos + j] = EMPTY
			grid[goal_pos + i, goal_pos + j] = EMPTY

	return grid



def generate_blocky_random_map(size):
	size = 10
	mid = size//2
	offset = 2
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	grid = np.zeros((size, size))
	mean_length = 4
	num_squares = (size-1)**2
	block_fraction = .3

	def assign_blocks(block_type, block_fraction=block_fraction):
		for _ in range(int(num_squares*block_fraction/mean_length)):
			loc = np.random.randint(offset, size-offset, size=2).squeeze()
			# pdb.set_trace()
			grid[tuple(loc)] = block_type
			while not np.random.rand() < 1/(mean_length+1):
				step_loc = lambda loc : (loc + np.array(random.sample(noise_samples, 1)).squeeze())%size
				loc = step_loc(loc)
				grid[tuple(loc)] = block_type
			print(tuple(loc))
	assign_blocks(LOWCHANCE_BREAKING_DOOR)
	assign_blocks(NONBREAKING_DOOR, block_fraction)
	assign_blocks(BLOCK, block_fraction=block_fraction/2)

	for i in range(size):
		#Borders
		grid[0,i] = BLOCK
		grid[size-1,i] = BLOCK
		grid[i,0] = BLOCK
		grid[i, size-1] = BLOCK

	adj = [-1, 0, 1]
	for i in adj:
		for j in adj:
			grid[start_pos + i, start_pos + j] = EMPTY
			grid[goal_pos + i, goal_pos + j] = EMPTY

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


def get_class_constructor(env_type):
	if env_type == "linear":
		env = lambda size, start, goal:  SimpleDynamicsGridworldEnv(size, start, goal)
	elif env_type == "asteroids": 
		env = lambda size, start, goal:  AsteroidsGridworldEnv(size, start, goal)
	elif env_type == "car": 
		env = lambda size, start, goal:  CarGridworldEnv(size, start, goal)
	else: 
		print(f"No dynamics environment matches name {env_type}")
		raise Exception
	return env

def random_map(env_type="linear"): 
	size = 14
	grid = generate_random_map(size)
	mid = size//2
	offset = 3
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	env = get_class_constructor(env_type)(size, start, goal)
	env.grid = grid
	return RotationEnv()
	return env

def random_blocky_map(env_type="linear"): 
	size = 14
	grid = generate_blocky_random_map(size)
	mid = size//2
	offset = 3
	start_pos = offset
	goal_pos  = size-offset-1
	start  = np.array([start_pos]*2)
	goal  = np.array([goal_pos]*2)
	env = get_class_constructor(env_type)(size, start, goal)
	env.grid = grid
	return env

