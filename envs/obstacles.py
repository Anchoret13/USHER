import random
from typing import Tuple, Sequence

State=Tuple[int, int]
Action=Tuple[int, int]

noise_samples = [(1,0), (-1, 0), (0, 1), (0,1)]

def add(a: Sequence, b: Sequence) -> Sequence:
	assert len(a) == len(b)
	return (a[i] + b[i] for i in range(len(a)))

def sample_noise(k: int) -> State:
	return random.sample(noise_samples + [(0,0)]*k)

class Obstacle: 
	# def __init__(self):
	# 	return NotImplementedError
	def interact(self, state: State, action: Action) -> State:
		# state, action -> next_state, reward
		return NotImplementedError 


class RandomDoor(Obstacle):
	def interact(self, state: State, action: Action) -> State:
		if random.random() < .01:
			return add(state, action), 0
		else: 
			return state, 0

class Block(Obstacle):
	def interact(self, state: State, action: Action) -> State:
		return state, 0

class Wind(Obstacle):
	def __init__(self, k: int=6):
		self.k = k

	def interact(self, state: State, action: Action) -> State:
		next_state = add( add( state, sample_noise(self.k)), action)
		return next_state, 0

class ObstacleSet:
	