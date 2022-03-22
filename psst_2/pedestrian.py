class Pedestrian: 
	def __init__(self, size):
		self.size = size
		self.t_steps = 10

	def dynamics(self, state, dt):
		return {"pos": pos, 
				"goal": , 
				"timer": ,
				"avoidance timer": 
				}

	def reset(self, initial_position): 
		return {"pos": initial_position, 
				"goal": np.random.uniform(1, self.size-1, 2), 
				"timer": self.t_steps,
				"avoidance timer": 0
				}

	def state_to_obs(self, state) -> np.ndarray:
		# return state
		return state['pos']

	def state_to_location(self, state) -> np.ndarray:
		# return state
		return state['pos']

	def stop(self, proposed_invalid_state, prev_state): return prev_state