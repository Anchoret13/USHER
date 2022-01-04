from gym.core import Wrapper
import numpy as np

class ActionRandomnessWrapper(Wrapper):

	def __init__(self, env, rand):
		self.env = env
		self.rand = rand

		self._action_space = None
		self._observation_space = None
		self._reward_range = None
		self._metadata = None
		self.size = env.action_space.sample().shape

		self.action_space = env.action_space
		self._max_episode_steps = env._max_episode_steps

	def step(self, action):
		return self.env.step(action + np.random.normal(scale = self.rand, size=self.size))