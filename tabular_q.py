import numpy as np
from typing import Tuple, Sequence, Callable, List
from functools import reduce

from gridworld import create_map_1
from display import *
from constants import *
import pdb

State=np.ndarray
Action=np.ndarray
ActionIndex = int

def mean(lst):
	return sum(lst)/len(lst)

def softmax(arr, temp: float):
	unnormed_vals = 2**(temp*arr)
	return unnormed_vals/unnormed_vals.sum()

def softmax_sample(arr, temp: float):
	probabilities = softmax(arr, temp)
	return np.random.choice(np.arange(len(arr)), p=probabilities)



env = create_map_1()
episodes = 10**4
base_lr = .5
gamma = .8



class Q: 
	def __init__(self, size):
		self.q_table = np.ones((env.size, env.size, 5))
		# self.q_table = np.zeros((env.size, env.size, 5))
		self.v_table = np.zeros((env.size, env.size))

	def sample_action(self, state: np.ndarray, temp=5) -> int:
		q = self.q_table[tuple(state)]
		return softmax_sample(q, temp)

	def argmax_action(self, state: np.ndarray) -> int:
		q = self.q_table[tuple(state)]
		return q.argmax()

	def state_value(self, state: np.ndarray) -> float:
		return self.q_table[tuple(state)].max()

	def _update_q(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, lr: float=.05) -> None:
		bellman_update = (1-gamma)*reward + gamma*self.q_table[tuple(next_state)].max()
		a = (1-lr)*self.q_table[tuple(state)][action]
		b = lr*bellman_update
		self.q_table[tuple(state)][action] = a + b

	def _update_v(self, state: np.ndarray, cumulative_r: float, lr: float=.05) -> None:
		a = (1-lr)*self.v_table[tuple(state)]
		b = lr*cumulative_r
		self.v_table[tuple(state)] = a + b

	def update(self, episode: list, lr: float=.05) -> None:
		# cumulative_r = 0#self.v_table[tuple(episode[-1][2])]
		for frame in reversed(episode): 
			state, action, next_state, reward = frame
			# cumulative_r = (1-gamma)*reward + gamma*cumulative_r 
			self._update_q(state, action, next_state, reward, lr)
			# self._update_v(state, cumulative_r, lr)

	def update_v(self, episode: list, lr: float=.05) -> None:
		cumulative_r = self.v_table[tuple(episode[-1][2])]
		for frame in reversed(episode): 
			state, action, next_state, reward = frame
			cumulative_r = (1-gamma)*reward + gamma*cumulative_r 
			self._update_v(state, cumulative_r, lr)


index_to_action = {
	0: np.array([1,0]),
	1: np.array([-1,0]),
	2: np.array([0,1]),
	3: np.array([0,-1]),
	4: np.array([0,0])
}

# def observe_transition(env, q: Q, temp: float=5) -> Tuple[State, Action, State, float]:
# 	state = env.get_state()
# 	action = q.sample_action(state, temp)
# 	env_action = index_to_action[action]
# 	obs, reward, done, info = env.step(env_action)
# 	next_state=obs['observation']
# 	return (state, action, next_state, reward)


# def observe_episode(env, q: Q, temp: float=5) -> list:
# 	env.reset()
# 	return [observe_transition(env, q, temp) for _ in range(50)]

def observe_transition(env, q: Q, policy: Callable) -> Tuple[State, ActionIndex, State, float]:
	state = env.get_state()
	action = policy(state)
	env_action = index_to_action[action]
	obs, reward, done, info = env.step(env_action)
	next_state=obs['observation']
	return (state, action, next_state, reward)


def observe_episode(env, q: Q, policy: Callable) -> List[Tuple]:
	env.reset()
	return [observe_transition(env, q, policy) for _ in range(50)]



# exit()
def learn_q_function():
	q = Q(env.size)
	ave_r = 0

	display_init(env, q)

	for episode in range(episodes):
		draw_grid(env, q)
		state = env.reset()['observation']
		lr = base_lr*2**(-2*episode/episodes)
		# lr = .1*base_lr*episodes/(.1*episodes + episode)
		# lr = .05
		if episode%100 == 0: 
			print("---------------------------")
			get_ave_r = lambda ep: (1-gamma)*sum([gamma**i*ep[i][-1] for i in range(len(ep))])
			eps = [observe_episode(env, q, q.argmax_action) for _ in range(50)]
			[q.update_v(ep) for ep in eps]
			ave_r = mean([get_ave_r(ep) for ep in eps])
			print(f"Average reward: {ave_r}")
			print(f"Base q value: {q.q_table[tuple(state)].max()}")
			print(f"Base v value: {q.v_table[tuple(state)]}")
			

		ep = observe_episode(env, q, lambda s: q.sample_action(s))

		q.update(ep, lr=lr)

learn_q_function()
