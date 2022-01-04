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
gamma = .9

traj_steps = 10


class Q: 
	def __init__(self, size: int, compute_reward, default_goal: np.ndarray):
		# self.q_table = np.ones((env.size, env.size, 5))
		self.q_table = np.zeros((env.size, env.size, env.size, env.size,  5)) + 1
		self.pure_q_table = np.zeros((env.size, env.size, env.size, env.size,  5)) + 1
		# self.v_table = np.zeros((env.size, env.size))
		self.compute_reward = compute_reward
		self.default_goal = default_goal

	def sample_action(self, state: np.ndarray, goal: np.ndarray, temp=5) -> int:
		q = self.q_table[tuple(state) + tuple(goal)]
		return softmax_sample(q, temp)

	def argmax_action(self, state: np.ndarray, goal: np.ndarray) -> int:
		q = self.q_table[tuple(state) + tuple(goal)]
		return q.argmax()

	def state_value(self, state: np.ndarray) -> float:
		return self.q_table[tuple(state) + tuple(self.default_goal)].max()

	def _update_q(self, state: np.ndarray, goal: np.ndarray, action: np.ndarray, next_state: np.ndarray, 
			reward: float, lr: float=.05) -> None:
		bellman_update = (1-gamma)*reward + gamma*self.q_table[tuple(next_state) + tuple(goal)].max()
		a = (1-lr)*self.q_table[tuple(state) + tuple(goal)][action]
		b = lr*bellman_update
		self.q_table[tuple(state) + tuple(goal)][action] = a + b

	def _update_pure_q(self, state: np.ndarray, goal: np.ndarray, action: np.ndarray, next_state: np.ndarray, 
			reward: float, lr: float=.05) -> None:
			#Not updated by HER
		bellman_update = (1-gamma)*reward + gamma*self.pure_q_table[tuple(next_state) + tuple(goal)].max()
		a = (1-lr)*self.pure_q_table[tuple(state) + tuple(goal)][action]
		b = lr*bellman_update
		self.pure_q_table[tuple(state) + tuple(goal)][action] = a + b


	def update(self, episode: list, lr: float=.05) -> None:
		# cumulative_r = 0#self.v_table[tuple(episode[-1][2])]
		k = 4
		#shorten trajectory?
		for i in range(len(episode)): 
			frame = episode[-i]
			state, desired_goal, achieved_goal, action, next_state, reward = frame
			self._update_pure_q(state, desired_goal, action, next_state, reward, lr/5)
			self._update_q(state, desired_goal, action, next_state, reward, lr/5)
			for _ in range(k):
			# for _ in range(i, k):
				if i == 0: 
					rand_i = 0
				else: 
					rand_i = np.random.geometric(1-gamma) % i
					# rand_i = np.random.randint(len(episode) - i, len(episode))
				goal = episode[rand_i][2] #Achieved goal
				self._update_q(state, goal, action, next_state, reward, lr)





index_to_action = {
	0: np.array([1,0]),
	1: np.array([-1,0]),
	2: np.array([0,1]),
	3: np.array([0,-1]),
	4: np.array([0,0])
}



def observe_transition(env, q: Q, policy: Callable) -> Tuple[State, State, State, ActionIndex, State, float]:
	state = env.get_state()
	action = policy(state)
	env_action = index_to_action[action]
	obs, reward, done, info = env.step(env_action)
	dg=obs['desired_goal']
	ag=obs['achieved_goal']
	next_state=obs['observation']
	return (state, dg, ag, action, next_state, reward)


def observe_episode(env, q: Q, policy: Callable) -> List[Tuple]:
	env.reset()
	return [observe_transition(env, q, policy) for _ in range(traj_steps)]



# exit()
def learn_q_function():
	compute_reward = env.compute_reward
	default_goal = env.new_goal
	q = Q(env.size, compute_reward, default_goal)
	ave_r = 0

	display_init(env, q)

	for episode in range(episodes):
		draw_grid(env, q)
		state = env.reset()['observation']
		lr = base_lr*2**(-5*episode/episodes)
		# lr = .1*base_lr*episodes/(.1*episodes + episode)
		# lr = .05
		if episode%100 == 0: 
			print("---------------------------")
			get_ave_r = lambda ep: (1-gamma)*sum([gamma**i*ep[i][-1] for i in range(len(ep))])
			eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal)) for _ in range(50)]
			# [q.update_v(ep) for ep in eps]
			ave_r = mean([get_ave_r(ep) for ep in eps])
			print(f"Average reward: {ave_r}")
			print(f"HER q value: {q.q_table[tuple(state) + tuple(default_goal)].max()}")
			print(f"Pure q value: {q.pure_q_table[tuple(state) + tuple(default_goal)].max()}")
			# print(f"Base v value: {q.v_table[tuple(state)]}")
			

		ep = observe_episode(env, q, lambda s: q.sample_action(s, default_goal))

		q.update(ep, lr=lr)


learn_q_function()