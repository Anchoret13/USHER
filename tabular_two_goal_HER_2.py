import numpy as np		#type: ignore
from typing import Tuple, Sequence, Callable, List
from functools import reduce

from gridworld import create_map_1
from display import *
from constants import *
import pdb
# from plot import plot
import matplotlib.pyplot as plt #type: ignore
 
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
episodes = 5*10**4
base_lr = .01
gamma = .8

traj_steps = 50#25


class Q: 
	def __init__(self, size: int, compute_reward, default_goal: np.ndarray, k: int = 4):
		# self.q_table = np.ones((env.size, env.size, 5))
		self.q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size,  5)) + 1
		self.pure_q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size, 5)) + 1
		self.usher_q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size, 5)) + 1
		# self.v_table = np.zeros((env.size, env.size))
		self.compute_reward = compute_reward
		self.default_goal = default_goal
		self.k = k
		self.p = 1/(1+self.k)

	def sample_action(self, state: np.ndarray, goal: np.ndarray, pol_goal: np.ndarray =None, temp=5) -> int:
		use_pol_goal: np.ndarray = goal if type(pol_goal == None) else pol_goal
		loc: tuple = tuple(state) + tuple(goal) + tuple(use_pol_goal)
		q = self.q_table[loc]
		return softmax_sample(q, temp)

	def argmax_action(self, state: np.ndarray, goal: np.ndarray, pol_goal: np.ndarray=None) -> int:
		use_pol_goal = goal if type(pol_goal == None) else pol_goal
		q = self.q_table[tuple(state) + tuple(goal) + tuple(use_pol_goal)]
		return q.argmax()

	def state_value(self, state: np.ndarray) -> float:
		return self.q_table[tuple(state) + tuple(self.default_goal) + tuple(self.default_goal)].max()

	def _update_q(self, state: np.ndarray, goal: np.ndarray, pol_goal: np.ndarray, action: int, 
			next_state: np.ndarray, reward: float, lr: float=.05) -> None:
		bellman_update = (1-gamma)*reward + gamma*self.q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		a = (1-lr)*self.q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = lr*bellman_update
		self.q_table[tuple(state) + tuple(goal) + tuple(pol_goal) ][action] = a + b

	# def

	def _update_usher_q(self, state: np.ndarray, goal: np.ndarray, pol_goal: np.ndarray, action: int, 
			next_state: np.ndarray, reward: float, lr: float=.05, p=None, t_remaining=None) -> None:
		
		if (goal == pol_goal).all():
			p = self.p if type(p) == type(None) else p
			# next_action = self.usher_q_table.sample_action(state, goal, pol_goal)
			ratio = (
				(p + (1-p)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action])/
				(p + (1-p)*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())
				)
		else: 
			p = .01*lr
			# p = self.p if type(p) == type(None) else p
			#Not mathematically necessary, but reduces likelihood of overflow errors and stabilizes convergence a bit
			ratio = (
				(p + (1-p)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action])/
				(p + (1-p)*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())
				)
		new_lr = min(lr*ratio, 1)
		bellman_update = (1-gamma)*reward + gamma*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		a = (1-new_lr)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = new_lr*bellman_update
		self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action] = a + b

	def _update_pure_q(self, state: np.ndarray, goal: np.ndarray, pol_goal: np.ndarray, action: int, 
			next_state: np.ndarray, reward: float, lr: float=.05) -> None:
			#Not updated by HER
		bellman_update = (1-gamma)*reward + gamma*self.pure_q_table[tuple(next_state) + tuple(goal) + tuple(goal)].max()
		a = (1-lr)*self.pure_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = lr*bellman_update
		self.pure_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action] = a + b


	def update(self, episode: list, lr: float=.05) -> None:
		# cumulative_r = 0#self.v_table[tuple(episode[-1][2])]
		k = self.k
		c = 1
		p = self.p*c
		#shorten trajectory?
		for i in range(len(episode)): 
			frame = episode[-i]
			state, desired_goal, achieved_goal, action, next_state, reward = frame
			self._update_pure_q(state, desired_goal, desired_goal, action, next_state, reward, lr)
			self._update_q(state, desired_goal, desired_goal, action, next_state, reward, lr)
			self._update_usher_q(state, desired_goal, desired_goal, action, next_state, reward, lr, p=1)

			# self._update_usher_q(state, desired_goal, desired_goal, action, next_state, reward, lr*c, p=p + (1-p)*gamma**(i))#1)
			for _ in range(k):
			# for _ in range(i, k):
				if i == 0: 
					rand_i = 0
				else: 
					rand_i = np.random.geometric(1-gamma) % i
					# rand_i = np.random.randint(len(episode) - i, len(episode))
				goal = episode[rand_i][2] #Achieved goal
				self._update_q(state, goal, desired_goal, action, next_state, reward, lr)
				rand_i = np.random.geometric(1-gamma)
				goal = episode[rand_i][2] if rand_i < i else desired_goal
				# self._update_usher_q(state, goal, desired_goal, action, next_state, reward, lr, p=p + (1-p)*gamma**(i))
				self._update_usher_q(state, goal, desired_goal, action, next_state, reward, lr, p=gamma**i)
				# self._update_usher_q(state, goal, desired_goal, action, next_state, reward, lr, p=self.p, t_remaining = i)





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
def learn_q_function(k: int = 4):
	compute_reward = env.compute_reward
	default_goal = env.new_goal
	q = Q(env.size, compute_reward, default_goal, k)
	ave_r = 0
	record_num = 100

	display_init(env, q)
	iterations = []
	her_vals = []
	usher_vals = []
	q_vals = []
	ave_r_vals = []

	for episode in range(episodes):
		draw_grid(env, q)
		state = env.reset()['observation']
		power = .8
		lr = (2**(-100*episode/episodes) + base_lr/(episodes**power/100+1))
		# lr = .1*base_lr*episodes/(.1*episodes + episode)
		# lr = .05
		if episode%record_num == 0: 
			print("---------------------------")
			print(f"Episode {episode} of {episodes}")
			get_ave_r = lambda ep: (1-gamma)*sum([gamma**i*ep[i][-1] for i in range(len(ep))])
			eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal)) for _ in range(50)]
			# [q.update_v(ep) for ep in eps]
			# ave_lr = 1/(episodes+1)#2**(-50*episode/episodes)
			ave_lr = (2**(-100*episode/episodes) + .5/(episodes**power/100+1))

			ave_r = ave_r*(1-ave_lr) + ave_lr*mean([get_ave_r(ep) for ep in eps])
			her_val = q.q_table[tuple(state) + tuple(default_goal) + tuple(default_goal)].max()
			usher_val = q.usher_q_table[tuple(state) + tuple(default_goal) + tuple(default_goal)].max()
			q_val = q.pure_q_table[tuple(state) + tuple(default_goal) + tuple(default_goal)].max()

			print(f"HER q value: \t\t\t{her_val}")
			print(f"USHER weighted q value: \t{usher_val}")
			print(f"Pure q value: \t\t\t{q_val}")
			print(f"Average reward: \t\t{ave_r}")

			iterations.append(episode*record_num*traj_steps)
			her_vals.append(her_val)
			usher_vals.append(usher_val)
			q_vals.append(q_val)
			ave_r_vals.append(ave_r)			

		ep = observe_episode(env, q, lambda s: q.sample_action(s, default_goal))

		q.update(ep, lr=lr)

	return {
		"her": her_vals[-1],
		"usher": usher_vals[-1],
		"q": q_vals[-1],
	}

	plt.plot(iterations, her_vals, 	label="her_vals")
	plt.plot(iterations, usher_vals,label="usher_vals")
	plt.plot(iterations, q_vals, 	label="q_vals")
	plt.plot(iterations, ave_r_vals,label="ave_r_vals")
	plt.legend()
	plt.show()

def show_k_vals():
	k_list = [0,1,2,3,4,6,8,16]
	output_list = {
		"her": [],
		"usher": [],
		"q": [],
	}
	for k in k_list:
		output = learn_q_function(k)
		for key in output_list.keys(): 
			output_list[key].append(output[key])
		# output_list = {key: output[key] + output_list[key] for key in output_list.keys()}

	for key in output_list.keys(): 
		plt.plot(k_list, output_list[key], label=key)

	plt.legend()
	plt.show()

learn_q_function(4)