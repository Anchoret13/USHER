import numpy as np		#type: ignore
from typing import Tuple, Sequence, Callable, List, Union
from functools import reduce

from gridworld import create_map_1
# from test_gridworld import create_map_1
from display import *
from constants import *
import pdb
# from plot import plot
import matplotlib.pyplot as plt #type: ignore

# import typeguard
from typeguard import typechecked
 
Obs=Tuple[np.ndarray, bool]
State=np.ndarray
Action=np.ndarray
ActionIndex = Union[int, np.int64]
PolicyType = str#Union["Q", "HER", "USHER"]

env = create_map_1()
episodes = int(2.5*10**4)
base_lr = .01
gamma = .9

traj_steps = 50#25

# ALL_AVES = False
ALL_AVES = True


def mean(lst):
	return sum(lst)/len(lst)

def softmax(arr, temp: float):
	unnormed_vals = 2**(temp*arr)
	return unnormed_vals/unnormed_vals.sum()

def softmax_sample(arr, temp: float):
	probabilities = softmax(arr, temp)
	return np.random.choice(np.arange(len(arr)), p=probabilities)


# @typechecked
class Q: 
	def __init__(self, size: int, compute_reward, default_goal: np.ndarray, k: int = 4):
		# self.q_table = np.ones((env.size, env.size, 5))
		self.q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size,  5)) + 1
		self.pure_q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size, 5)) + 1
		self.usher_q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size, 5)) + 1
		self.g_pi_table = np.zeros((env.size, env.size, env.size, env.size, 5)) + 1
		# self.v_table = np.zeros((env.size, env.size))
		self.compute_reward = compute_reward
		self.default_goal = default_goal
		self.k = k
		self.p = 1/(1+self.k)


	def select_table(self, policy: PolicyType):
		if policy == "Q":
			q_table = self.pure_q_table
		elif policy == "HER": 
			q_table = self.q_table
		elif policy == "USHER":
			q_table = self.usher_q_table
		elif policy =="G_PI":
			q_table = self.g_pi_table
		return q_table


	def sample_action(self, obs: Obs, goal: np.ndarray, pol_goal: np.ndarray =None, 
		temp=5) -> ActionIndex:
		use_pol_goal: np.ndarray = goal if type(pol_goal == None) else pol_goal
		state = obs[0]
		loc: tuple = tuple(state) + tuple(goal) + tuple(use_pol_goal)
		q = self.q_table[loc]
		return softmax_sample(q, temp)

	def argmax_action(self, obs: Obs, goal: np.ndarray, pol_goal: np.ndarray=None, 
		policy: PolicyType = "HER") -> ActionIndex:
		use_pol_goal = goal if type(pol_goal == None) else pol_goal
		q_table = self.select_table(policy)
		state = obs[0]
		q = q_table[tuple(state) + tuple(goal) + tuple(use_pol_goal)]
		return q.argmax()

	def state_value(self, obs: Obs, policy: PolicyType = "HER") -> float:
		state, broken = Obs
		if broken:
			# assert False
			return 0
		else: 
			q_table = self.select_table(policy)
			return q_table[tuple(state) + tuple(self.default_goal) + tuple(self.default_goal)].max()

	def _update_q(self, obs: Obs, goal: np.ndarray, pol_goal: np.ndarray, action: ActionIndex, 
			next_obs: Obs, lr: float=.05) -> None:
		state, broken = obs
		next_state, next_broken = next_obs
		reward = 1 if (next_state == goal).all() else 0
		if broken:
			# assert False
			return
		if next_broken and not broken: 
			bellman_update = (1-gamma)*reward + gamma*0
		else: 
			bellman_update = (1-gamma)*reward + gamma*self.q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		bellman_update = (1-gamma)*reward + gamma*self.q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		a = (1-lr)*self.q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = lr*bellman_update
		self.q_table[tuple(state) + tuple(goal) + tuple(pol_goal) ][action] = a + b

	def _update_g_pi(self, obs: Obs, goal: np.ndarray, desired_goal: np.ndarray, action: ActionIndex, 
		next_obs: Obs, lr: float=.05) -> None:

		state, broken = obs
		next_state, next_broken = next_obs
		reward = 1 if (desired_goal == goal).all() else 0
		if broken:
			# assert False
			return
		if next_broken and not broken: 
			bellman_update = (1-gamma)*reward + gamma*0
		else: 
			bellman_update = (1-gamma)*reward + gamma*self.g_pi_table[tuple(next_state) + tuple(goal)][action]
		
		a = (1-lr)*self.g_pi_table[tuple(state) + tuple(desired_goal)][action]
		b = lr*bellman_update
		self.g_pi_table[tuple(state) + tuple(desired_goal)][action] = a + b

	def _update_on_target_usher_q(self, obs: Obs, goal: np.ndarray, pol_goal: np.ndarray, action: ActionIndex, 
			next_obs: Obs, lr: float=.05) -> None:

		state, broken = obs
		next_state, next_broken = next_obs
		reward = 1 if (next_state == goal).all() else 0
		if broken:
			# assert False
			return
		if next_broken and not broken: 
			bellman_update = (1-gamma)*reward + gamma*0
		else: 
			bellman_update = (1-gamma)*reward + gamma*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		new_lr = lr
		a = (1-new_lr)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = new_lr*bellman_update
		self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action] = a + b

	def _update_off_target_usher_q(self, obs: Obs, goal: np.ndarray, pol_goal: np.ndarray, action: ActionIndex, 
			next_obs: Obs, lr: float=.05, p=None, t_remaining=None) -> None:

		state, broken = obs
		next_state, next_broken = next_obs
		reward = 1 if (next_state == goal).all() else 0
		if broken:
			return
		if next_broken and not broken: 
			bellman_update = (1-gamma)*reward + gamma*0
		else: 
			bellman_update = (1-gamma)*reward + gamma*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		if (goal == pol_goal).all():
			p = self.p if type(p) == type(None) else p
			next_action = self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(goal)].argmax()
			# ratio = (
			# 	((self.g_pi_table[tuple(state) + tuple(pol_goal)][action])*(p*gamma + (1-p*gamma)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]))/
			# 	((self.g_pi_table[tuple(next_state) + tuple(pol_goal)][next_action])*(p*gamma + (1-p*gamma)*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()))
			# 	)
			ratio = (
				(p + (1-p)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action])/
				(p + (1-p)*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())
				)
		else: 
			p = .01*lr
			ratio = self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]/(self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())
		new_lr = min(lr*ratio, 1)
		a = (1-new_lr)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = new_lr*bellman_update
		self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action] = a + b

	def _update_usher_q(self, obs: Obs, goal: np.ndarray, pol_goal: np.ndarray, action: ActionIndex, 
			next_obs: Obs,  lr: float=.05, p=None, t_remaining=None) -> None:

		state, broken = obs
		next_state, next_broken = next_obs
		reward = 1 if (next_state == goal).all() else 0
		if broken:
			return
		if next_broken and not broken: 
			bellman_update = (1-gamma)*reward + gamma*0
		else: 
			bellman_update = (1-gamma)*reward + gamma*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max()
		if (goal == pol_goal).all():
			p = self.p if type(p) == type(None) else p
			ratio = (
				(p*gamma + (1-p*gamma)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action])/
				(p + (1-p)*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())
				)
			# ratio = (
			# 	(p + (1-p)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action])/
			# 	(p + (1-p)*self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())
			# 	)
		else: 
			p = .01*lr
			ratio = self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]/(self.usher_q_table[tuple(next_state) + tuple(goal) + tuple(pol_goal)].max())
		new_lr = min(lr*ratio, 1)
		a = (1-new_lr)*self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action]
		b = new_lr*bellman_update
		self.usher_q_table[tuple(state) + tuple(goal) + tuple(pol_goal)][action] = a + b


	def _update_pure_q(self, obs: Obs, goal: np.ndarray, pol_goal: np.ndarray, action: ActionIndex, 
			next_obs: Obs, lr: float=.05) -> None:
			#Not updated by HER
		state, broken = obs
		next_state, next_broken = next_obs
		reward = 1 if (next_state == goal).all() else 0
		if broken:
			return
		if next_broken and not broken: 
			bellman_update = (1-gamma)*reward + gamma*0
		else: 
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

			self._update_g_pi(state, desired_goal, desired_goal, action, next_state, lr)

			self._update_pure_q(state, desired_goal, desired_goal, action, next_state, lr)
			self._update_q(state, desired_goal, desired_goal, action, next_state, lr)
			self._update_on_target_usher_q(state, desired_goal, desired_goal, action, next_state, lr)

			# self._update_usher_q(state, desired_goal, desired_goal, action, next_state, reward, lr*c, p=p)#1)
			for _ in range(k):
			# if np.random.rand() <
			# for _ in range(i, k):
				if i == 0: 
					rand_i = 0
				else: 
					rand_i = np.random.geometric(1-gamma) % i

				rand_i = np.random.geometric(1-gamma)
				goal = episode[rand_i][2] if rand_i < i else desired_goal

				# self._update_g_pi(state, goal, desired_goal, action, next_state, lr)
				if (goal == desired_goal).all():
					self._update_q(state, goal, goal, action, next_state, lr)
					self._update_off_target_usher_q(state, goal, goal, action, next_state, lr, p=gamma**i)


index_to_action = {
	0: np.array([1,0]),
	1: np.array([-1,0]),
	2: np.array([0,1]),
	3: np.array([0,-1]),
	4: np.array([0,0])
}


# @typechecked
def observe_transition(env, q: Q, policy: Callable) -> Tuple[Obs, State, State, ActionIndex, Obs, float]:
	state = env.get_state()
	action = policy(state)
	env_action = index_to_action[action]
	obs, reward, done, info = env.step(env_action)
	dg=obs['desired_goal']
	ag=obs['achieved_goal']
	next_state=obs['observation']
	return (state, dg, ag, action, next_state, reward)


# @typechecked
def observe_episode(env, q: Q, policy: Callable) -> List[Tuple]:
	env.reset()
	return [observe_transition(env, q, policy) for _ in range(traj_steps)]



# exit()

# @typechecked
def learn_q_function(k: int = 4):
	compute_reward = env.compute_reward
	default_goal = env.new_goal
	q = Q(env.size, compute_reward, default_goal, k)
	ave_r = 0
	ave_q_r = 0
	ave_usher_r = 0
	record_num = 100

	# display_init(env, q)
	iterations = []
	her_vals = []
	usher_vals = []
	q_vals = []
	ave_r_vals = []
	ave_q_r_vals = []
	ave_usher_r_vals = []

	for episode in range(episodes):
		# draw_grid(env, q)
		state = env.reset()['observation'][0]

		power = .8
		lr = (2**(-100*episode/episodes) + base_lr/(episodes**power/100+1))
		# lr = .1*base_lr*episodes/(.1*episodes + episode)
		# lr = .05
		if episode%record_num == 0: 
			print("---------------------------")
			print(f"Episode {episode} of {episodes}")
			get_ave_r = lambda ep: (1-gamma)*sum([gamma**i*ep[i][-1] for i in range(len(ep))])
			# pdb.set_trace()
			eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal)) for _ in range(50)]
			# [q.update_v(ep) for ep in eps]
			# ave_lr = 1/(episodes+1)#2**(-50*episode/episodes)
			ave_lr = min(traj_steps*(2**(-100*episode/episodes) + base_lr/(episodes**power/100+1)), .9)

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


			#-------------------------------------------------------------------		
			if ALL_AVES:
				q_eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal, policy="Q")) for _ in range(50)]
				ave_q_r = ave_q_r*(1-ave_lr) + ave_lr*mean([get_ave_r(ep) for ep in q_eps])
				usher_eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal, policy="Q")) for _ in range(50)]
				ave_usher_r = ave_usher_r*(1-ave_lr) + ave_lr*mean([get_ave_r(ep) for ep in usher_eps])
				
				print(f"Average Q return: \t\t{ave_q_r}")
				print(f"Average USHER return: \t\t{ave_usher_r}")


				ave_q_r_vals.append(ave_q_r)	
				ave_usher_r_vals.append(ave_usher_r)	

		

		ep = observe_episode(env, q, lambda s: q.sample_action(s, default_goal))

		q.update(ep, lr=lr)

	# return {
	# 	"her": her_vals[-1],
	# 	"usher": usher_vals[-1],
	# 	"q": q_vals[-1],
	# }

	plt.plot(iterations, her_vals, 	label="HER Q value")
	plt.plot(iterations, ave_r_vals,label="HER true reward")
	plt.plot(iterations, q_vals, 	label="Regular Q value")
	plt.plot(iterations, usher_vals,label="USHER Q value")
	plt.plot(iterations, ave_q_r_vals,label="Regular Q true reward")
	plt.plot(iterations, ave_usher_r_vals,label="Usher true reward")
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