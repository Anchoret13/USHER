#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x2598af6e

# Compiled with Coconut version 1.6.0 [Vocational Guidance Counsellor]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop, annotations
import sys as _coconut_sys, os as _coconut_os
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get("__coconut__")
if _coconut_cached_module is not None and _coconut_os.path.dirname(_coconut_cached_module.__file__) != _coconut_file_dir:
    del _coconut_sys.modules["__coconut__"]
_coconut_sys.path.insert(0, _coconut_file_dir)
_coconut_module_name = _coconut_os.path.splitext(_coconut_os.path.basename(_coconut_file_dir))[0]
if _coconut_module_name and _coconut_module_name[0].isalpha() and all(c.isalpha() or c.isdigit() for c in _coconut_module_name) and "__init__.py" in _coconut_os.listdir(_coconut_file_dir):
    _coconut_full_module_name = str(_coconut_module_name + ".__coconut__")
    import __coconut__ as _coconut__coconut__
    _coconut__coconut__.__name__ = _coconut_full_module_name
    for _coconut_v in vars(_coconut__coconut__).values():
        if getattr(_coconut_v, "__module__", None) == "__coconut__":
            try:
                _coconut_v.__module__ = _coconut_full_module_name
            except AttributeError:
                _coconut_v_type = type(_coconut_v)
                if getattr(_coconut_v_type, "__module__", None) == "__coconut__":
                    _coconut_v_type.__module__ = _coconut_full_module_name
    _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_tail_call, _coconut_tco, _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

import numpy as np  #type: ignore  #1 (line num in coconut source)
from typing import Tuple  #2 (line num in coconut source)
from typing import Sequence  #2 (line num in coconut source)
from typing import Callable  #2 (line num in coconut source)
from typing import List  #2 (line num in coconut source)
from typing import Optional  #2 (line num in coconut source)
from functools import reduce  #3 (line num in coconut source)

from gridworld import create_map_1  #5 (line num in coconut source)
# import display 
# from display import *
from constants import *  #8 (line num in coconut source)
import pdb  #9 (line num in coconut source)
# from plot import plot
import matplotlib.pyplot as plt  #type: ignore  #11 (line num in coconut source)


def mean(lst):  #14 (line num in coconut source)
    return (sum(lst) / len(lst))  #15 (line num in coconut source)

@_coconut_tco  #17 (line num in coconut source)
def softmax(arr: Sequence[float], temp: float) -> Sequence[float]:  #17 (line num in coconut source)
# unnormed_vals = 2**(temp*arr)
    unnormed_vals: Sequence[float] = tuple((2**(temp * a) for a in arr))  #19 (line num in coconut source)
    sum_val = sum(unnormed_vals)  #20 (line num in coconut source)
# return unnormed_vals/unnormed_vals.sum()
    return _coconut_tail_call(tuple, (v / sum_val for v in unnormed_vals))  #22 (line num in coconut source)

@_coconut_tco  #24 (line num in coconut source)
def softmax_sample(arr: Action, temp: float) -> ActionIndex:  #24 (line num in coconut source)
    probabilities = softmax(arr, temp)  #25 (line num in coconut source)
    return _coconut_tail_call(np.random.choice, tuple((i for i in range(len(arr)))), p=probabilities)  #26 (line num in coconut source)



env = create_map_1()  #30 (line num in coconut source)
episodes = 5 * 10**4  #31 (line num in coconut source)
base_lr = .01  #32 (line num in coconut source)
gamma = .8  #33 (line num in coconut source)

traj_steps = 50  #25  #35 (line num in coconut source)


class Q:  #38 (line num in coconut source)
    def __init__(self, size: int, compute_reward: Callable, default_goal: Goal, k: int=4):  #39 (line num in coconut source)
# self.q_table = np.ones((env.size, env.size, 5))
        self.q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size, 5)) + 1  #41 (line num in coconut source)
        self.pure_q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size, 5)) + 1  #42 (line num in coconut source)
        self.usher_q_table = np.zeros((env.size, env.size, env.size, env.size, env.size, env.size, 5)) + 1  #43 (line num in coconut source)
# self.v_table = np.zeros((env.size, env.size))
        self.compute_reward = compute_reward  #45 (line num in coconut source)
        self.default_goal = default_goal  #46 (line num in coconut source)
        self.k = k  #47 (line num in coconut source)
        self.p = 1 / (1 + self.k)  #48 (line num in coconut source)

# def sample_action(self, state: State, goal: Goal, pol_goal: Goal | None =None, temp: float=5) -> int:
    def sample_action(self, state: State, goal: Goal, pol_goal: Optional[Goal]=None, temp: float=5) -> int:  #51 (line num in coconut source)
# use_pol_goal: Goal = default(pol_goal, goal)
        pol_goal = goal if pol_goal is None else pol_goal  #53 (line num in coconut source)
        q = self.q_table[state + goal + pol_goal]  #54 (line num in coconut source)
        samp = softmax_sample(q, temp)  #55 (line num in coconut source)
        return (samp)  #56 (line num in coconut source)

# def argmax_action(self, state: State, goal: Goal, pol_goal: Goal | None =None, temp: float=5) -> int:
    @_coconut_tco  #59 (line num in coconut source)
    def argmax_action(self, state: State, goal: Goal, pol_goal: Optional[Goal]=None, temp: float=5) -> int:  #59 (line num in coconut source)
        pol_goal = goal if pol_goal is None else pol_goal  #60 (line num in coconut source)
        q = self.q_table[state + goal + pol_goal]  #61 (line num in coconut source)
        return _coconut_tail_call(q.argmax)  #62 (line num in coconut source)

    @_coconut_tco  #64 (line num in coconut source)
    def state_value(self, state: State) -> float:  #64 (line num in coconut source)
        return _coconut_tail_call(self.q_table[state + self.default_goal + self.default_goal].max)  #65 (line num in coconut source)

    def _update_q(self, state: State, goal: Goal, pol_goal: Goal, action: ActionIndex, next_state: State, reward: float, lr: float=.05) -> None:  #67 (line num in coconut source)
        bellman_update = (1 - gamma) * reward + gamma * self.q_table[next_state + goal + pol_goal].max()  #69 (line num in coconut source)
        a = (1 - lr) * self.q_table[state + goal + pol_goal][action]  #70 (line num in coconut source)
        b = lr * bellman_update  #71 (line num in coconut source)
        self.q_table[state + goal + pol_goal][action] = a + b  #72 (line num in coconut source)

# def

    def _update_usher_q(self, state: State, goal: Goal, pol_goal: Goal, action: ActionIndex, next_state: State, reward: float, lr: float=.05, p: Optional[float]=None, t_remaining=None) -> None:  #76 (line num in coconut source)

        p = self.p if p is None else p  #79 (line num in coconut source)
        if goal == pol_goal:  #80 (line num in coconut source)
            ratio = ((p + (1 - p) * self.usher_q_table[state + goal + pol_goal][action]) / (p + (1 - p) * self.usher_q_table[next_state + goal + pol_goal].max()))  #81 (line num in coconut source)
        else:  #85 (line num in coconut source)
            offset = .01 * lr  #86 (line num in coconut source)
#Not mathematically necessary, but reduces likelihood of overflow errors and stabilizes convergence a bit
            ratio = ((offset + (1 - offset) * self.usher_q_table[state + goal + pol_goal][action]) / (offset + (1 - offset) * self.usher_q_table[next_state + goal + pol_goal].max()))  #88 (line num in coconut source)
        new_lr = min(lr * ratio, 1)  #92 (line num in coconut source)
        bellman_update = (1 - gamma) * reward + gamma * self.usher_q_table[next_state + goal + pol_goal].max()  #93 (line num in coconut source)
        a = (1 - new_lr) * self.usher_q_table[state + goal + pol_goal][action]  #94 (line num in coconut source)
        b = new_lr * bellman_update  #95 (line num in coconut source)
        self.usher_q_table[state + goal + pol_goal][action] = a + b  #96 (line num in coconut source)

    def _update_pure_q(self, state: State, goal: Goal, pol_goal: Goal, action: ActionIndex, next_state: State, reward: float, lr: float=.05) -> None:  #98 (line num in coconut source)
#Not updated by HER
        bellman_update = (1 - gamma) * reward + gamma * self.pure_q_table[next_state + goal + pol_goal].max()  #101 (line num in coconut source)
        a = (1 - lr) * self.pure_q_table[state + goal + pol_goal][action]  #102 (line num in coconut source)
        b = lr * bellman_update  #103 (line num in coconut source)
        self.pure_q_table[state + goal + pol_goal][action] = a + b  #104 (line num in coconut source)


    def update(self, episode: list, lr: float=.05) -> None:  #107 (line num in coconut source)
# cumulative_r = 0#self.v_table[tuple(episode[-1][2])]
        k = self.k  #109 (line num in coconut source)
        c = 1  #110 (line num in coconut source)
        p = self.p * c  #111 (line num in coconut source)
#shorten trajectory?
        for i in range(len(episode)):  #113 (line num in coconut source)
            frame = episode[-i]  #114 (line num in coconut source)
            state, desired_goal, achieved_goal, action, next_state, reward = frame  #115 (line num in coconut source)
            self._update_pure_q(state, desired_goal, desired_goal, action, next_state, reward, lr)  #116 (line num in coconut source)
            self._update_q(state, desired_goal, desired_goal, action, next_state, reward, lr)  #117 (line num in coconut source)
            self._update_usher_q(state, desired_goal, desired_goal, action, next_state, reward, lr, p=1)  #118 (line num in coconut source)

# self._update_usher_q(state, desired_goal, desired_goal, action, next_state, reward, lr*c, p=p + (1-p)*gamma**(i))#1)
            for _ in range(k):  #121 (line num in coconut source)
# for _ in range(i, k):
                if i == 0:  #123 (line num in coconut source)
                    rand_i = 0  #124 (line num in coconut source)
                else:  #125 (line num in coconut source)
                    rand_i = np.random.geometric(1 - gamma) % i  #126 (line num in coconut source)
# rand_i = np.random.randint(len(episode) - i, len(episode))
                goal = episode[rand_i][2]  #Achieved goal  #128 (line num in coconut source)
                self._update_q(state, goal, desired_goal, action, next_state, reward, lr)  #129 (line num in coconut source)
                rand_i = np.random.geometric(1 - gamma)  #130 (line num in coconut source)
                goal = episode[rand_i][2] if rand_i < i else desired_goal  #131 (line num in coconut source)
# self._update_usher_q(state, goal, desired_goal, action, next_state, reward, lr, p=p + (1-p)*gamma**(i))
                self._update_usher_q(state, goal, desired_goal, action, next_state, reward, lr, p=gamma**i)  #133 (line num in coconut source)
# self._update_usher_q(state, goal, desired_goal, action, next_state, reward, lr, p=self.p, t_remaining = i)





index_to_action = {0: np.array([1, 0]), 1: np.array([-1, 0]), 2: np.array([0, 1]), 3: np.array([0, -1]), 4: np.array([0, 0])}  #140 (line num in coconut source)



def observe_transition(env, state: State, q: Q, policy: Callable) -> Transition:  #150 (line num in coconut source)
    action = q.sample_action(state, env.new_goal, env.new_goal, 5)  #151 (line num in coconut source)
# action = policy(state)
    env_action = index_to_action[action]  #153 (line num in coconut source)
    obs, reward, done, info = env.step(state, env_action)  #154 (line num in coconut source)
    dg = obs['desired_goal']  #155 (line num in coconut source)
    ag = obs['achieved_goal']  #156 (line num in coconut source)
    next_state = obs['observation']  #157 (line num in coconut source)
    return ((state, dg, ag, action, next_state, reward))  #158 (line num in coconut source)


def observe_episode(env, q: Q, policy: Callable) -> List[Transition]:  #161 (line num in coconut source)
    obs = env.reset()  #162 (line num in coconut source)
# ep = [observe_transition(env, obs['state'], q, policy) for _ in range(traj_steps)]
    state = obs['state']  #164 (line num in coconut source)
    ep = []  #165 (line num in coconut source)
    for _ in range(traj_steps):  #166 (line num in coconut source)
        observation = observe_transition(env, state, q, policy)  #167 (line num in coconut source)
        ep.append(observation)  #168 (line num in coconut source)
        state = observation[4]  #169 (line num in coconut source)
    return (ep)  #170 (line num in coconut source)



# exit()
def learn_q_function(k: int=4):  #175 (line num in coconut source)
    compute_reward = env.compute_reward  #176 (line num in coconut source)
    default_goal = env.new_goal  #177 (line num in coconut source)
    q = Q(env.size, compute_reward, default_goal, k)  #178 (line num in coconut source)
    ave_r = 0  #179 (line num in coconut source)
    record_num = 100  #180 (line num in coconut source)

# display_init(env, q)
    iterations = []  #183 (line num in coconut source)
    her_vals = []  #184 (line num in coconut source)
    usher_vals = []  #185 (line num in coconut source)
    q_vals = []  #186 (line num in coconut source)
    ave_r_vals = []  #187 (line num in coconut source)

    for episode in range(episodes):  #189 (line num in coconut source)
# draw_grid(env, q)
        state = env.reset()['observation']  #191 (line num in coconut source)
        power = .8  #192 (line num in coconut source)
        lr = (2**(-100 * episode / episodes) + base_lr / (episodes**power / 100 + 1))  #193 (line num in coconut source)
# lr = .1*base_lr*episodes/(.1*episodes + episode)
# lr = .05
        if episode % record_num == 0:  #196 (line num in coconut source)
            print("---------------------------")  #197 (line num in coconut source)
            print(f"Episode {episode} of {episodes}")  #198 (line num in coconut source)
            get_ave_r = lambda ep: (1 - gamma) * sum([gamma**i * ep[i][-1] for i in range(len(ep))])  #199 (line num in coconut source)
            eps = [observe_episode(env, q, lambda s: q.argmax_action(s, default_goal)) for _ in range(50)]  #200 (line num in coconut source)
# [q.update_v(ep) for ep in eps]
# ave_lr = 1/(episodes+1)#2**(-50*episode/episodes)
            ave_lr = (2**(-100 * episode / episodes) + .5 / (episodes**power / 100 + 1))  #203 (line num in coconut source)

            ave_r = ave_r * (1 - ave_lr) + ave_lr * mean([get_ave_r(ep) for ep in eps])  #205 (line num in coconut source)
            her_val = q.q_table[tuple(state) + tuple(default_goal) + tuple(default_goal)].max()  #206 (line num in coconut source)
            usher_val = q.usher_q_table[tuple(state) + tuple(default_goal) + tuple(default_goal)].max()  #207 (line num in coconut source)
            q_val = q.pure_q_table[tuple(state) + tuple(default_goal) + tuple(default_goal)].max()  #208 (line num in coconut source)

            print(f"HER q value: \t\t\t{her_val}")  #210 (line num in coconut source)
            print(f"USHER weighted q value: \t{usher_val}")  #211 (line num in coconut source)
            print(f"Pure q value: \t\t\t{q_val}")  #212 (line num in coconut source)
            print(f"Average reward: \t\t{ave_r}")  #213 (line num in coconut source)

            iterations.append(episode * record_num * traj_steps)  #215 (line num in coconut source)
            her_vals.append(her_val)  #216 (line num in coconut source)
            usher_vals.append(usher_val)  #217 (line num in coconut source)
            q_vals.append(q_val)  #218 (line num in coconut source)
            ave_r_vals.append(ave_r)  #219 (line num in coconut source)

        ep = observe_episode(env, q, lambda s: q.sample_action(s, default_goal))  #221 (line num in coconut source)

        q.update(ep, lr=lr)  #223 (line num in coconut source)

    return ({"her": her_vals[-1], "usher": usher_vals[-1], "q": q_vals[-1]})  #225 (line num in coconut source)

# plt.plot(iterations, her_vals, 	label="her_vals")
# plt.plot(iterations, usher_vals,label="usher_vals")
# plt.plot(iterations, q_vals, 	label="q_vals")
# plt.plot(iterations, ave_r_vals,label="ave_r_vals")
# plt.legend()
# plt.show()

def show_k_vals():  #238 (line num in coconut source)
    k_list = [0, 1, 2, 3, 4, 6, 8, 16]  #239 (line num in coconut source)
    output_list = {"her": [], "usher": [], "q": []}  #240 (line num in coconut source)
    for k in k_list:  #245 (line num in coconut source)
        output = learn_q_function(k)  #246 (line num in coconut source)
        for key in output_list.keys():  #247 (line num in coconut source)
            output_list[key].append(output[key])  #248 (line num in coconut source)
# output_list = {key: output[key] + output_list[key] for key in output_list.keys()}

    for key in output_list.keys():  #251 (line num in coconut source)
        plt.plot(k_list, output_list[key], label=key)  #252 (line num in coconut source)

    plt.legend()  #254 (line num in coconut source)
    plt.show()  #255 (line num in coconut source)

learn_q_function(4)  #257 (line num in coconut source)
