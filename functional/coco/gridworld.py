#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xd4248783

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

import gym  #type: ignore  #1 (line num in coconut source)
from gym.core import GoalEnv  #type: ignore  #2 (line num in coconut source)
from gym import error  #type: ignore  #3 (line num in coconut source)

import numpy as np  #5 (line num in coconut source)
import random  #6 (line num in coconut source)
import typing  #7 (line num in coconut source)
from typing import Tuple  #8 (line num in coconut source)
import pdb  #9 (line num in coconut source)
# import constants
from constants import *  #11 (line num in coconut source)
# from obstacles


noise_samples = [(1, 0), (-1, 0), (0, 1), (0, 1)]  #15 (line num in coconut source)

def state_noise(k: int) -> State:  #17 (line num in coconut source)
    return (random.sample(noise_samples + [(0, 0), ] * k, 1)[0])  #18 (line num in coconut source)



transitions = {EMPTY: lambda last_state, state: state, BLOCK: lambda last_state, state: last_state, WIND: lambda last_state, state: state + state_noise(4), RANDOM_DOOR: lambda last_state, state: state if random.random() < 0.1 else last_state}  #22 (line num in coconut source)




class GridworldEnv(GoalEnv):  #34 (line num in coconut source)
    def __init__(self, size: int, start: State, new_goal: Goal):  #35 (line num in coconut source)
        self.size = size  #36 (line num in coconut source)
        self.start = start  #37 (line num in coconut source)
        self.new_goal = new_goal  #38 (line num in coconut source)
        self.grid = np.zeros((size, size))  #39 (line num in coconut source)

        self.reward_range = (0, 1)  #41 (line num in coconut source)

    @_coconut_tco  #43 (line num in coconut source)
    def reset(self) -> ObservationDict:  #43 (line num in coconut source)
        return _coconut_tail_call(self.get_obs, self.start, self.new_goal)  #44 (line num in coconut source)

    def step(self, state: State, action: Action) -> Tuple[ObservationDict, float, bool, dict]:  #46 (line num in coconut source)
        proposed_next_state = add(state, action)  #47 (line num in coconut source)
        next_state_type = self.grid[proposed_next_state]  #48 (line num in coconut source)
        next_state = transitions[next_state_type](state, proposed_next_state)  #49 (line num in coconut source)

# print(next_state_type)
# pdb.set_trace()

# if np.abs(sub(state, next_state)).sum() > 1.05:
# 	pdb.set_trace()

        reward = self.compute_reward(next_state, self.new_goal)  #57 (line num in coconut source)
        return (self.get_obs(next_state, self.new_goal), self.compute_reward(next_state, self.new_goal), False, {})  #58 (line num in coconut source)

    def compute_reward(self, ag: Goal, dg: Goal) -> float:  #60 (line num in coconut source)
        return (1 if all([a == d for (a, d) in zip(ag, dg)]) else 0)  #61 (line num in coconut source)

    def rand_state(self) -> State:  #63 (line num in coconut source)
        return ((np.random.randint(0, self.size), np.random.randint(0, self.size)))  #64 (line num in coconut source)

    def get_obs(self, state: State, goal: Goal) -> ObservationDict:  #66 (line num in coconut source)
        return ({"state": state, "observation": state, "achieved_goal": state, "desired_goal": goal})  #67 (line num in coconut source)

def create_map_1():  #74 (line num in coconut source)
    size = 8  #75 (line num in coconut source)
    start = tuple([1, size // 2 - 1])  #76 (line num in coconut source)
    new_goal = tuple([1, size // 2 + 1])  #77 (line num in coconut source)
    gridworld = GridworldEnv(size, start, new_goal)  #78 (line num in coconut source)
    for i in range(size):  #79 (line num in coconut source)
#Borders
        gridworld.grid[0, i] = BLOCK  #81 (line num in coconut source)
        gridworld.grid[size - 1, i] = BLOCK  #82 (line num in coconut source)
        gridworld.grid[i, 0] = BLOCK  #83 (line num in coconut source)
        gridworld.grid[i, size - 1] = BLOCK  #84 (line num in coconut source)

#Wall through the middle
        gridworld.grid[i, size // 2] = BLOCK  #87 (line num in coconut source)


    gridworld.grid[1, size // 2] = RANDOM_DOOR  #90 (line num in coconut source)
    gridworld.grid[size - 2, size // 2] = EMPTY  #91 (line num in coconut source)

    return (gridworld)  #93 (line num in coconut source)
