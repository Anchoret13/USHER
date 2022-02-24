import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Tuple
from gym.core import Wrapper
from .env_generator import EnvironmentCollection

from math import pi, cos, sin
import numpy as np
import math

#from gym.envs.classic_control.rendering  import make_circle, Transform
from gym_extensions.continuous.gym_navigation_2d import gym_rendering  
import os
import logging 
import pdb

from typing import NewType, TypeVar, Tuple, Dict

# UnscaledObs = NewType('UnscaledObs', np.ndarray)
# ScaledObs = NewType('ScaledObs', np.ndarray)
# UnscaledGoal = NewType('UnscaledGoal', np.ndarray)
# ScaledGoal = NewType('ScaledGoal', np.ndarray)
State = NewType('State', dict)
Action = NewType('Action', np.ndarray)
Unscaled = NewType('Unscaled', np.ndarray)
Scaled = NewType('Scaled', np.ndarray)

low = -100
high = 600

def scale(x, x_low, x_high, target_low=-1, target_high=1):
    unit_x = (x-x_low)/(x_high-x_low)
    target_x = unit_x*(target_high-target_low) + target_low
    return target_x

def scale_down(x: Unscaled) -> Scaled:
    return scale(x, low, high, -1, 1)
    # return Scaled(scale(x, low, high, -1, 1))

def scale_up(x: Scaled) -> Unscaled:
    return scale(x, -1, 1, low, high)
    # return Unscaled(scale(x, -1, 1, low, high))

def extend_box(box, low, high):
    new_low =  np.append(box.low, low)
    new_high =  np.append(box.high, high)
    return spaces.Box(new_low, new_high, dtype='float32')


# class Dynamics:
#     """docstring for Dynamics"""
#     def __init__(self, arg):
#         self.arg = arg
#         self.low = []
#         self.high = []

#     def move(self, s: State, a: Action) -> State: 
#         assert False, "Not implemented yet"
        
#     def dyn_obs(self, s: State) -> Tuple[Unscaled, Unscaled]: 
#         assert False, "Not implemented yet"


class SimpleDynamicsEnv:#(gym.GoalEnv):
    def __init__(self):
        self.action_dim = 2
        self.state_dim = 2
        self.goal_dim = 2

        # self.low = 100
        # self.high = 400
        self.low = -100
        self.high = 600
        # self.low = -1
        # self.high = 1

        self.xy_low = np.array([self.low, self.low], dtype='float32')
        self.xy_high = np.array([self.high, self.high], dtype='float32')
        max_time = 10
        # self.dt = max_time
        max_speed = (self.xy_high[0] - self.xy_low[0])/max_time
        self.dt = max_speed
        self.threshold = (self.high - self.low)/20#max_speed*2

        self.action_space   =spaces.Box(-1, 1, shape=(self.action_dim,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal    =spaces.Box(self.xy_low, self.xy_high, shape= (self.goal_dim,), dtype='float32'),
            achieved_goal   =spaces.Box(self.xy_low, self.xy_high, shape= (self.goal_dim,), dtype='float32'),
            observation     =spaces.Box(self.xy_low, self.xy_high, shape=(self.state_dim,), dtype='float32'),
        ))

    def reset(self): 
        self.goal  = self.observation_space['desired_goal'].sample()
        # state = self.observation_space['observation' ].sample()
        state = np.random.randn(2)*20
        self.position = state[:2]
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Scaled]:
        unscaled_observation = Unscaled(self.position)
        assert type(unscaled_observation) == np.ndarray
        assert unscaled_observation.shape == (self.state_dim,)
        assert self.position.shape == (2,)

        observation = scale_down(unscaled_observation)#, self.xy_low, self.xy_high)
        dg = scale_down(self.goal)#scale(self.goal, self.xy_low[:self.goal_dim], self.xy_high[:self.goal_dim])
        ag = Scaled(observation[:self.goal_dim])
        obs = {'observation': observation, 'achieved_goal': ag, 'desired_goal': dg}
        return obs


    def _goal_distance(self, a: Unscaled, b: Unscaled) -> np.ndarray: 
        return ((a-b)**2).sum(axis=-1)**.5 

    def is_close(self, state1: Scaled, state2: Scaled, info=None) -> np.ndarray:
        # unscaled_s1 = scale(state1, -1, 1, self.low, self.high)
        # unscaled_s2 = scale(state2, -1, 1, self.low, self.high)
        unscaled_s1 = scale_up(state1)
        unscaled_s2 = scale_up(state2)
        # reward = (self._goal_distance(state1, state2) < self.threshold) - 1
        return (self._goal_distance(unscaled_s1, unscaled_s2) < self.threshold) 

    def compute_reward(self, state1: Scaled, state2: Scaled, info=None) -> np.ndarray:
        return self.is_close(state1, state2) - 1


    def step(self, action: Action) -> Tuple[Scaled, float, bool, dict]:
        new_position = np.clip(self.position + action*self.dt, a_min=self.low, a_max=self.high)
        self.position = new_position

        obs = self._get_obs()
        ag = obs['achieved_goal']
        dg = obs['desired_goal']
        reward = self.compute_reward(ag, dg)
        is_success = reward > -.5
        info = {'is_success': is_success}

        return obs,reward, False, info

class LimitedRangeBasedPOMDPNavigation2DEnv(gym.GoalEnv):
    def __init__(self,
                 worlds_pickle_filename=os.path.join(os.path.dirname(__file__), "assets", "worlds_640x480_v0.pkl"),
                 world_idx=0,
                 initial_position = np.array([-20.0, -20.0]),
                 destination = np.array([520.0, 400.0]),
                 max_observation_range = 100.0,
                 destination_tolerance_range=50.0,
                 add_self_position_to_observation=False,
                 add_goal_position_to_observation=False):

        self.env = SimpleDynamicsEnv()

        worlds = EnvironmentCollection()
        worlds.read(worlds_pickle_filename)

        self.world = worlds.map_collection[world_idx]
        self.destination = destination

        self.num_beams = 16
        self.max_observation_range = self.env.dt#*1.5
        self.extension_low = [0]*self.num_beams + [-1]
        self.extension_high = [np.log(self.max_observation_range+1)]*self.num_beams + [1]
        obs_obs_space = extend_box(self.env.observation_space['observation'], low, high)
        self.observation_space = spaces.Dict(dict(
            desired_goal    =self.env.observation_space['desired_goal'],
            achieved_goal   =self.env.observation_space['achieved_goal'],
            observation     =obs_obs_space,
        ))
        # self.observation_space = self.env.observation_space
        # self.observation_space['observation'] = obs_obs_space
        self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space

        self.compute_reward = self.env.compute_reward
        # self.reset = self.env.reset
        # self.step = self.env.step

        # self.threshold = .1

    def reset(self) -> Scaled: 
        self.collided = False
        observation = self.env.reset()
        destination = scale_up(observation['desired_goal'])
        while not self.world.point_is_in_free_space(destination[0], destination[1], epsilon=0.25):
            observation = self.env.reset()
            destination = scale_up(observation['desired_goal'])
            # self.destination = sample_goal()

        # obs = np.append(observation['observation'], [1 if self.collided else -1])
        # observation['observation'] = obs
        # assert observation['observation'].shape == (3,)
        observation = self.extend_observation(observation)
        self.last_position = observation
        return observation

    def step(self, action) -> Scaled: 
        observation, reward, done, info = self.env.step(action)
        collided = False
        new_state = scale_up(observation['achieved_goal'])
        old_state = scale_up(self.last_position['achieved_goal'])
        # pdb.set_trace()
        if not self.world.point_is_in_free_space(new_state[0], new_state[1], epsilon=0.25):
            collided = True
        elif not self.world.segment_is_in_free_space(old_state[0], old_state[1],
                                                   new_state[0], new_state[1],
                                                   epsilon=0.25):
            collided = True
    # collided = False
        # if collided: print("collided")
        break_chance = 0
        if collided == True:
            # collided = True if np.random.rand() < break_chance else False
            if np.random.rand() < break_chance:
                self.collided = collided or self.collided

        # self.collided = collided or self.collided

        # if self.collided:
        if collided:# or self.collided:
            return self.last_position, reward, done, info
        else: 
            observation = self.extend_observation(observation)
            self.last_position = observation
            return observation, reward, done, info

        # obs = np.append(observation['observation'], [1 if self.collided else -1])
        # observation['observation']= obs
        # assert observation['observation'].shape == (3,)


    def extend_observation(self, observation) -> Tuple[Scaled, Scaled]:
        ag = observation['achieved_goal']
        assert (ag >= -1).all()
        assert (ag <= 1).all()
        unscaled_ag = scale_up(ag)
        # ag = scale_down(unscaled_ag)
        # obs = scale_down(unscaled_obs)

        delta_angle = 2*pi/self.num_beams if self.num_beams > 0 else 0
        ranges = [self.world.raytrace(unscaled_ag,
                                      i * delta_angle,
                                      self.max_observation_range,
                                      n_evals=50) for i in range(self.num_beams)]
        ranges = [r - self.extension_high[0] for r in ranges]
        ranges +=  [1 if self.collided else -1]

        obs = np.append(observation['observation'], ranges)
        observation['observation'] = obs
        return observation

    # def compute_reward(self, state1: Scaled, state2: Scaled, info=None) -> np.ndarray:
    #     return self.is_close(state1, state2) - 1