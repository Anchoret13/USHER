import numpy as np
import gym
import os, sys
from mpi4py import MPI
import random
import torch
import itertools
# from rl_modules.multi_goal_env2 import *
from HER_mod.arguments import get_args
from environments.velocity_env import MultiGoalEnvironment, CarEnvironment
from environments.car_env import RotationEnv, NewCarEnv, SimpleMovementEnvironment
from environments.torus_env import Torus
# from environments.continuous_acrobot import ContinuousAcrobotEnv

import pickle

# from continuous_gridworld import create_map_1, random_map

from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment#, EnvironmentCollection, Obstacle

from gym.wrappers.time_limit import TimeLimit

import pickle
from action_randomness_wrapper import ActionRandomnessWrapper, RepeatedActionWrapper


"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

LOGGING = True
seed = True

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    # print(params)
    return params

def launch(args, time=True, hooks=[], vel_goal=False, seed=True):
    # create the ddpg_agent
    # if args.env_name == "MultiGoalEnvironment":
    #     env = MultiGoalEnvironment("MultiGoalEnvironment", time=time, vel_goal=vel_goal)
    # elif args.env_name == "PendulumGoal":
    #     env = TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200)
    # else:
    #     env = gym.make(args.env_name)

    if args.env_name == "MultiGoalEnvironment":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=False)
    elif args.env_name == "MultiGoalEnvironmentVelGoal":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=True)
    elif args.env_name == "Car":
        # env = CarEnvironment("CarEnvironment", time=True, vel_goal=False)
        env = TimeLimit(NewCarEnv(vel_goal=False), max_episode_steps=50)
        # env = TimeLimit(CarEnvironment("CarEnvironment", time=True, vel_goal=False), max_episode_steps=50)
    elif "Gridworld" in args.env_name: 
        # from continuous_gridworld import create_map_1#, random_blocky_map, two_door_environment, random_map
        # from alt_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map #create_map_1,
        from environments.solvable_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map #create_map_1,
        # from gridworld_reimplementation import random_map

        max_steps = 50 if "Alt" in args.env_name else 20
        if args.env_name == "TwoDoorGridworld":
            env=TimeLimit(two_door_environment(), max_episode_steps=50)
        else:
            if "RandomBlocky" in args.env_name:
                mapmaker = random_blocky_map
            elif "Random" in args.env_name:
                mapmaker = random_map
            elif "Test" in args.env_name: 
                mapmaker = create_test_map
            else: 
                mapmaker = create_map_1

            if "Asteroids" in args.env_name: 
                env_type="asteroids"
            elif "StandardCar" in args.env_name:
                env_type = "standard_car"
            elif "Car" in args.env_name:
                env_type = "car"
            else: 
                env_type = "linear"
            print(f"env type: {env_type}")
            env = TimeLimit(mapmaker(env_type=env_type), max_episode_steps=max_steps)
    elif "AsteroidsVelGoal" in args.env_name:
        env = TimeLimit(RotationEnv(vel_goal=True), max_episode_steps=50)
    elif "Asteroids" in args.env_name:
        env = TimeLimit(RotationEnv(vel_goal=False), max_episode_steps=50)
    elif "SimpleMovement" in args.env_name:
        env = TimeLimit(SimpleMovementEnvironment(vel_goal=False), max_episode_steps=50)
    elif "Torus" in args.env_name:
        freeze = "Freeze" in args.env_name or "freeze" in args.env_name
        if freeze: 
            n = args.env_name[len("TorusFreeze"):]
        else: 
            n = args.env_name[len("Torus"):]
        try: 
            dimension = int(n)
        except:
            print("Could not parse dimension. Using n=2")
            dimension=2
        print(f"Dimension = {dimension}")
        print(f"Freeze = {freeze}")
        env = TimeLimit(Torus(dimension, freeze), max_episode_steps=50)
    elif "2DNav" in args.env_name or "2Dnav" in args.env_name: 
        env = gym.make("Limited-Range-Based-Navigation-2d-Map4-Goal0-v0")
        env._max_episode_steps=50
    else:
        env = gym.make(args.env_name)

    print(f"Using environment {env}")
    env = ActionRandomnessWrapper(env, args.action_noise)
    # env =  RepeatedActionWrapper(env, 5)
    # env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    # env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    # env = MultiGoalEnvironment("MultiGoalEnvironment", time=time, vel_goal=vel_goal)#, epsilon=.1/4) 
    # problem = doubleIntegratorTest()
    # problem = pendulumTest()
    # env = PlanningEnvGymWrapper(problem)
    # env = KinomaticGymWrapper(problem)
    # set random seeds for reproduce
    if seed: 
        try: 
            env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        except: 
            pass
        random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
        if args.cuda:
            torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # return
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params, vel_goal=vel_goal)
    # if vel_goal: 
    #     ddpg_trainer = ddpg_agent(args, env, env_params, vel_goal=vel_goal)
    # else: 
    #     ddpg_trainer = her_ddpg_agent(args, env, env_params)
    # pdb.set_trace()
    ddpg_trainer.learn(hooks)
    # [hook.finish() for hook in hooks]
    return ddpg_trainer, [hook.finish() for hook in hooks]




if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()

    # agent = launch(args, time=False, hooks=[])#hooks=[DistancePlottingHook()])
    # agent = launch(args, time=True, hooks=[DistancePlottingHook(), PlotPathCostsHook(args)], vel_goal=True)
    # agent = launch(args, time=True, hooks=[DistancePlottingHook(), PlotPathCostsHook(args)], vel_goal=False)
    # try:
    hook_list = []


    train_pos_agent = lambda : launch(args, time=True, hooks=[], vel_goal=False, seed=False)[0]
    train_vel_agent = lambda : launch(args, time=True, hooks=[], vel_goal=True, seed=False)[0]

    from HER_mod.rl_modules.new_ratio_agent import ddpg_agent
    agent, run_times = launch(args, time=True, hooks=[], vel_goal=False, seed=False)
    suffix = ""


    
    n = 100
    evs = [agent._eval_agent() for _ in range(n)]
    success_rate = sum([evs[i]['success_rate'] for i in range(n)])/n
    reward_rate = sum([evs[i]['reward_rate'] for i in range(n)])/n
    value_rate = sum([evs[i]['value_rate'] for i in range(n)])/n
    if LOGGING and MPI.COMM_WORLD.Get_rank() == 0:
        # pdb.set_trace()
        log_file_name = f"logging/{args.env_name}.txt"
        # success_rate = sum([agent._eval_agent()[0] for _ in range(n)])/n
        text = f"action_noise: {args.action_noise}, "   
        text +=f"\ttwo_goal: {args.two_goal}, \n"            
        text +=f"\tsuccess_rate: {success_rate}\n"         
        text +=f"\taverage_reward: {reward_rate}\n"        
        text +=f"\taverage_initial_value: {value_rate}\n"  
        text +="\n"

        with open(log_file_name, "a") as f:
            f.write(text)

        print("Log written")

