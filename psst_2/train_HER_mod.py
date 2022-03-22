import numpy as np
import gym
import os, sys
from mpi4py import MPI
import random
import torch
import itertools
# from rl_modules.multi_goal_env2 import *
from HER_mod.arguments import get_args
# from HER_mod.rl_modules.ddpg_agent import ddpg_agent
# from HER_mod.rl_modules.value_prior_agent import ddpg_agent
# from HER.rl_modules.her_ddpg_agent import her_ddpg_agent
from HER_mod.rl_modules.velocity_env import *
from HER_mod.rl_modules.car_env import *
from HER_mod.rl_modules.continuous_acrobot import ContinuousAcrobotEnv
# from pomp.planners.plantogym import *
from HER_mod.rl_modules.value_map import *
from HER_mod.rl_modules.hooks import *
from HER_mod.rl_modules.models import StateValueEstimator
from HER_mod.rl_modules.tsp import *
from HER_mod.rl_modules.get_path_costs import get_path_costs, get_random_search_costs, method_comparison

from pomp.planners.plantogym import PlanningEnvGymWrapper, KinomaticGymWrapper
from pomp.example_problems.doubleintegrator import doubleIntegratorTest
from pomp.example_problems.dubins import dubinsCarTest
from pomp.example_problems.pendulum import pendulumTest

# from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
# from pomp.example_problems.robotics.fetch.push import FetchPushEnv
# from pomp.example_problems.robotics.fetch.slide import FetchSlideEnv
# from pomp.example_problems.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv


# from continuous_gridworld import create_map_1, random_map
from torus_env import Torus

from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment#, EnvironmentCollection, Obstacle

from pomp.example_problems.gym_pendulum_baseenv import PendulumGoalEnv
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
        from solvable_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map #create_map_1,
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
        # if args.env_name == "Gridworld" :
        #     env = TimeLimit(create_map_1(), max_episode_steps=50)
        # elif args.env_name == "RandomGridworld" :
        #     env = TimeLimit(random_map(), max_episode_steps=50)
        # elif args.env_name == "RandomGridworld" :
        #     env = TimeLimit(random_blocky_map(), max_episode_steps=50)
        # elif args.env_name == "AsteroidsGridworld" :
        #     env = TimeLimit(create_map_1(env_type="asteroids"), max_episode_steps=50)
        # elif args.env_name == "AsteroidsRandomGridworld" :
        #     env = TimeLimit(random_map(env_type="asteroids"), max_episode_steps=50)
        # elif args.env_name == "CarGridworld" :
        #     env = TimeLimit(create_map_1(env_type="car"), max_episode_steps=50)
        # elif args.env_name == "CarRandomGridworld" :
        #     env = TimeLimit(random_map(env_type="car"), max_episode_steps=50)
        # else: 
        #     print(f"No environment with the name {args.env_name}")
        #     raise Exception
    elif "AsteroidsVelGoal" in args.env_name:
        env = TimeLimit(RotationEnv(vel_goal=True), max_episode_steps=50)
    elif "Asteroids" in args.env_name:
        env = TimeLimit(RotationEnv(vel_goal=False), max_episode_steps=50)
    elif "SimpleMovement" in args.env_name:
        env = TimeLimit(SimpleMovementEnvironment(vel_goal=False), max_episode_steps=50)
    elif args.env_name == "PendulumGoal":
        env = TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200)
    elif "FetchReach" in args.env_name:
        env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    elif "FetchPush" in args.env_name:
        env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    elif "FetchSlide" in args.env_name:
        env = TimeLimit(FetchSlideEnv(), max_episode_steps=50)
    elif "FetchPickAndPlace" in args.env_name:
        env = TimeLimit(FetchPickAndPlaceEnv(), max_episode_steps=50)
    elif "ContinuousAcrobot" in args.env_name:
        env = TimeLimit(ContinuousAcrobotEnv(), max_episode_steps=50)
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
    hook_list = [
                # ValueMapHook(target_vels=[[0,0], [.5/2**.5,.5/2**.5]]),
                # DiffMapHook(), 
                # # EmpiricalVelocityValueMapHook(),
                # VelocityValueMapHook(), 
                # GradientDescentShortestPathHook(),
                # GradientDescentShortestPathHook(gd_steps=5),
                # GradientDescentShortestPathHook(gd_steps=10),
                # GradientDescentShortestPathHook(gd_steps=15),
                # GradientDescentShortestPathHook(args=([0, 5,10,20,40], False)),
                # # GradientDescentShortestPathHook(args=([0,5,10,20,40], True)),
                # PlotPathCostsHook()
                ]
    # hook_list = []
    pos_hook_list = [#DiffMapHook(), 
                ValueMapHook(target_vels=[[0,0]]),#target_vels=[[0,0], [.5/2**.5,.5/2**.5]]),
                # GradientDescentShortestPathHook(),
                # GradientDescentShortestPathHook(gd_steps=5),
                # GradientDescentShortestPathHook(gd_steps=10),
                # GradientDescentShortestPathHook(gd_steps=15),
                GradientDescentShortestPathHook(args=([-1], False)),
                # GradientDescentShortestPathHook(args=([-1], True)),
                # PlotPathCostsHook()
                ]
    vel_hook_list = [
                GradientDescentShortestPathHook(args=([0,5,10,20], False)),
                GradientDescentShortestPathHook(args=([0,5,10,20], True)),
                PlotPathCostsHook()
                ]

    # hook_list = []
    # pos_hook_list = []
    # vel_hook_list = []

    train_pos_agent = lambda : launch(args, time=True, hooks=[], vel_goal=False, seed=False)[0]
    train_vel_agent = lambda : launch(args, time=True, hooks=[], vel_goal=True, seed=False)[0]
    # get_path_costs(train_pos_agent, train_vel_agent)
    # train_pos_agent()
    # train_vel_agent()
    # for i in range(10):
    #     args.seed += 1
        # agent, run_times = launch(args, time=True, hooks=hook_list, vel_goal=True, seed=False)
    # agent, run_times = launch(args, time=True, hooks=hook_list, vel_goal=True, seed=False)

    if args.p2p: 
        # if "Fetch" in args.env_name:
        #     from HER.rl_modules.fetch_specific_p2p import ddpg_agent
        # else: 
        #     from HER.rl_modules.p2p_agent import ddpg_agent
        agent, run_times = launch(args, time=True, hooks=[], vel_goal=True, seed=False)
        suffix = "_p2p"
    else: 
        # if args.two_goal:
        #     # from HER_mod.rl_modules.usher_agent import ddpg_agent
        #     from HER_mod.rl_modules.two_goal_usher import ddpg_agent
        # else:
        #     # from HER_mod.rl_modules.ddpg_agent import ddpg_agent
        #     from HER_mod.rl_modules.sac import ddpg_agent
        # if args.apply_ratio: 
        #     # from HER_mod.rl_modules.true_ratio_two_goal_usher import ddpg_agent
        #     from HER_mod.rl_modules.t_conditioned_two_goal_usher import ddpg_agent
        # else:
        #     from HER_mod.rl_modules.two_goal_usher import ddpg_agent
        # from HER_mod.rl_modules.unit_reward_usher import ddpg_agent

        # from HER_mod.rl_modules.t_conditioned_two_goal_usher import ddpg_agent
        from HER_mod.rl_modules.new_ratio_agent import ddpg_agent
        # from HER_mod.rl_modules.delta_ddpg import ddpg_agent

        # from HER_mod.rl_modules.new_ratio_unit_reward_agent import ddpg_agent



        # from HER_mod.rl_modules.sac import ddpg_agent
        # from HER.rl_modules.sac_agent import ddpg_agent
        agent, run_times = launch(args, time=True, hooks=[], vel_goal=False, seed=False)
        suffix = ""


    # with open("saved_models/her_mod_" + args.env_name + suffix + ".pkl", 'wb') as f:
    #     pickle.dump(agent.actor_network, f)
    #     print("Saved agent")

    # value_estimator = StateValueEstimator(agent.actor_network, agent.critic.critic_1, args.gamma)

    # with open("saved_models/her_mod_" + args.env_name + "_value" + suffix + ".pkl", 'wb') as f:
    #     pickle.dump(value_estimator, f)
    #     print("Saved value estimator")
    with open("saved_models/her_" + args.env_name + suffix + ".pkl", 'wb') as f:
        pickle.dump(agent.actor_network, f)
        print("Saved agent")

    value_estimator = StateValueEstimator(agent.actor_network, agent.critic.critic_1, args.gamma)

    with open("saved_models/her_" + args.env_name + "_value" + suffix + ".pkl", 'wb') as f:
        pickle.dump(value_estimator, f)
        print("Saved value estimator")




    
    n = 100
    evs = [agent._eval_agent() for _ in range(n)]
    # success_rate = sum([agent._eval_agent() for _ in range(n)])/n
    # for _ in range(3):
    #     agent.learn(1)
    #     evs += [agent._eval_agent() for _ in range(n)]
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

