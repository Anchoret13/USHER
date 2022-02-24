from her_types import * #type: ignore
from rl_modules.typed_usher_with_ratio import ddpg_agent #type: ignore
from arguments import get_args #type: ignore
import gym #type: ignore

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    env = gym.make(args.env_name)
    # env = ActionRandomnessWrapper(env, args.action_noise)
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment 
    # step_method = env.step
    # env.step = lambda action: step_method(
    #         action + np.random.normal(
    #             loc=[0]*env_params['action'], 
    #             scale=[args.action_noise]*env_params['action']
    #             )
    #         )

    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()
    return ddpg_trainer

if __name__ == '__main__':
    args = get_args()
    agent = launch(args)