import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
# class actor(nn.Module):
#     def __init__(self, env_params):
#         super(actor, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.action_out = nn.Linear(256, env_params['action'])

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         actions = self.max_action * torch.tanh(self.action_out(x))

#         return actions

# class critic(nn.Module):
#     def __init__(self, env_params):
#         super(critic, self).__init__()
#         self.max_action = env_params['action_max']
#         # self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.q_out = nn.Linear(256, 1)

#     def forward(self, x, actions):
#         x = torch.cat([x, actions / self.max_action], dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         q_value = self.q_out(x)

#         return q_value


# class actor(nn.Module):
#     def __init__(self, env_params):
#         super(actor, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.norm1 = nn.BatchNorm1d(env_params['obs'] + env_params['goal'])
#         self.norm2 = nn.BatchNorm1d(256)
#         self.norm3 = nn.BatchNorm1d(256)
#         # self.norm1 = nn.LayerNorm(256)
#         # self.norm2 = nn.LayerNorm(256)
#         # self.norm3 = nn.LayerNorm(256)
#         self.action_out = nn.Linear(256, env_params['action'])
#         # self.action_out.weight.data.fill_(0)
#         # self.action_out.bias.data.fill_(0)
        

#     def forward(self, x):
#         x = F.relu(self.fc1(self.norm1(x)))
#         x = F.relu(self.fc2(self.norm2(x)))
#         x = F.relu(self.fc3(self.norm3(x)))
#         # x = F.relu(self.norm1(self.fc1(x)))
#         # x = F.relu(self.norm2(self.fc2(x)))
#         # x = F.relu(self.norm3(self.fc3(x)))
#         actions = self.max_action * torch.tanh(self.action_out(x))

#         return actions

# class critic(nn.Module):
#     def __init__(self, env_params):
#         super(critic, self).__init__()
#         self.max_action = env_params['action_max']
#         # self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.norm1 = nn.BatchNorm1d(env_params['obs'] + env_params['goal']  + env_params['action'])
#         self.norm2 = nn.BatchNorm1d(256)
#         self.norm3 = nn.BatchNorm1d(256)
#         # self.norm1 = nn.LayerNorm(256)
#         # self.norm2 = nn.LayerNorm(256)
#         # self.norm3 = nn.LayerNorm(256)
#         self.q_out = nn.Linear(256, 1)
#         # self.q_out.weight.data.fill_(0)
#         # self.q_out.bias.data.fill_(0)

#     def forward(self, x, actions):
#         x = torch.cat([x, actions / self.max_action], dim=1)
#         # x = F.relu(self.norm1(self.fc1(x)))
#         # x = F.relu(self.norm2(self.fc2(x)))
#         # x = F.relu(self.norm3(self.fc3(x)))
#         x = F.relu(self.fc1(self.norm1(x)))
#         x = F.relu(self.fc2(self.norm2(x)))
#         x = F.relu(self.fc3(self.norm3(x)))
#         q_value = self.q_out(x)

#         return q_value





class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x, deterministic=True):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


    def set_normalizers(self, o, g): 
        self.o_norm = o
        self.g_norm = g

    # def normed_forward(self, obs, g, deterministic=False): 
    #     obs_norm = self.o_norm.normalize(obs)
    #     g_norm = self.g_norm.normalize(g)
    #     # concatenate the stuffs
    #     inputs = np.concatenate([obs_norm, g_norm])
    #     inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    #     return self.forward(inputs, deterministic=deterministic)

    def _get_norms(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return obs_norm, g_norm

    def _get_denorms(self, obs, g):
        obs_denorm = self.o_norm.denormalize(obs)
        g_denorm = self.g_norm.denormalize(g)
        return obs_denorm, g_denorm

    def normed_forward(self, obs, g, deterministic=False): 
        obs_norm, g_norm = self._get_norms(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        # concatenate the stuffs
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)
        return self.forward(inputs)


class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.goal_dim = env_params['goal']
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, x[...,-self.goal_dim:], actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class usher_critic(nn.Module):
    def __init__(self, env_params):
        super(usher_critic, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        # self.her_goal_range = (env_params['obs'] + env_params['goal'] -1, env_params['obs'] + 2*env_params['goal']-1)
        self.her_goal_range = (env_params['obs'] , env_params['obs'] + env_params['goal'])
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        mult_val = torch.ones_like(x)
        # mult_val[...,:-4] /= 1#2**.5
        # her_goal_scale =  1/10
        # policy_goal_scale =  2
        her_goal_scale =  1
        policy_goal_scale =  1
        # diff_scale = 2
        # mult_val[...,self.her_goal_range[0]:self.her_goal_range[1]] *= her_goal_scale
        # mult_val[...,self.her_goal_range[1]:] *= policy_goal_scale
        # x[...,self.her_goal_range[1]:] = diff_scale*(x[...,self.her_goal_range[1]:] - x[...,self.her_goal_range[0]:self.her_goal_range[1]])
            #[policy_goal] = scale*([policy_goal] - [HER goal])
            #Makes the difference between goals more explicit and hopefully more learnable by seperatinig them more in space
        x = torch.cat([x*mult_val, actions / self.max_action], dim=1)
        # x = torch.cat([x[...,:-2], actions / self.max_action], dim=1)
        # x = torch.cat([x[...,:-2], actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value




class ratio_critic(nn.Module):
    def __init__(self, env_params):
        super(ratio_critic, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        # self.her_goal_range = (env_params['obs'] + env_params['goal'] -1, env_params['obs'] + 2*env_params['goal']-1)
        self.her_goal_range = (env_params['obs'] , env_params['obs'] + env_params['goal'])
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.p_out = nn.Linear(256, 1)

    def forward(self, x, actions, return_p=False):
        mult_val = torch.ones_like(x)
        # mult_val[...,:-4] /= 1#2**.5
        # her_goal_scale =  1/10
        # policy_goal_scale =  2
        her_goal_scale =  1
        policy_goal_scale =  1
        # diff_scale = 2
        # mult_val[...,self.her_goal_range[0]:self.her_goal_range[1]] *= her_goal_scale
        # mult_val[...,self.her_goal_range[1]:] *= policy_goal_scale
        # x[...,self.her_goal_range[1]:] = diff_scale*(x[...,self.her_goal_range[1]:] - x[...,self.her_goal_range[0]:self.her_goal_range[1]])
            #[policy_goal] = scale*([policy_goal] - [HER goal])
            #Makes the difference between goals more explicit and hopefully more learnable by seperating them more in space
        x = torch.cat([x*mult_val, actions / self.max_action], dim=1)
        # x = torch.cat([x[...,:-2], actions / self.max_action], dim=1)
        # x = torch.cat([x[...,:-2], actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        if return_p: 
            #exponentiate p to ensure it's non-negative
            exp = .5 #exponent for p
            base = 4 #Initially give states small probability. 
                # If they're not visited, they won't be updated, so they should remain small
                # States that are visited will grow, which is what we want
            p_value =  2**(exp*self.p_out(x) - base)
            # p_value =  self.p_out(x) #+ 1
            return q_value, p_value
        else: 
            return q_value


class T_conditioned_ratio_critic(nn.Module):
    def __init__(self, env_params):
        super(T_conditioned_ratio_critic, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + 1 + env_params['action'], 256)
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        # self.her_goal_range = (env_params['obs'] + env_params['goal'] -1, env_params['obs'] + 2*env_params['goal']-1)
        self.her_goal_range = (env_params['obs'] , env_params['obs'] + env_params['goal'])
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.p_out = nn.Linear(256, 1)

    def forward(self, x, T, actions, return_p=False):
        mult_val = torch.ones_like(x)
        new_x = torch.cat([x*mult_val, T, actions / self.max_action], dim=1)
        assert new_x.shape[0] == x.shape[0] and new_x.shape[-1] == (x.shape[-1] + 1 + self.env_params['action'])
        x = new_x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        if return_p: 
            #exponentiate p to ensure it's non-negative
            exp = .5 #exponent for p
            base = 4 #Initially give states small probability. 
                # If they're not visited, they won't be updated, so they should remain small
                # States that are visited will grow, which is what we want
            p_value =  2**(exp*self.p_out(x) - base)
            # p_value =  self.p_out(x) #+ 1
            return q_value, p_value
        else: 
            return q_value




class value_prior_actor(nn.Module):
    def __init__(self, env_params):
        super(value_prior_actor, self).__init__()
        self.max_action = env_params['action_max']
        self.goal_dim = env_params['goal']
        self.trim_ag = True
        if self.trim_ag: 
            self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
            self.skip_layer = nn.Linear(env_params['obs'] + env_params['goal'], env_params['action'])
        else:
            self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'], 256)
            self.skip_layer = nn.Linear(env_params['obs'] + 2*env_params['goal'], env_params['action'])
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        if self.trim_ag: 
            x = x[...,:-self.goal_dim]
        skip_val = self.skip_layer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x) + skip_val)

        return actions


    def set_normalizers(self, o, g): 
        self.o_norm = o
        self.g_norm = g

    # def normed_forward(self, obs, g, deterministic=False): 
    #     obs_norm = self.o_norm.normalize(obs)
    #     g_norm = self.g_norm.normalize(g)
    #     # concatenate the stuffs
    #     inputs = np.concatenate([obs_norm, g_norm])
    #     inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    #     return self.forward(inputs, deterministic=deterministic)

    def _get_norms(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return obs_norm, g_norm

    def _get_denorms(self, obs, g):
        obs_denorm = self.o_norm.denormalize(obs)
        g_denorm = self.g_norm.denormalize(g)
        return obs_denorm, g_denorm

    def normed_forward(self, obs, g, ag, deterministic=False): 
        obs_norm, g_norm = self._get_norms(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        ag_norm = self.g_norm.normalize(torch.tensor(ag, dtype=torch.float32))
        # concatenate the stuffs
        inputs = torch.cat([obs_norm, g_norm, ag_norm])
        inputs = inputs.unsqueeze(0)
        return self.forward(inputs)


class value_prior_critic(nn.Module):
    def __init__(self, env_params):
        super(value_prior_critic, self).__init__()
        self.max_action = env_params['action_max']
        self.goal_dim = env_params['goal']
        self.gamma = env_params['gamma']
        self.trim_ag = True
        if self.trim_ag: 
            self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
            self.skip_layer = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 1)
            self.skip_layer_dist = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], self.goal_dim)
        else:
            self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
            self.skip_layer = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 1)
            self.skip_layer_dist = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], self.goal_dim)
        # self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        # self.q_out = nn.Linear(256, 1)
        self.mult_out = nn.Linear(256, 1)
        self.add_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        g = x[...,-self.goal_dim:]
        ag = x[...,-2*self.goal_dim:-self.goal_dim]
        if self.trim_ag: 
            x = x[...,:-self.goal_dim]
        x = torch.cat([x, actions / self.max_action], dim=1)
        skip_val = self.skip_layer(x)
        skip_val_dist = self.skip_layer_dist(x)
        dist = ((g-ag + skip_val_dist)**2).sum(dim=-1)**.5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # q_value = self.q_out(x)
        gamma = self.gamma
        # q_value = 1/gamma*(gamma**(skip_val)-1) + self.add_out(x)
        # q_value = 1/gamma*(gamma**(5*dist*F.elu(self.mult_out(x)))-1) + self.add_out(x)
        q_value = 1/gamma*(gamma**(dist/5)-1) + self.add_out(x) + skip_val

        return q_value

class StateValueEstimator(nn.Module):
    def __init__(self, actor, critic, gamma):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.gamma = gamma

    def q2time(self, q):
        # max_q = 1/(1-self.args.gamma)
        # ratio = -.99*torch.clip(q/max_q, -1,0) #.99 for numerical stability
        return torch.log(1+q*(1-self.gamma)*.998)/torch.log(torch.tensor(self.gamma))

    def forward(self, o: torch.Tensor, g: torch.Tensor, norm=True): 
        assert type(o) == torch.Tensor
        assert type(g) == torch.Tensor
        if norm: 
            obs_norm, g_norm = self.actor._get_norms(o,g)
        else: 
            obs_norm, g_norm = o, g
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)

        action = self.actor(inputs)
        value = self.critic(inputs, action).squeeze()

        # return self.q2time(value)
        return value
