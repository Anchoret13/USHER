# (generated with --quick)

import her_types
from typing import Any, Dict, Optional, Tuple, Type

ActionArray: Type[her_types.ActionArray]
ActionTensor: Type[her_types.ActionTensor]
Array: Type[her_types.Array]
GoalArray: Type[her_types.GoalArray]
GoalTensor: Type[her_types.GoalTensor]
MPI: Any
NormalActionArray: Type[her_types.NormalActionArray]
NormalActionTensor: Type[her_types.NormalActionTensor]
NormalGoalArray: Type[her_types.NormalGoalArray]
NormalGoalTensor: Type[her_types.NormalGoalTensor]
NormalObsArray: Type[her_types.NormalObsArray]
NormalObsTensor: Type[her_types.NormalObsTensor]
NormedArray: Type[her_types.NormedArray]
NormedTensor: Type[her_types.NormedTensor]
ObsArray: Type[her_types.ObsArray]
ObsTensor: Type[her_types.ObsTensor]
Tensor: Type[her_types.Tensor]
actor: Any
argparse: module
critic: Any
critic_constructor: Any
datetime: Any
her_sampler: Any
math: Any
normalizer: Any
np: module
os: Any
pdb: module
replay_buffer: Any
sync_grads: Any
sync_networks: Any
torch: module
train_on_target: bool

class ddpg_agent:
    actor_network: Any
    actor_optim: Any
    actor_target_network: Any
    args: argparse.Namespace
    buffer: Any
    critic_network: Any
    critic_optim: Any
    critic_target_network: Any
    env: Any
    env_params: Any
    g_norm: Any
    global_count: int
    her_module: Any
    model_path: Any
    o_norm: Any
    t: int
    def __init__(self, args: argparse.Namespace, env, env_params) -> None: ...
    def _eval_agent(self, final = ...) -> Dict[str, Any]: ...
    def _preproc_inputs(self, obs: her_types.ObsArray, g: her_types.GoalArray, gpi: Optional[her_types.GoalArray] = ...) -> her_types.Tensor: ...
    def _preproc_og(self, o: her_types.ObsArray, g: her_types.GoalArray) -> Tuple[her_types.ObsArray, her_types.GoalArray]: ...
    def _select_actions(self, pi: her_types.ActionTensor) -> her_types.ActionArray: ...
    def _soft_update_target_network(self, target, source) -> None: ...
    def _update_network(self) -> None: ...
    def _update_normalizer(self, episode_batch: list) -> None: ...
    def learn(self) -> None: ...

def reward_offset(t) -> int: ...
