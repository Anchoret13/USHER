# (generated with --quick)

import numpy
from typing import Any, Dict, List, Optional, Tuple

MPI: Any
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
torch: Any
train_on_target: bool

class ActionArray(numpy.ndarray):
    def __init__(self, val: numpy.ndarray) -> None: ...

class ActionTensor(Tensor):
    def __init__(self, val: Tensor) -> None: ...

class Array(numpy.ndarray):
    def __init__(self, val: numpy.ndarray) -> None: ...

class GoalArray(numpy.ndarray):
    def __init__(self, val: numpy.ndarray) -> None: ...

class GoalTensor(Tensor):
    def __init__(self, val: Tensor) -> None: ...

class NormalActionArray(numpy.ndarray):
    def __init__(self, val: numpy.ndarray) -> None: ...

class NormalActionTensor(NormedTensor):
    def __init__(self, val: NormedTensor) -> None: ...

class NormalGoalArray(numpy.ndarray):
    def __init__(self, val: numpy.ndarray) -> None: ...

class NormalGoalTensor(NormedTensor):
    def __init__(self, val: NormedTensor) -> None: ...

class NormalObsArray(numpy.ndarray):
    def __init__(self, val: numpy.ndarray) -> None: ...

class NormalObsTensor(NormedTensor):
    def __init__(self, val: NormedTensor) -> None: ...

class NormedArray(numpy.ndarray):
    def __init__(self, val: numpy.ndarray) -> None: ...

class NormedTensor(Any):
    def __new__(cls, x) -> Any: ...

class ObsArray(numpy.ndarray):
    def __init__(self, val: numpy.ndarray) -> None: ...

class ObsTensor(Tensor):
    def __init__(self, val: Tensor) -> None: ...

class Tensor(Any):
    def __new__(cls, x) -> Any: ...

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
    def _preproc_inputs(self, obs: ObsArray, g: GoalArray, gpi: Optional[GoalArray] = ...) -> NormedTensor: ...
    def _preproc_og(self, o: ObsArray, g: GoalArray) -> Tuple[ObsArray, GoalArray]: ...
    def _select_actions(self, pi: ActionTensor) -> ActionArray: ...
    def _soft_update_target_network(self, target, source) -> None: ...
    def _update_network(self) -> None: ...
    def _update_normalizer(self, episode_batch: List[numpy.ndarray]) -> None: ...
    def learn(self) -> None: ...

def reward_offset(t) -> int: ...
