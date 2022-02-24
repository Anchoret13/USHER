# (generated with --quick)

import HER.rl_modules.typed_usher_with_ratio
import her_types
from typing import Any, Dict, Type

ActionArray: Type[her_types.ActionArray]
ActionTensor: Type[her_types.ActionTensor]
Array: Type[her_types.Array]
GoalArray: Type[her_types.GoalArray]
GoalTensor: Type[her_types.GoalTensor]
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
agent: HER.rl_modules.typed_usher_with_ratio.ddpg_agent
args: Any
ddpg_agent: Type[HER.rl_modules.typed_usher_with_ratio.ddpg_agent]
get_args: Any
gym: Any
np: module
torch: module

def get_env_params(env) -> Dict[str, Any]: ...
def launch(args) -> HER.rl_modules.typed_usher_with_ratio.ddpg_agent: ...
