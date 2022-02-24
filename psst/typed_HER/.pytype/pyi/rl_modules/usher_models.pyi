# (generated with --quick)

from typing import Any, Tuple, TypeVar

F: module
LOG_STD_MAX: int
LOG_STD_MIN: int
Normal: Any
clip_max: int
nn: module
np: module
pdb: module
scipy: module
torch: module

_T0 = TypeVar('_T0')
_T1 = TypeVar('_T1')

class StateValueEstimator(Any):
    gamma: Any
    def __init__(self, actor, critic, gamma) -> None: ...
    def actor(self, _1) -> Any: ...
    def critic(self, _1, _2) -> Any: ...
    def forward(self, o, g, norm = ...) -> Any: ...
    def q2time(self, q) -> Any: ...

class T_conditioned_ratio_critic(Any):
    env_params: Any
    fc1: Any
    fc2: Any
    fc3: Any
    her_goal_range: Tuple[Any, Any]
    max_action: Any
    norm1: Any
    norm2: Any
    norm3: Any
    p_out: Any
    q_out: Any
    def __init__(self, env_params) -> None: ...
    def forward(self, x, T, actions, return_p = ...) -> Any: ...

class actor(Any):
    fc1: Any
    fc2: Any
    fc3: Any
    g_norm: Any
    log_std_layer: Any
    max_action: Any
    mu_layer: Any
    norm1: Any
    norm2: Any
    norm3: Any
    o_norm: Any
    def __init__(self, env_params) -> None: ...
    def _get_denorms(self, obs, g) -> Tuple[Any, Any]: ...
    def _get_norms(self, obs, g) -> Tuple[Any, Any]: ...
    def forward(self, x, with_logprob = ..., deterministic = ..., forced_exploration = ...) -> Any: ...
    def normed_forward(self, obs, gr, deterministic = ...) -> Any: ...
    def set_normalizers(self, o, g) -> None: ...

class critic(Any):
    fc1: Any
    fc2: Any
    fc3: Any
    max_action: Any
    norm1: Any
    norm2: Any
    norm3: Any
    q_out: Any
    def __init__(self, env_params) -> None: ...
    def forward(self, x, actions) -> Any: ...

class dual_critic(Any):
    q1: critic
    q2: critic
    def __init__(self, env_params) -> None: ...
    def dual(self, x, actions) -> Tuple[Any, Any]: ...
    def forward(self, x, actions) -> Any: ...

class dual_value_prior_critic(Any):
    q1: value_prior_critic
    q2: value_prior_critic
    def __init__(self, env_params) -> None: ...
    def dual(self, x, actions) -> Tuple[Any, Any]: ...
    def forward(self, x, actions) -> Any: ...

class tdm_critic(Any):
    fc1: Any
    fc2: Any
    fc3: Any
    max_action: Any
    norm1: Any
    norm2: Any
    norm3: Any
    q_out: Any
    def __init__(self, env_params) -> None: ...
    def forward(self, x, actions, vec = ...) -> Any: ...

class test_T_conditioned_ratio_critic(Any):
    fc1: Any
    fc2: Any
    fc3: Any
    max_action: Any
    norm: bool
    p_out: Any
    q_out: Any
    def __init__(self, env_params) -> None: ...
    def forward(self, x, T, actions, return_p = ...) -> Any: ...

class value_prior_actor(Any):
    fc1: Any
    fc2: Any
    fc3: Any
    g_norm: Any
    goal_dim: Any
    log_std_layer: Any
    max_action: Any
    mu_layer: Any
    norm1: Any
    norm2: Any
    norm3: Any
    o_norm: Any
    def __init__(self, env_params) -> None: ...
    def _get_denorms(self, obs, g) -> Tuple[Any, Any]: ...
    def _get_norms(self, obs, g) -> Tuple[Any, Any]: ...
    def forward(self, x, with_logprob = ..., deterministic = ..., forced_exploration = ...) -> Any: ...
    def normed_forward(self, obs, g, ag, deterministic = ...) -> Any: ...
    def set_normalizers(self, o, g) -> None: ...

class value_prior_critic(Any):
    add_out: Any
    fc1: Any
    fc2: Any
    fc3: Any
    goal_dim: Any
    max_action: Any
    mult_out: Any
    norm1: Any
    norm2: Any
    norm3: Any
    def __init__(self, env_params) -> None: ...
    def forward(self, x, actions) -> Any: ...

def combined_shape(length: _T0, shape: _T1 = ...) -> tuple: ...
def count_vars(module) -> Any: ...
def mlp(sizes, activation, output_activation = ...) -> Any: ...
