# (generated with --quick)

from typing import Any, Tuple

F: module
LOG_STD_MAX: int
LOG_STD_MIN: int
Normal: Any
clip_max: int
nn: module
np: module
scipy: module
torch: module

class actor(Any):
    fc1: Any
    fc2: Any
    fc3: Any
    g_norm: Any
    log_std_layer: Any
    max_action: Any
    mu_layer: Any
    o_norm: Any
    def __init__(self, env_params) -> None: ...
    def _get_denorms(self, obs, g) -> Tuple[Any, Any]: ...
    def _get_norms(self, obs, g) -> Tuple[Any, Any]: ...
    def forward(self, x, with_logprob = ..., deterministic = ..., forced_exploration = ...) -> Any: ...
    def normed_forward(self, obs, g, deterministic = ...) -> Any: ...
    def set_normalizers(self, o, g) -> None: ...
