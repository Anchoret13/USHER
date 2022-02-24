# (generated with --quick)

from typing import Any, Dict

np: module
threading: module

class replay_buffer:
    T: Any
    buffers: Dict[str, Any]
    current_size: Any
    env_params: Any
    lock: threading.Lock
    n_transitions_stored: Any
    size: Any
    def __init__(self, env_params, buffer_size, sample_func) -> None: ...
    def _get_storage_idx(self, inc = ...) -> Any: ...
    def sample(self, batch_size) -> Any: ...
    def sample_func(self, _1: Dict[str, Any], _2) -> Any: ...
    def store_episode(self, episode_batch) -> None: ...
