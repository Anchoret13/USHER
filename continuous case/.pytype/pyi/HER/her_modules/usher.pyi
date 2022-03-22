# (generated with --quick)

from typing import Any

copy: module
np: module

class her_sampler:
    future_p: Any
    gamma: Any
    geometric: Any
    replay_k: Any
    replay_strategy: Any
    two_goal: Any
    def __init__(self, replay_strategy, replay_k, reward_func, gamma, two_goal, geometric) -> None: ...
    def reward_func(self, _1, _2, _3: None) -> Any: ...
    def sample_her_transitions(self, episode_batch, batch_size_in_transitions) -> Any: ...
