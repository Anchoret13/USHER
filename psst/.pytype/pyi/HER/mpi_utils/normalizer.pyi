# (generated with --quick)

from typing import Any, Tuple, TypeVar

MPI: Any
np: module
threading: module
torch: module

_T0 = TypeVar('_T0')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')

class normalizer:
    default_clip_range: Any
    eps: Any
    local_count: Any
    local_sum: Any
    local_sumsq: Any
    lock: threading.Lock
    mean: Any
    size: Any
    std: Any
    total_count: Any
    total_sum: Any
    total_sumsq: Any
    def __init__(self, size, eps = ..., default_clip_range = ...) -> None: ...
    def _mpi_average(self, x) -> Any: ...
    def denormalize(self, v) -> Any: ...
    def get_torch_normalizer(self) -> torch_normalizer: ...
    def normalize(self, v, clip_range = ...) -> Any: ...
    def recompute_stats(self) -> None: ...
    def sync(self, local_sum: _T0, local_sumsq: _T1, local_count: _T2) -> Tuple[_T0, _T1, _T2]: ...
    def update(self, v) -> None: ...

class torch_normalizer:
    clip_range: Any
    mean: Any
    std: Any
    def __init__(self, means, stds, clip_range) -> None: ...
    def denormalize(self, v) -> Any: ...
    def normalize(self, v) -> Any: ...
