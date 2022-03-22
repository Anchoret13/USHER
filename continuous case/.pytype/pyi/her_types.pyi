# (generated with --quick)

import numpy
from typing import Any

np: module
torch: module

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
