import torch
from typing import *
import numpy as np

T = TypeVar('T')

class Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x):
        return super().__new__(cls, x)

class NormedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x):
        return super().__new__(cls, x)
# Tensor = NewType('Tensor', torch.tensor)
# NormedTensor = NewType('NormedTensor', torch.tensor)
# ObsTensor = NewType('ObsTensor', Tensor)
# ActionTensor = NewType('ActionTensor', Tensor)
# GoalTensor = NewType('GoalTensor', Tensor)
# NormalObsTensor = NewType('NormalObsTensor', NormedTensor)
# NormalActionTensor = NewType('NormalActionTensor', NormedTensor)
# NormalGoalTensor = NewType('NormalGoalTensor', NormedTensor)


class Array(np.ndarray):
    @staticmethod
    def __new__(cls, x):
        return super().__new__(cls, x)

class NormedArray(np.ndarray):
    @staticmethod
    def __new__(cls, x):
        return super().__new__(cls, x)

# Array = NewType('Array', np.ndarray)
# NormedArray = NewType('NormedArray', np.ndarray)
# ObsArray = NewType('ObsArray', np.ndarray)
# ActionArray = NewType('ActionArray', np.ndarray)
# GoalArray = NewType('GoalArray', np.ndarray)
# NormalObsArray = NewType('NormalObsArray', np.ndarray)
# NormalActionArray = NewType('NormalActionArray', np.ndarray)
# NormalGoalArray = NewType('NormalGoalArray', np.ndarray)

def default(x: Optional[T], default_value: T) -> T: 
    if type(T) == type(None): 
        result = default_value
    else: 
        result = x
    assert type(result) != type(None)
    return result