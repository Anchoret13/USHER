
from typing import Tuple, Dict

State = Tuple[int, int]
Goal = Tuple[int, int]
Action = Tuple[int, int]
ActionIndex = int
Observation = Tuple[int, int]
Reward = float
# ObservationDict = dict[State, Observation, Goal, Goal]
ObservationDict = Dict[str, Tuple[int, int]]
Transition = Tuple[State, Goal, Goal, ActionIndex, State, float]


def add(s: tuple, a: tuple) -> tuple:
	return tuple(a + b for (a,b) in zip(s, a))


def sub(s: tuple, a: tuple) -> tuple:
	return tuple(a - b for (a,b) in zip(s, a))

EMPTY = 0
BLOCK = 1
WIND = 2
RANDOM_DOOR = 3