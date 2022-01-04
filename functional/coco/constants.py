#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x24be5ff6

# Compiled with Coconut version 1.6.0 [Vocational Guidance Counsellor]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os as _coconut_os
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os.path.dirname(_coconut_cached_module.__file__) != _coconut_file_dir:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_dir)
_coconut_module_name = _coconut_os.path.splitext(_coconut_os.path.basename(_coconut_file_dir))[0]
if _coconut_module_name and _coconut_module_name[0].isalpha() and all(c.isalpha() or c.isdigit() for c in _coconut_module_name) and "__init__.py" in _coconut_os.listdir(_coconut_file_dir):
    _coconut_full_module_name = str(_coconut_module_name + ".__coconut__")
    import __coconut__ as _coconut__coconut__
    _coconut__coconut__.__name__ = _coconut_full_module_name
    for _coconut_v in vars(_coconut__coconut__).values():
        if getattr(_coconut_v, "__module__", None) == str("__coconut__"):
            try:
                _coconut_v.__module__ = _coconut_full_module_name
            except AttributeError:
                _coconut_v_type = type(_coconut_v)
                if getattr(_coconut_v_type, "__module__", None) == str("__coconut__"):
                    _coconut_v_type.__module__ = _coconut_full_module_name
    _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_tail_call, _coconut_tco, _coconut_call_set_names, _coconut_handle_cls_kwargs, _coconut_handle_cls_stargs, _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------


from typing import Tuple  #2 (line num in coconut source)
from typing import Dict  #2 (line num in coconut source)
from typing import Optional  #2 (line num in coconut source)
from typing import TypeVar  #2 (line num in coconut source)

State = Tuple[int, int]  #4 (line num in coconut source)
Goal = Tuple[int, int]  #5 (line num in coconut source)
Action = Tuple[int, int]  #6 (line num in coconut source)
ActionIndex = int  #7 (line num in coconut source)
Observation = Tuple[int, int]  #8 (line num in coconut source)
Reward = float  #9 (line num in coconut source)
# ObservationDict = dict[State, Observation, Goal, Goal]
ObservationDict = Dict[str, Tuple[int, int]]  #11 (line num in coconut source)
Transition = Tuple[State, Goal, Goal, ActionIndex, State, float]  #12 (line num in coconut source)

T = TypeVar('T')  #14 (line num in coconut source)


@_coconut_tco  #17 (line num in coconut source)
def add(s,  # type: tuple
     a  # type: tuple
    ):  #17 (line num in coconut source)
# type: (...) -> tuple
    return _coconut_tail_call(tuple, (a + b for (a, b) in zip(s, a)))  #18 (line num in coconut source)


@_coconut_tco  #21 (line num in coconut source)
def sub(s,  # type: tuple
     a  # type: tuple
    ):  #21 (line num in coconut source)
# type: (...) -> tuple
    return _coconut_tail_call(tuple, (a - b for (a, b) in zip(s, a)))  #22 (line num in coconut source)

def default(x,  # type: Optional[T]
     default_val  # type: T
    ):  #24 (line num in coconut source)
# type: (...) -> T
    if x is not None:  #25 (line num in coconut source)
        outval = x  #26 (line num in coconut source)
    else:  #27 (line num in coconut source)
        outval = default_val  #28 (line num in coconut source)
# assert outval is not None
    return (outval)  #30 (line num in coconut source)


EMPTY = 0  #33 (line num in coconut source)
BLOCK = 1  #34 (line num in coconut source)
WIND = 2  #35 (line num in coconut source)
RANDOM_DOOR = 3  #36 (line num in coconut source)
