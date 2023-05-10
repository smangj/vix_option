#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/6 13:23
# @Author   : wsy
# @email    : 631535207@qq.com
from collections import namedtuple
from typing import Any
from qlib.rl.simulator import Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from typing import Tuple
from gym import spaces
import numpy as np

State = namedtuple("State", ["value", "last_action"])


class SimpleSimulator(Simulator[float, State, float]):
    def __init__(self, initial: float, nsteps: int, **kwargs: Any) -> None:
        super().__init__(initial)

        self.value = initial
        self.last_action = 0.0
        self.remain_steps = nsteps

    def step(self, action: float) -> None:
        assert 0.0 <= action <= self.value
        self.last_action = action
        self.remain_steps -= 1

    def get_state(self) -> State:
        return State(self.value, self.last_action)

    def done(self) -> bool:
        return self.remain_steps == 0


class SimpleStateInterpreter(StateInterpreter[Tuple[float, float], np.ndarray]):
    def interpret(self, state: State) -> np.ndarray:
        return np.array([State.value], dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)


state_interpreter = SimpleStateInterpreter()


class SimpleActionInterpreter(ActionInterpreter[State, int, float]):
    def __init__(self, n_value: int) -> None:
        self.n_value = n_value

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.n_value + 1)

    def interpret(self, simulator_state: State, action: int) -> float:
        assert 0 <= action <= self.n_value

        return simulator_state.value * (action / self.n_value)


action_interpreter = SimpleActionInterpreter(n_value=10)


class SimpleReward(Reward[State]):
    def reward(self, simulator_state: State) -> float:
        rew = simulator_state.last_action / simulator_state.value
        return rew


reward = SimpleReward()
