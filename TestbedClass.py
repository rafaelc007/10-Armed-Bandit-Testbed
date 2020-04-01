from ArmClass import Arm, DeviantArm
import numpy as np


class TestBed:
    """
    The testbed includes a number 'n_arms' of bandits, each ruled by a gaussian reward rule.
    """
    _arms = []
    _n_arms = 0

    def __init__(self, n_size=10):
        self._n_arms = n_size
        self._arms = [Arm() for i in range(self._n_arms)]

    def trial(self, action):
        if action > self._n_arms or action < 0:
            print("action a out of bounds")
            return None
        return self._arms[action].get_reward()

    def get_best(self):
        vals = [arm.get_actionval() for arm in self._arms]
        max_act = np.argmax(vals)
        return [max_act, vals[max_act]]


class DeviantTestBed(TestBed):
    def __init__(self, n_size=10):
        self._n_arms = n_size
        self._arms = [DeviantArm() for i in range(self._n_arms)]

    def trial(self, action):
        [arm.update_actionval() for arm in self._arms]
        return super().trial(action)
