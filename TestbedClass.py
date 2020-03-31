from ArmClass import Arm, DeviantArm
import numpy as np


class TestBed:
    """
    The testbed includes a number 'n_arms' of bandits, each ruled by a gaussian reward rule.
    """
    arms = []
    n_arms = 0

    def __init__(self, n_size=10):
        self.n_arms = n_size
        self.arms = [Arm(100+i) for i in range(self.n_arms)]

    def trial(self, action):
        if action > self.n_arms or action < 0:
            print("action a out of bounds")
            return None
        return self.arms[action].get_reward()


    def get_best(self):
        vals = [arm.action_val for arm in self.arms]
        return [np.argmax(vals), np.max(vals)]


class DeviantTestBed(TestBed):
    def __init__(self, n_size=10):
        self.n_arms = n_size
        self.arms = [DeviantArm(100 + i) for i in range(self.n_arms)]