import random as rd
import numpy as np

class GreedyBandit:
    """
    Purely greedy implementation. It saves some processing to use this class instead of EGreedyBandit with eps=0.
    """
    arm_size = 0
    Qn = []
    step_n = 0

    @staticmethod
    def alpha(x):
        return 1/x

    def __init__(self, arm_size):
        self.arm_size = arm_size
        self.Qn = [0]*self.arm_size

    def train(self, action, reward):
        self.step_n += 1
        self.Qn[action] += self.alpha(self.step_n) * (reward - self.Qn[action])

    def take_action(self):
        return np.argmax(self.Qn)

    def reset(self):
        self.__init__(self.arm_size)


class EGreedyBandit(GreedyBandit):
    """
    Epsilon greedy bandit, ruled by the factor eps which decides when to take actions and when to explore.
    """
    eps = 0

    def __init__(self, arm_size, eps_val=0.5):
        super().__init__(arm_size)
        self.eps = eps_val

    def set_eps(self, value):
        self.eps = value

    def take_action(self):
        if rd.random() > self.eps:
            return np.argmax(self.Qn)
        else:
            return rd.sample(range(self.arm_size), 1)[0]

    def reset(self):
        self.__init__(self.arm_size, self.eps)