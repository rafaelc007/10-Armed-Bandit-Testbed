import random as rd
import numpy as np


class GreedyBandit:
    """
    Purely greedy implementation. It saves some processing to use this class instead of EGreedyBandit with eps=0.
    """
    __arm_size = 0
    __Qn = []
    __step_n = 0
    __alpha_val = None

    @staticmethod
    def randargmax(b, **kw):
        """ a random tie-breaking argmax"""
        b = np.array(b)
        return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)

    def alpha(self):
        if self.__alpha_val is None:
            return 1/self.__step_n
        else:
            return self.__alpha_val

    def get_armsize(self):
        return self.__arm_size

    def get_alphaval(self):
        return self.__alpha_val

    def __init__(self, arm_size, alpha=None):
        if alpha:
            self.__alpha_val = alpha
        self.__arm_size = arm_size
        self.__Qn = [0] * self.__arm_size

    def train(self, action, reward):
        self.__step_n += 1
        self.__Qn[action] += self.alpha() * (reward - self.__Qn[action])

    def take_action(self):
        return self.randargmax(self.__Qn)

    def reset(self):
        self.__init__(self.__arm_size)


class EGreedyBandit(GreedyBandit):
    """
    Epsilon greedy bandit, ruled by the factor eps which decides when to take actions and when to explore.
    """
    __eps = 0

    def __init__(self, arm_size, alpha=None, eps_val=0.5):
        super().__init__(arm_size, alpha=alpha)
        self.__eps = eps_val

    def set_eps(self, value):
        self.__eps = value

    def take_action(self):
        if rd.random() > self.__eps:
            return self.randargmax(self.__Qn)
        else:
            return rd.sample(range(self.__arm_size), 1)[0]

    def reset(self):
        self.__init__(self.get_armsize(), self.get_alphaval(), self.__eps)
