import random as rd
import numpy as np


class GreedyBandit:
    """
    Purely greedy implementation. It saves some processing to use this class instead of EGreedyBandit with eps=0.
    """
    _arm_size = 0
    _Qn = []
    _step_n = 0
    _alpha_val = None

    @staticmethod
    def randargmax(b, **kw):
        """ a random tie-breaking argmax"""
        b = np.array(b)
        return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)

    def alpha(self):
        if self._alpha_val is None:
            return 1/self._step_n
        else:
            return self._alpha_val

    def get_armsize(self):
        return self._arm_size

    def get_alphaval(self):
        return self._alpha_val

    def __init__(self, arm_size, alpha=None):
        if alpha:
            self._alpha_val = alpha
        self._arm_size = arm_size
        self._Qn = [0] * self._arm_size

    def train(self, action, reward):
        self._step_n += 1
        self._Qn[action] += self.alpha() * (reward - self._Qn[action])

    def take_action(self):
        return self.randargmax(self._Qn)

    def reset(self):
        self.__init__(self._arm_size)


class EGreedyBandit(GreedyBandit):
    """
    Epsilon greedy bandit, ruled by the factor eps which decides when to take actions and when to explore.
    """
    _eps = 0

    def __init__(self, arm_size, alpha=None, eps_val=0.5):
        super().__init__(arm_size, alpha=alpha)
        self._eps = eps_val

    def set_eps(self, value):
        self._eps = value

    def take_action(self):
        if rd.random() > self._eps:
            return self.randargmax(self._Qn)
        else:
            return rd.sample(range(self._arm_size), 1)[0]

    def reset(self):
        self.__init__(self._arm_size, self._alpha_val, self._eps)
