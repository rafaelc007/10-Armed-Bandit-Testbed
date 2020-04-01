import random as rd

class Arm:
    """
    Implements a bandit arm with gaussian distribution for the reward.
    """
    _action_val = 0

    def __init__(self, seed=100):
        rd.seed(seed)
        self._action_val = rd.gauss(0, 1)

    def get_actionval(self):
        return self._action_val

    def get_reward(self):
        return rd.gauss(self._action_val, 1)


class DeviantArm(Arm):
    """
    The Deviant Arm implements an arm driven by a random walk on its action value. The random walk presents mean zero
    and std = 0.01.
    """
    def get_reward(self):
        self._action_val += rd.gauss(0, 0.01)
        return rd.gauss(self._action_val, 1)
