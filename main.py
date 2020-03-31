import numpy as np
import random as rd
import matplotlib.pyplot as plt


class Arm:
    """
    Implements a bandit arm with gaussian distribution for the reward.
    """
    action_val = 0

    def __init__(self, seed=100):
        rd.seed(seed)
        self.action_val = rd.gauss(0, 1)

    def get_reward(self):
        return rd.gauss(self.action_val, 1)


class DeviantArm(Arm):
    """
    The Deviant Arm implements an arm driven by a random walk on its action value. The random walk presents mean zero
    and std = 0.01.
    """
    def get_reward(self):
        self.action_val += rd.gauss(0, 0.01)
        return rd.gauss(self.action_val, 1)


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


def plot_data(best: int, *args, names=[]):
    """
    Plot
    :param data:
    :return:
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(args))))

    if not names:
        names = ["data_{}".format(k) for k in range(len(args))]

    for n, data in enumerate(args):
        col = next(color)
        data = np.array(data)
        act_data = data[:, :, 0].astype(int)
        rwd_data = data[:, :, 1]

        ax1.plot(np.sum(rwd_data, axis=0)/rwd_data.shape[0], 'r-', label=names[n], c=col)
        ax1.grid(1)
        ax1.set_xlabel("trial")
        ax1.set_ylabel("reward")
        ax1.legend()

        b_data = act_data == best[0]
        ax2.plot(100*np.sum(b_data, axis=0)/b_data.shape[0], 'r-', label=names[n], c=col)
        ax2.grid(1)
        ax2.set_xlabel("trial")
        ax2.set_ylabel("% of best")
        ax2.legend()


    plt.tight_layout()
    plt.show()


def plot_testbed(test_bed: TestBed):
    """
    Plot a graph showing the distribution of a given testbed
    :param test_bed: The testbed to extract the data to plot
    :return:
    """
    arm_values = []
    for arm in test_bed.arms:
        arm_values.append(arm.action_val)

    reward_vals = [[arm.get_reward() for k in range(200)] for arm in test_bed.arms]

    plt.figure()
    plt.boxplot(reward_vals)
    plt.scatter(range(1, test_bed.n_arms+1), arm_values, c="blue")
    plt.grid(1)
    plt.show()


if __name__ == "__main__":
    test_bed = TestBed(10)
    best = test_bed.get_best()
    # plot_testbed(test_bed)

    n_steps = 1000
    n_mean = 2000
    log = [[] for i in range(3)]    #TODO: remove this hardcode here
    for k in range(n_mean):
        print("Training iteration: ", k)
        bandits = [GreedyBandit(10), EGreedyBandit(10, 0.1), EGreedyBandit(10, 0.01)]
        for b_bum, bandit in enumerate(bandits):
            trial = []
            for n in range(n_steps):
                act = bandit.take_action()
                reward = test_bed.trial(act)
                bandit.train(act, reward)
                trial.append([act, reward])
            log[b_bum].append(trial)

    print("Best choice {}, value: {}".format(best[0], best[1]))
    plot_data(best, log[0], log[1], log[2], names=["greedy", "e=0.1", "e=0.01"])
