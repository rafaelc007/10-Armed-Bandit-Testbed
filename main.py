import numpy as np
import random as rd
import matplotlib.pyplot as plt


class Arm:
    action_val = 0

    def __init__(self):
        self.action_val = rd.gauss(0, 1)

    def get_reward(self):
        return rd.gauss(self.action_val, 1)


class TestBed:
    arms = []
    n_arms = 0

    def __init__(self, n_size=10):
        self.n_arms = n_size
        self.arms = [Arm() for i in range(self.n_arms)]

    def trial(self, action):
        if action > self.n_arms or action < 0:
            print("action a out of bounds")
            return None
        return self.arms[action].get_reward()


    def get_best(self):
        vals = [arm.action_val for arm in self.arms]
        return [np.argmax(vals), np.max(vals)]


class GreedyBandit:
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
    eps = 0

    def set_eps(self, value, epsilon=0.5):
        self.eps = epsilon
        self.eps = value

    def take_action(self):
        if rd.random() > self.eps:
            return np.argmax(self.Qn)
        else:
            return rd.sample(range(self.arm_size), 1)[0]


def plot_data(best: int, *args):
    """
    Plot
    :param data:
    :return:
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(args))))

    for n, data in enumerate(args):
        col = next(color)
        data = np.array(data)
        act_data = data[:, :, 0].astype(int)
        rwd_data = data[:, :, 1]

        ax1.plot(np.sum(rwd_data, axis=0)/rwd_data.shape[0], 'r-', label="rwd{}".format(n), c=col)
        ax1.grid(1)
        ax1.set_xlabel("trial")
        ax1.set_ylabel("reward")
        ax1.legend()

        b_data = act_data == best[0]
        ax2.plot(100*np.sum(b_data, axis=0)/b_data.shape[0], 'r-', label="best{}".format(n), c=col)
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
    # rd.seed(100)
    test_bed = TestBed(10)
    best = test_bed.get_best()
    # plot_testbed(test_bed)

    n_steps = 100
    n_mean = 100
    log1 = []
    log2 = []
    for k in range(n_mean):
        print("Training iteration: ", k)
        bandit1 = GreedyBandit(10)
        bandit2 = EGreedyBandit(10)
        trial1 = []
        trial2 = []
        for n in range(n_steps):
            act1 = bandit1.take_action()
            act2 = bandit2.take_action()

            reward1 = test_bed.trial(act1)
            reward2 = test_bed.trial(act2)

            bandit1.train(act1, reward1)
            bandit2.train(act2, reward2)

            trial1.append([act1, reward1])
            trial2.append([act2, reward2])

        log1.append(trial1)
        log2.append(trial2)

    print("Best choice {}, value: {}".format(best[0], best[1]))
    plot_data(best, log1, log2)