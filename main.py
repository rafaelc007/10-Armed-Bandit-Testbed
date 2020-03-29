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
    eps = 0.1

    def set_eps(self, value):
        self.eps = value

    def take_action(self):
        if rd.random() > self.eps:
            return np.argmax(self.Qn)
        else:
            return rd.sample(range(self.arm_size), 1)[0]


def plot_data(data):
    act_data, rw_data = zip(*data)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(rw_data, 'r-', label="reward")
    plt.grid(1)
    plt.xlabel("trial")
    plt.ylabel("reward")
    plt.subplot(1, 2, 2)
    plt.plot(act_data, 'o-', label="action")
    plt.grid(1)
    plt.xlabel("action")
    plt.ylabel("reward")
    plt.ylim((0, 10))
    plt.show()


def plot_best(best_data, n_tot):
    b_data = 100* np.array(best_data) / n_tot
    plt.figure()
    plt.plot(b_data, 'r-', label="reward")
    plt.grid(1)
    plt.xlabel("trial")
    plt.ylabel("best")
    plt.show()


def plot_testbed(test_bed: TestBed):
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
    #rd.seed(100)
    test_bed = TestBed(10)
    best = test_bed.get_best()
    # plot_testbed(test_bed)

    n_steps = 100
    n_mean = 200
    n_best1 = [0]*n_steps
    n_best2 = [0]*n_steps
    log1 = np.array([[0.0, 0.0]] * n_steps)
    log2 = np.array([[0.0, 0.0]] * n_steps)
    for k in range(n_mean):
        print("Training iteration: ", k)
        bandit1 = GreedyBandit(10)
        bandit2 = EGreedyBandit(10)
        for n in range(n_steps):
            act1 = bandit1.take_action()
            act2 = bandit2.take_action()
            if act1 == best[0]: n_best1[n] += 1
            if act2 == best[0]: n_best2[n] += 1

            reward1 = test_bed.trial(act1)
            reward2 = test_bed.trial(act2)

            bandit1.train(act1, reward1)
            bandit2.train(act2, reward2)
            log1[n] = np.add(log1[n], 1 / n_mean * np.array([act1, reward1]))
            log2[n] = np.add(log2[n], 1 / n_mean * np.array([act2, reward2]))

    print("Best choice {}, value: {}".format(best[0], best[1]))
    plot_data(log1)
    # plot_best(n_best2, n_mean)
    plot_data(log2)