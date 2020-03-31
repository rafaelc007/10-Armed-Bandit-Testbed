import numpy as np
import matplotlib.pyplot as plt
from TestbedClass import TestBed
from BanditClass import GreedyBandit, EGreedyBandit


def plot_data(best: int, *args, names=[]):
    """
    Plot the data obtained in a run in the way shown in the RL book.
    :param best: tuple (action, reward) containing the best choice found for the given run.
    :param args: sequence of data to plot in the format (n_mean, n_steps, 2), each data is treated as (action, reward).
    :param names: labels to put in the data plot.
    :return:
    """
    n_args = len(args)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_args)))

    if not names or len(names) != n_args:
        names = ["data_{}".format(k) for k in range(n_args)]

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


def run_10_armed():
    """
    Run the 10-armed testbed example that is depicted in the RL book.
    :return: plot graph
    """
    test_bed = TestBed(10)
    best = test_bed.get_best()
    # plot_testbed(test_bed)

    n_steps = 100
    n_mean = 200
    bandits = [GreedyBandit(10), EGreedyBandit(10, 0.1), EGreedyBandit(10, 0.01)]
    log = [[] for i in range(len(bandits))]
    for k in range(n_mean):
        print("Training iteration: ", k)
        [bandit.reset() for bandit in bandits]  # restart all bandits
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


if __name__ == "__main__":
    run_10_armed()