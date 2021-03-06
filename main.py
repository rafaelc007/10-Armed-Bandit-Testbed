import numpy as np
import matplotlib.pyplot as plt
from TestbedClass import TestBed, DeviantTestBed
from BanditClass import GreedyBandit, EGreedyBandit


def plot_data(*args, names=[]):
    """
    Plot the data obtained in a run in the way shown in the RL book.
    :param best: tuple (is best action, reward) containing the best choice found for the given run.
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
        bact_data = data[:, :, 0].astype(int)   # is best action? data
        rwd_data = data[:, :, 1]        # reward data

        ax1.plot(np.average(rwd_data, axis=0), label=names[n], c=col, alpha=0.8)
        ax1.grid(1)
        ax1.set_xlabel("trial")
        ax1.set_ylabel("reward")
        ax1.legend()

        ax2.plot(100*np.average(bact_data, axis=0), label=names[n], c=col, alpha=0.8)
        ax2.grid(1)
        ax2.set_xlabel("trial")
        ax2.set_ylabel("% of choosing best")
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
    for arm in test_bed.__arms:
        arm_values.append(arm.action_val)

    reward_vals = [[arm.get_reward() for k in range(200)] for arm in test_bed.__arms]

    plt.figure()
    plt.boxplot(reward_vals)
    plt.scatter(range(1, test_bed.__n_arms + 1), arm_values, c="blue")
    plt.grid(1)
    plt.show()

def run_testbed(n_steps, n_mean, n_arms, bed_params, verbose=0):
    """
    Runs a testbed simulation
    :param bed_params: (TestBed, [Bandits])  instances
    :param n_teps: number of steps to train the testbed
    :param n_mean: number of trials to use when computing the mean
    :param n_arms: number of arms to use
    :param verbose: 0 -> show nothing, 1 -> show steps, 2 or more -> show steps and testbed graph
    :return: log data collected from the trials
    """
    test_bed = bed_params[0]
    bandits = bed_params[1]

    if verbose > 1:
        plot_testbed(test_bed)

    log = [[] for i in range(len(bandits))]
    for k in range(n_mean):
        if verbose > 0:
            print("Training iteration: ", k)
        [bandit.reset() for bandit in bandits]  # restart all bandits
        for b_bum, bandit in enumerate(bandits):
            trial = []
            for n in range(n_steps):
                act = bandit.take_action()
                reward = test_bed.trial(act)
                if reward:
                    bandit.train(act, reward)
                else:
                    raise Exception("reward returned None")
                trial.append([act == test_bed.get_best()[0], reward])
            log[b_bum].append(trial)

    return log

def run_10_armed(n_steps, n_mean, run_type="classic"):
    """
    :param n_teps: number of steps to train the testbed
    :param n_mean: number of trials to use when computing the mean
    :param run_type:
        run_type="classic" -> Run the 10-armed testbed example that is depicted in the RL book.
        run_type="deviant" -> Run the deviant 10-armed testbed example requested in exercise 2.5 of the RL book.
    :return: plot graph
    """
    n_arms = 10
    if run_type == "classic":
        test_bed = TestBed(n_arms)
        bandits = [GreedyBandit(n_arms), EGreedyBandit(n_arms, eps_val=0.1), EGreedyBandit(n_arms, eps_val=0.01)]
    elif run_type == "deviant":
        test_bed = DeviantTestBed(n_arms)
        bandits = [EGreedyBandit(n_arms, eps_val=0.1), EGreedyBandit(n_arms, alpha=0.1, eps_val=0.1)]
    else:
        raise Exception("Testbed type not supported")

    log = run_testbed(n_steps, n_mean, n_arms, [test_bed, bandits], verbose=1)

    if run_type == "classic":
        plot_data(log[0], log[1], log[2], names=["greedy", "e=0.1", "e=0.01"])
    else:
        plot_data(log[0], log[1], names=["average", "const alpha"])


if __name__ == "__main__":
    run_10_armed(1000, 2000, "classic")