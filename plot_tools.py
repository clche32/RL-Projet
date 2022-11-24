import matplotlib.pyplot as plt
import numpy as np
from bandit import NonStationaryBandit

def plot_cumul_regret(instances, runs):
    for strategy, params in runs.items():
        cumul_regret = []
        for n in range(len(instances)):
            means = instances[n]
            bandit = NonStationaryBandit(means, seed=n)
            strategy(bandit, **params)
            cumul_regret.append(bandit.get_cumulative_regret())

        avg_cumul_regret = np.mean(cumul_regret, axis=0)
        std_cumul_regret = np.std(cumul_regret, axis=0)
        plt.plot(avg_cumul_regret, label="%s.%s"%(strategy.__module__, strategy.__name__))
        plt.fill_between(range(len(avg_cumul_regret)), avg_cumul_regret, avg_cumul_regret+std_cumul_regret, alpha=0.4)
    plt.legend()
    plt.show()

def plot_regret(instances, runs):
    for strategy, params in runs.items():
        regret = []
        for n in range(len(instances)):
            means = instances[n]
            bandit = NonStationaryBandit(means, seed=n)
            strategy(bandit, **params)
            regret.append(bandit.regret)

        avg_regret = np.mean(regret, axis=0)
        std_regret = np.std(regret, axis=0)
        plt.plot(avg_regret, label="%s.%s"%(strategy.__module__, strategy.__name__))
        plt.fill_between(range(len(avg_regret)), avg_regret, avg_regret+std_regret, alpha=0.4)
    plt.legend()
    plt.show()