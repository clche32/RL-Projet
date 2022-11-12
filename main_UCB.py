import numpy as np
import matplotlib.pyplot as plt

from bandit import NonStationaryBandit
from policy.UCB import discounted_ucb
from policy.UCB import sliding_window
from policy.UCB import f_dsw

def mu_0(t) :
    return 0.5

def mu_1(t) :
    return 0.3

def mu_2(t) :
    if t < 3000 or t >= 5000 :
        return 0.4
    else :
        return 0.9

means = [mu_0, mu_1, mu_2]
bandit = NonStationaryBandit(means, seed=42)

# Avec des plus grandes valeurs de T (et donc par exemple t<3000 et t>5000), on semble avoir des meilleurs résultats, comme si l'algo
# avait le temps d'apprendre contrairement à T = 2000.

T = 10000

N=10

functions = [sliding_window, discounted_ucb, f_dsw]
params = [ [T, int(4*np.sqrt(T*np.log(T))), 0.5, 0.6], [T, 0.9, 0.5, 0.6], [T, int(4*np.sqrt(T*np.log(T))), 0.9, 0.5, 0.6, np.min] ]
for func, params in zip(functions, params):
    cumul_regret = []
    print(func.__name__)
    for i in range(N) :
        bandit.regret = []
        func(bandit, *params)
        cumul_regret.append(bandit.get_cumulative_regret())

    avg_cumul_regret = np.mean(cumul_regret, axis=0)
    std_cumul_regret = np.std(cumul_regret, axis=0)

    plt.plot(avg_cumul_regret, label="func = " + str(func.__name__))
    plt.fill_between(np.arange(T), avg_cumul_regret, avg_cumul_regret+std_cumul_regret, alpha=0.4)

    plt.xlabel("Time steps")
    plt.ylabel("Cumulative pseudo-regret")

    plt.legend()
plt.show()