import matplotlib.pyplot as plt

from bandit import NonStationaryBandit
from policy.TS import discounted_ts
from policy.TS import sliding_window_ts

def mu_0(t) :
    return 0.5

def mu_1(t) :
    return 0.3

def mu_2(t) :
    if t < 300 or t >= 500 :
        return 0.4
    else :
        return 0.9

means = [mu_0, mu_1, mu_2]
T = 2000
alpha, beta = 1, 1 # Uniform prior
gamma = 0.9 # Discount factor
tau = 30 # Sliding window size

bandit = NonStationaryBandit(means, seed=42)
discounted_ts(bandit, T, alpha, beta, gamma, seed=42)
cumul_regret = bandit.get_cumulative_regret()
plt.plot(cumul_regret, label="Discounted TS")

bandit = NonStationaryBandit(means, seed=42)
sliding_window_ts(bandit, T, alpha, beta, tau, seed=42)
cumul_regret = bandit.get_cumulative_regret()
plt.plot(cumul_regret, label="Sliding window TS")

plt.xlabel("Pas de temps")
plt.ylabel("Pseudo-regret cumulatif")
plt.legend()
plt.show()