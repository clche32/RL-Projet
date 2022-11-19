import matplotlib.pyplot as plt

from bandit import NonStationaryBandit
from means import *
from policy.TS import discounted_ts
from policy.TS import sliding_window_ts

means = [mu_stable_0, mu_stable_0, mu_abrupt_thin]
T = 2000
alpha, beta = 1, 1 # Uniform prior
gamma = 0.999 # Discount factor
tau = 100 # Sliding window size

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

plt.figure()

bandit = NonStationaryBandit(means, seed=42)
sliding_window_ts(bandit, T, alpha, beta, tau, seed=42)
N = bandit.get_N()
for k in range(3):
    plt.plot(N[k,:], label="Arm {}".format(k))

plt.title("Sliding window : number of times each arm is chosen")
plt.xlabel("Pas de temps")
plt.ylabel("$N_k$")
plt.legend()
plt.show()