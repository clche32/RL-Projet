import matplotlib.pyplot as plt

from bandit import NonStationaryBandit
from policy.TS import discounted_ts
from policy.TS import sliding_window_ts

def mu_0(t) :
    return 0.5

def mu_1(t) :
    return 0.3

def mu_2(t) :
    if t < 300 or t >= 1000 :
        return 0.1
    else :
        return 0.9

means = [mu_0, mu_1, mu_2]
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