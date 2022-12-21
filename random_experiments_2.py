import numpy as np
import matplotlib.pyplot as plt

from bandit import NonStationaryBandit
from policy import UCB
from policy import TS
from means import RandomAbruptFunction
from plot_tools import *

K = 2 # number of arms
N = 50 # number of instances
instances = []
for n in range(N):
    means = []
    for k in range(K):
        mu_random_abrupt = RandomAbruptFunction(d=0.001, seed=[n,k])
        means.append(mu_random_abrupt)
    instances.append(means)

sw_ucb = [[UCB.sliding_window, {'T':1000, 'tau':200, 'xi':0.5}],
          [UCB.sliding_window, {'T':1000, 'tau':100, 'xi':0.5}],
          [UCB.sliding_window, {'T':1000, 'tau':50, 'xi':0.5}]]

plt.figure(figsize=(30,10))

plt.subplot(121)
plt.title('d = 0.1%', fontweight='bold', fontsize='x-large', y=0.92)
plot_cumul_regret(instances, runs=sw_ucb, N=1, label_mode="params")

instances = []
for n in range(N):
    means = []
    for k in range(K):
        mu_random_abrupt = RandomAbruptFunction(d=0.01, seed=[n,k])
        means.append(mu_random_abrupt)
    instances.append(means)

sw_ucb = [[UCB.sliding_window, {'T':1000, 'tau':200, 'xi':0.5}],
          [UCB.sliding_window, {'T':1000, 'tau':100, 'xi':0.5}],
          [UCB.sliding_window, {'T':1000, 'tau':50, 'xi':0.5}]]

plt.subplot(122)
plt.title('d = 1%', fontweight='bold', fontsize='x-large', y=0.92)
plot_cumul_regret(instances, runs=sw_ucb, N=1, label_mode="params")

plt.show()
