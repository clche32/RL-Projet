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
        mu_random_abrupt = RandomAbruptFunction(d=0.01, seed=[n,k])
        means.append(mu_random_abrupt)
    instances.append(means)

sw_ucb = [[UCB.sliding_window, {'T':1000, 'tau':200, 'xi':0.5}],
          [UCB.sliding_window, {'T':1000, 'tau':100, 'xi':0.5}], # the best
          [UCB.sliding_window, {'T':1000, 'tau':50, 'xi':0.5}]]

d_ucb = [[UCB.discounted, {'T':1000, 'gamma':0.999, 'xi':0.5}],
         [UCB.discounted, {'T':1000, 'gamma':0.99, 'xi':0.5}], # the best
         [UCB.discounted, {'T':1000, 'gamma':0.9, 'xi':0.5}]]

sw_ts  = [[TS.sliding_window, {'T':1000, 'tau':200, 'alpha':1, 'beta':1}],
          [TS.sliding_window, {'T':1000, 'tau':100, 'alpha':1, 'beta':1}], # the best
          [TS.sliding_window, {'T':1000, 'tau':50, 'alpha':1, 'beta':1}]]

d_ts  = [[TS.discounted, {'T':1000, 'gamma':0.999, 'alpha':1, 'beta':1}],
         [TS.discounted, {'T':1000, 'gamma':0.99, 'alpha':1, 'beta':1}], # the best (0.98 even better)
         [TS.discounted, {'T':1000, 'gamma':0.9, 'alpha':1, 'beta':1}]]

#plot_means(instances[14], T=1000)
plt.figure(figsize=(12,8))

plt.subplot(221)
plt.title('SW-UCB')
plot_cumul_regret(instances, runs=sw_ucb, N=1, label_mode="params")

plt.subplot(222)
plt.title('D-UCB')
plot_cumul_regret(instances, runs=d_ucb, N=1, label_mode="params")

plt.subplot(223)
plt.title('SW-TS')
plot_cumul_regret(instances, runs=sw_ts, N=1, label_mode="params")

plt.subplot(224)
plt.title('D-TS')
plot_cumul_regret(instances, runs=d_ts, N=1, label_mode="params")

plt.tight_layout()
plt.show()
