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
        mu_random_abrupt = RandomAbruptFunction(d=0.005, seed=[n,k])
        means.append(mu_random_abrupt)
    instances.append(means)

sw_ucb = [[UCB.sliding_window, {'T':1000, 'tau':1000, 'xi':2}],
          [UCB.sliding_window, {'T':1000, 'tau':100, 'xi':2}],
          [UCB.sliding_window, {'T':1000, 'tau':10, 'xi':2}]]

d_ucb = [[UCB.discounted, {'T':1000, 'gamma':1, 'xi':2}],
         [UCB.discounted, {'T':1000, 'gamma':0.99, 'xi':2}],
         [UCB.discounted, {'T':1000, 'gamma':0.9, 'xi':2}]]

sw_ts  = [[TS.sliding_window, {'T':1000, 'tau':1000, 'alpha':1, 'beta':1}],
          [TS.sliding_window, {'T':1000, 'tau':100, 'alpha':1, 'beta':1}],
          [TS.sliding_window, {'T':1000, 'tau':10, 'alpha':1, 'beta':1}]]

d_ts  = [[TS.discounted, {'T':1000, 'gamma':1, 'alpha':1, 'beta':1}],
         [TS.discounted, {'T':1000, 'gamma':0.99, 'alpha':1, 'beta':1}],
         [TS.discounted, {'T':1000, 'gamma':0.9, 'alpha':1, 'beta':1}]]

#plot_means(instances[0], T=1000)
plot_estimated_means(instances[0], runs=sw_ucb, N=100, title_mode="params")
plot_cumul_regret(instances, runs=sw_ts, N=1, label_mode="params")
plt.show()
