import numpy as np
import matplotlib.pyplot as plt

from bandit import NonStationaryBandit
from policy import UCB
from policy import TS
from means import *
from plot_tools import *

means = [mu_stable_0, mu_abrupt]
instances = [means]

ucb = [[UCB.vanilla, {'T':1000, 'xi':0.5}],
       [UCB.discounted, {'T':1000, 'gamma':0.995, 'xi':0.5}],
       [UCB.sliding_window, {'T':1000, 'tau':200, 'xi':0.5}],
       [UCB.f_dsw, {'T':1000, 'tau':200, 'gamma':0.995, 'f':np.min, 'xi':0.5}]]

ts = [[TS.vanilla, {'T':1000, 'alpha':1, 'beta':1}],
      [TS.discounted, {'T':1000, 'gamma':0.985, 'alpha':1, 'beta':1}],
      [TS.sliding_window, {'T':1000, 'tau':150, 'alpha':1, 'beta':1}],
      [TS.f_dsw, {'T':1000, 'tau':150, 'gamma':0.985, 'f':np.mean, 'alpha':1, 'beta':1}]]

#plot_means(means, T=1000)
plt.figure(figsize=(4.5, 10))
plot_cumul_regret(instances, runs=ucb, N=100, label_mode="strategy")
plt.figure(figsize=(4.5, 10))
plot_ucbs(means, runs=ucb, N=100, title_mode="strategy")
plt.figure(figsize=(4.5, 10))
plot_cumul_regret(instances, runs=ts, N=100, label_mode="strategy")
plt.figure(figsize=(4.5, 10))
plot_samples(means, runs=ts, N=100, title_mode="strategy")
plt.show()
