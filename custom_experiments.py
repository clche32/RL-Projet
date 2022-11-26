import numpy as np
import matplotlib.pyplot as plt

from bandit import NonStationaryBandit
from policy import UCB
from policy import TS
from means import *
from plot_tools import *

means = [mu_stable_0, mu_abrupt]
instances = [means]

sw_ucb = [[UCB.sliding_window, {'T':1000, 'tau':1000, 'xi':0.5}],
          [UCB.sliding_window, {'T':1000, 'tau':100, 'xi':0.5}],
          [UCB.sliding_window, {'T':1000, 'tau':10, 'xi':0.5}]]

d_ucb = [[UCB.discounted, {'T':1000, 'gamma':1, 'xi':0.5}],
         [UCB.discounted, {'T':1000, 'gamma':0.99, 'xi':0.5}],
         [UCB.discounted, {'T':1000, 'gamma':0.9, 'xi':0.5}]]

sw_ts  = [[TS.sliding_window, {'T':1000, 'tau':1000, 'alpha':1, 'beta':1}],
          [TS.sliding_window, {'T':1000, 'tau':100, 'alpha':1, 'beta':1}],
          [TS.sliding_window, {'T':1000, 'tau':10, 'alpha':1, 'beta':1}]]

d_ts  = [[TS.discounted, {'T':1000, 'gamma':1, 'alpha':1, 'beta':1}],
         [TS.discounted, {'T':1000, 'gamma':0.99, 'alpha':1, 'beta':1}],
         [TS.discounted, {'T':1000, 'gamma':0.9, 'alpha':1, 'beta':1}]]

plot_means(means, T=1000)
#plt.figure(figsize=(6.4, 14))
#plot_estimated_means(means, runs=sw_ucb, N=100, title_mode="params")
#plt.figure(figsize=(6.4, 14))
#plot_arm_pulls(means, runs=sw_ucb, N=100, title_mode="params")
#plt.figure()
#plot_cumul_regret(instances, runs=sw_ucb, N=100, label_mode="params")
plt.show()
