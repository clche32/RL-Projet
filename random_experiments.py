import numpy as np
import matplotlib.pyplot as plt

from bandit import NonStationaryBandit
from policy import UCB
from policy import TS
from means import RandomAbruptFunction
from plot_tools import plot_cumul_regret

K = 3 # number of arms
N = 50 # number of instances
instances = []
for n in range(N):
    means = []
    for k in range(K):
        mu_random_abrupt = RandomAbruptFunction(d=0.01, seed=[n,k])
        means.append(mu_random_abrupt)
    instances.append(means)

runs = {UCB.sliding_window: {'T':1000, 'tau':50, 'B':0.5, 'xi':0.6},
        TS.sliding_window : {'T':1000, 'tau':50, 'alpha':1, 'beta':1}}

plot_cumul_regret(instances, runs)
