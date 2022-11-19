import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from bandit import NonStationaryBandit
from policy.UCB import discounted_ucb
from policy.UCB import sliding_window
from policy.UCB import f_dsw
from means import *

means = [mu_stable_0, mu_abrupt_thin, mu_incremental]
bandit = NonStationaryBandit(means, seed=42)

# Avec des plus grandes valeurs de T (et donc par exemple t<3000 et t>5000), on semble avoir des meilleurs résultats, comme si l'algo
# avait le temps d'apprendre contrairement à T = 2000.

T = 10000

N=10

# TODO Choisir la métrique entre 'regret' et 'cumulative_regret'
metric = 'regret'

# TODO Choisir les fonctions à tester [sliding_window, discounted_ucb, f_dsw]
functions =  [sliding_window] 
# TODO Choisir les paramètres respectifs de chaque fonction [ [T, int(4*np.sqrt(T*np.log(T))), 0.5, 0.6], [T, 0.7, 0.5, 0.6], [T, int(4*np.sqrt(T*np.log(T))), 0.7, 0.5, 0.6, np.min] ]
params =  [[T, 50, 0.5, 0.6]] 

for func, params in zip(functions, params):
    cumul_regret = []
    N_s = np.ndarray((10, 3, T))
    print(func.__name__)
    for i in range(N) :
        bandit.regret = []
        bandit.action_played = []
        func(bandit, *params)
        if metric == 'cumulative_regret' :
            cumul_regret.append(bandit.get_cumulative_regret())
            N_s[i] = bandit.get_N()
        elif metric == 'regret' :
            cumul_regret.append(bandit.regret)
            N_s[i] = bandit.get_N()

    avg_cumul_regret = np.mean(cumul_regret, axis=0, dtype=object)
    

    if metric == 'regret' :
        plt.subplot(311) 
        plt.plot(avg_cumul_regret, label="func = " + str(func.__name__))
        plt.xlabel("Time steps")
        plt.ylabel(metric)
        plt.legend()

        # Affichage du regret lissé
        plt.subplot(312)
        plt.plot(savgol_filter(avg_cumul_regret, T, 5), label="func = " + str(func.__name__))
        plt.xlabel("Time steps")
        plt.ylabel(metric)
        plt.legend()

        # TODO ATTENTION : si on fait les 3 fonctions en même temps, il vaut mieux n'afficher qu'un ou deux bras
        plt.subplot(313)        
        for k in range(3):
            avg_N_s_k = np.mean(N_s[:,k], axis=0)
            plt.plot(avg_N_s_k, label="func {}, Arm {}".format(func.__name__, k))
            plt.legend()

    else : 
        plt.subplot(211)
        plt.plot(avg_cumul_regret, label="func = " + str(func.__name__))
        # TODO Décommenter pour afficher la variance
        # std_cumul_regret = np.std(cumul_regret, axis=0)
        # plt.fill_between(np.arange(T), avg_cumul_regret, avg_cumul_regret+std_cumul_regret, alpha=0.4)
        plt.xlabel("Time steps")
        plt.ylabel(metric)
        plt.legend()

        plt.subplot(212)
        # TODO ATTENTION : si on fait les 3 fonctions en même temps, il vaut mieux n'afficher qu'un ou deux bras
        for k in range(3):
            avg_N_s_k = np.mean(N_s[:,k], axis=0)
            plt.plot(avg_N_s_k, label="func {}, Arm {}".format(func.__name__, k))
            plt.legend()

plt.show()