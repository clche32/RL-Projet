import matplotlib.pyplot as plt
import numpy as np
from bandit import NonStationaryBandit

module_dict = {"policy.TS": "TS",
               "policy.UCB": "UCB"}
name_dict = {"sliding_window": "SW",
             "discounted": "D",
             "f_dsw": "f-DSW",
             "ucb1": "Stationnaire",
             "ts_bernoulli": "Stationnaire"}
param_dict = {"sliding_window": "tau",
             "discounted": "gamma"}

def plot_cumul_regret(instances, runs, N, label_mode=None):
    for run_idx in range(len(runs)):
        strategy, params = runs[run_idx]
        cumul_regret = []
        for i in range(len(instances)):
            means = instances[i]
            # Repeat same run N times with different bandit seeds
            for n in range(N):
                bandit = NonStationaryBandit(means, seed=n)
                if module_dict[strategy.__module__] == "TS":
                    params['seed'] = n
                strategy(bandit, **params)
                cumul_regret.append(bandit.get_cumulative_regret())
        avg_cumul_regret = np.mean(cumul_regret, axis=0)
        std_cumul_regret = np.std(cumul_regret, axis=0)
        if label_mode=="strategy":
            plt.plot(avg_cumul_regret, color="C%i"%run_idx, label=name_dict[strategy.__name__] + "-" + module_dict[strategy.__module__])
        if label_mode=="params":
            param_name = param_dict[strategy.__name__]
            plt.plot(avg_cumul_regret, color="C%i"%run_idx, label="$\%s$ = %s"%(param_name, params[param_name]))
        else:
            plt.plot(avg_cumul_regret, color="C%i"%run_idx)
        plt.fill_between(range(len(avg_cumul_regret)), avg_cumul_regret, avg_cumul_regret+std_cumul_regret, alpha=0.4)
    plt.xlabel('$t$')
    plt.ylabel('Pseudo-regret cumulatif')
    plt.legend(loc=2)

def plot_regret(instances, runs, N, label_mode=None):
    for run_idx in range(len(runs)):
        strategy, params = runs[run_idx]
        regret = []
        for i in range(len(instances)):
            means = instances[i]
            # Repeat same run N times with different bandit seeds
            for n in range(N):
                bandit = NonStationaryBandit(means, seed=n)
                if module_dict[strategy.__module__] == "TS":
                    params['seed'] = n
                strategy(bandit, **params)
                regret.append(bandit.regret)
        avg_regret = np.mean(regret, axis=0)
        std_regret = np.std(regret, axis=0)
        if label_mode=="strategy":
            plt.plot(avg_regret, color="C%i"%run_idx, label=name_dict[strategy.__name__] + "-" + module_dict[strategy.__module__])
        if label_mode=="params":
            param_name = param_dict[strategy.__name__]
            plt.plot(avg_regret, color="C%i"%run_idx, label="$\%s$ = %s"%(param_name, params[param_name]))
        else:
            plt.plot(avg_regret, color="C%i"%run_idx)
        plt.fill_between(range(len(avg_regret)), avg_regret, avg_regret+std_regret, alpha=0.4)
    plt.xlabel('$t$')
    plt.ylabel('Pseudo-regret instantané')
    plt.legend()

def plot_means(means, T):
    '''Affiche les fonctions de récompenses moyennes jusqu'à T.'''
    for i in range(len(means)):
        y = np.zeros(T)
        for t in range(T):
            y[t] = means[i](t)
        plt.plot(y, label="$k$ = %i"%i)
    plt.ylim(0,1)
    plt.xlabel('$t$')
    plt.ylabel('$\mu_k(t)$')
    plt.legend()

def plot_estimated_means(means, runs, N, title_mode=None):
    '''Pour des runs UCB. Affiche la moyenne empirique au temps t + écart type.'''
    n_runs = len(runs)
#   fig = plt.figure(figsize=(6.4, n_runs*4.8))
    for run_idx in range(n_runs):
        strategy, params = runs[run_idx]
        ax = plt.subplot(n_runs, 1, run_idx+1)
        ax.set_ylim(0,1)
        if title_mode=="strategy":
            ax.set_title(name_dict[strategy.__name__] + "-" + module_dict[strategy.__module__])
        if title_mode=="params":
            param_name = param_dict[strategy.__name__]
            ax.set_title("$\%s$ = %s"%(param_name, params[param_name]),y=0.85)
        # Repeat same run N times with different bandit seeds
        est_means = []
        for n in range(N):
            bandit = NonStationaryBandit(means, seed=n)
            est_means.append(strategy(bandit, **params)[0,:,:])
        avg_est_means = np.mean(est_means, axis=0)
        std_est_means = np.std(est_means, axis=0)
        K = len(means)
        T = params["T"]
        for k in range(K):
            # Plot avg estimated mean
            ax.plot(range(K,T), avg_est_means[k,:], alpha=0.7)
            ax.fill_between(range(K,T), avg_est_means[k,:], avg_est_means[k,:]+std_est_means[k,:], alpha=0.4)
            # Plot true mean in dashed lines
            y = np.zeros(T)
            for t in range(T):
                y[t] = means[k](t)
            ax.plot(range(T), y, '--', color="C%i"%k, label="Arm %i"%k)
        plt.ylabel('$\hat{\mu}_k(t)$')
        plt.legend(loc=2)

def plot_ucbs(means, runs, N, title_mode=None):
    '''Pour des runs UCB. Affiche la moyenne empirique au temps t + intervalle de conf. '''
    n_runs = len(runs)
#   fig = plt.figure(figsize=(6.4, n_runs*4.8))
    for run_idx in range(n_runs):
        strategy, params = runs[run_idx]
        ax = plt.subplot(n_runs, 1, run_idx+1)
        ax.set_ylim(0,1)
        if title_mode=="strategy":
            ax.set_title(name_dict[strategy.__name__] + "-" + module_dict[strategy.__module__],y=0.80)
        if title_mode=="params":
            param_name = param_dict[strategy.__name__]
            ax.set_title("$\%s$ = %s"%(param_name, params[param_name]),y=0.80)
        # Repeat same run N times with different bandit seeds
        ucbs = []
        for n in range(N):
            bandit = NonStationaryBandit(means, seed=n)
            stats = strategy(bandit, **params)
            ucbs.append(stats[1,:,:])
        avg_ucbs = np.mean(ucbs, axis=0)
        std_ucbs = np.std(ucbs, axis=0)
        K = len(means)
        T = params["T"]
        for k in range(K):
            # Plot avg estimated mean
            ax.plot(range(K,T), avg_ucbs[k,:], alpha=0.7)
            ax.fill_between(range(K,T), avg_ucbs[k,:], avg_ucbs[k,:]+std_ucbs[k,:], alpha=0.4)
            # Plot true mean in dashed lines
            y = np.zeros(T)
            for t in range(T):
                y[t] = means[k](t)
            ax.plot(range(T), y, '--', color="C%i"%k, label="Arm %i"%k)
        plt.ylabel(r'$\mathrm{UCB}_k(t)$')
        plt.legend(loc=2)

def plot_samples(means, runs, N, title_mode=None):
    '''Pour des runs Thompson Sampling'''
    n_runs = len(runs)
#   fig = plt.figure(figsize=(6.4, n_runs*4.8))
    for run_idx in range(n_runs):
        strategy, params = runs[run_idx]
        ax = plt.subplot(n_runs, 1, run_idx+1)
        ax.set_ylim(0,1)
        if title_mode=="strategy":
            ax.set_title(name_dict[strategy.__name__] + "-" + module_dict[strategy.__module__], y=0.75)
        if title_mode=="params":
            param_name = param_dict[strategy.__name__]
            ax.set_title("$\%s$ = %s"%(param_name, params[param_name]),y=0.75)
        # Repeat same run N times with different bandit seeds
        est_means = []
        for n in range(N):
            bandit = NonStationaryBandit(means, seed=n)
            params['seed'] = n
            est_means.append(strategy(bandit, **params))
        avg_est_means = np.mean(est_means, axis=0)
        std_est_means = np.std(est_means, axis=0)
        K = len(means)
        T = params["T"]
        for k in range(K):
            # Plot avg estimated mean
            ax.plot(range(T), avg_est_means[k,:], alpha=0.7)
            ax.fill_between(range(T), avg_est_means[k,:], avg_est_means[k,:]+std_est_means[k,:], alpha=0.4)
            # Plot true mean in dashed lines
            y = np.zeros(T)
            for t in range(T):
                y[t] = means[k](t)
            ax.plot(range(T), y, '--', color="C%i"%k, label="Arm %i"%k)
        plt.ylabel(r'$\theta_k(t)$')
        plt.legend(loc=2)

def plot_arm_pulls(means, runs, N, title_mode=None):
    '''Affiche le nombre de fois qu'une action a été jouée.'''
    n_runs = len(runs)
#   fig = plt.figure(figsize=(6.4, n_runs*4.8))
    for run_idx in range(n_runs):
        strategy, params = runs[run_idx]
        ax = plt.subplot(n_runs, 1, run_idx+1)
        if title_mode=="strategy":
            ax.set_title(name_dict[strategy.__name__] + "-" + module_dict[strategy.__module__])
        if title_mode=="params":
            param_name = param_dict[strategy.__name__]
            ax.set_title("$\%s$ = %s"%(param_name, params[param_name]),y=0.85)
        # Repeat same run N times with different bandit seeds
        arm_pulls = []
        for n in range(N):
            bandit = NonStationaryBandit(means, seed=n)
            if module_dict[strategy.__module__] == "TS":
                params['seed'] = n
            strategy(bandit, **params)
            arm_pulls.append(bandit.get_N())
        avg_arm_pulls = np.mean(arm_pulls, axis=0)
        std_arm_pulls = np.std(arm_pulls, axis=0)
        K = len(means)
        T = params["T"]
        for k in range(K):
            # Plot avg pulls
            ax.plot(avg_arm_pulls[k,:], alpha=0.7, label="Arm %i"%k)
            ax.fill_between(range(T),avg_arm_pulls[k,:], avg_arm_pulls[k,:]+std_arm_pulls[k,:], alpha=0.4)
        plt.ylabel('$N_k(t)$')
        plt.legend(loc=2)