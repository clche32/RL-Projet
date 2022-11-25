import matplotlib.pyplot as plt
import numpy as np
from bandit import NonStationaryBandit

module_dict = {"policy.TS": "TS",
               "policy.UCB": "UCB"}
name_dict = {"sliding_window": "SW",
             "discounted": "D",
             "f_dsw": "f-DSW"}
param_dict = {"sliding_window": "tau",
             "discounted": "gamma"}

def plot_cumul_regret(instances, runs, N, label_mode=None):
    plt.figure()
    for run in runs:
        strategy, params = run
        cumul_regret = []
        for i in range(len(instances)):
            means = instances[i]
            # Repeat same run N times with different bandit seeds
            for n in range(N):
                bandit = NonStationaryBandit(means, seed=n)
                strategy(bandit, **params)
                cumul_regret.append(bandit.get_cumulative_regret())
        avg_cumul_regret = np.mean(cumul_regret, axis=0)
        std_cumul_regret = np.std(cumul_regret, axis=0)
        if label_mode=="strategy":
            plt.plot(avg_cumul_regret, label=name_dict[strategy.__name__] + "-" + module_dict[strategy.__module__])
        if label_mode=="params":
            param_name = param_dict[strategy.__name__]
            plt.plot(avg_cumul_regret, label="$\%s$ = %s"%(param_name, params[param_name]))
        else:
            plt.plot(avg_cumul_regret)
        plt.fill_between(range(len(avg_cumul_regret)), avg_cumul_regret, avg_cumul_regret+std_cumul_regret, alpha=0.4)
    plt.legend()

def plot_regret(instances, runs, N, label_mode=None):
    plt.figure()
    for run in runs:
        strategy, params = run
        regret = []
        for i in range(len(instances)):
            means = instances[i]
            # Repeat same run N times with different bandit seeds
            for n in range(N):
                bandit = NonStationaryBandit(means, seed=n)
                strategy(bandit, **params)
                regret.append(bandit.regret)
        avg_regret = np.mean(regret, axis=0)
        std_regret = np.std(regret, axis=0)
        if label_mode=="strategy":
            plt.plot(avg_regret, label=name_dict[strategy.__name__] + "-" + module_dict[strategy.__module__])
        if label_mode=="params":
            param_name = param_dict[strategy.__name__]
            plt.plot(avg_regret, label="$\%s$ = %s"%(param_name, params[param_name]))
        else:
            plt.plot(avg_regret)
        plt.fill_between(range(len(avg_regret)), avg_regret, avg_regret+std_regret, alpha=0.4)
    plt.legend()

def plot_means(means, T):
    '''Affiche les fonctions de récompenses moyennes jusqu'à T'''
    plt.figure()
    for i in range(len(means)):
        y = np.zeros(T)
        for t in range(T):
            y[t] = means[i](t)
        plt.plot(y, label="Arm %i"%i)
    plt.ylim(0,1)
    plt.xlabel('$t$')
    plt.ylabel('$\mu_k(t)$')
    plt.legend()

def plot_estimated_means(means, runs, N, title_mode=None):
    '''Pour des runs UCB.'''
    n_runs = len(runs)
    fig = plt.figure(figsize=(6.4, n_runs*4.8))
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
        plt.legend(loc=2)

def plot_ucbs(means, runs, N, title_mode=None):
    '''Pour des runs UCB.'''
    n_runs = len(runs)
    fig = plt.figure(figsize=(6.4, n_runs*4.8))
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
        trusts = []
        for n in range(N):
            bandit = NonStationaryBandit(means, seed=n)
            stats = strategy(bandit, **params)
            est_means.append(stats[0,:,:])
            trusts.append(stats[1,:,:])
        avg_est_means = np.mean(est_means, axis=0)
        avg_trusts = np.mean(trusts, axis=0)
        K = len(means)
        T = params["T"]
        for k in range(K):
            # Plot avg estimated mean
            ax.plot(range(K,T), avg_est_means[k,:], alpha=0.7)
            ax.fill_between(range(K,T), avg_est_means[k,:], avg_est_means[k,:]+avg_trusts[k,:], alpha=0.4)
            # Plot true mean in dashed lines
            y = np.zeros(T)
            for t in range(T):
                y[t] = means[k](t)
            ax.plot(range(T), y, '--', color="C%i"%k, label="Arm %i"%k)
        plt.legend(loc=2)

def plot_posterior_means(means, runs, N, title_mode=None):
    '''Pour des runs Thompson Sampling, mais pas vraiment utile'''
    n_runs = len(runs)
    fig = plt.figure(figsize=(6.4, n_runs*4.8))
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
        plt.legend(loc=2)

def plot_arm_pulls(means, runs, N, title_mode=None):
    '''Affiche le nombre de fois qu'une action a été jouée'''
    n_runs = len(runs)
    fig = plt.figure(figsize=(6.4, n_runs*4.8))
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
            strategy(bandit, **params)
            arm_pulls.append(bandit.get_N())
        avg_arm_pulls = np.mean(arm_pulls, axis=0)
        std_arm_pulls = np.std(arm_pulls, axis=0)
        K = len(means)
        T = params["T"]
        for k in range(K):
            # Plot avg pulls
            ax.plot(avg_arm_pulls[k,:], alpha=0.7)
            ax.fill_between(range(T),avg_arm_pulls[k,:], avg_arm_pulls[k,:]+std_arm_pulls[k,:], alpha=0.4)