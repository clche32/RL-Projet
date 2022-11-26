import numpy as np

def discounted(bandit, T, alpha, beta, gamma, seed=None):
    '''Play the given non-stationary bandit over T rounds using the
    discounted TS strategy for Bernoulli bandits with given priors
    alpha and beta, discount factor gamma, and (optional) random seed.'''
    K = bandit.get_K()

    # NEW : return statistics (samples) to plot stuff
    stats = np.zeros((K,T))
    # to compute the posterior beta distribution for each action
    # i.e : Beta(alpha + S_k, beta + F_k)
    S = np.zeros(K)
    F = np.zeros(K)

    random = np.random.RandomState(seed)

    for t in range(T):
        # draw random samples
        samples = np.zeros(K)
        for k in range(K):
            samples[k] = random.beta(alpha + S[k], beta + F[k])

        # log stats
        stats[:,t] = samples

        # play the action with highest sample
        k_t = np.argmax(samples)
        r_t = bandit.play(k_t, t)

        # update posterior parameters
        for k in range(K):
            S[k] = gamma*S[k]
            F[k] = gamma*F[k]
        S[k_t] += r_t
        F[k_t] += 1-r_t

    return stats

def sliding_window(bandit, T, alpha, beta, tau, seed=None):
    '''Play the given non-stationary bandit over T rounds using the
    sliding window TS strategy for Bernoulli bandits with given priors
    alpha and beta, window size tau, and (optional) random seed.'''
    K = bandit.get_K()

    # NEW : return statistics (samples) to plot stuff
    stats = np.zeros((K,T))
    # to compute the posterior beta distribution for each action
    # i.e : Beta(alpha + S_k, beta + F_k)
    S = np.zeros(K)
    F = np.zeros(K)

    # to track the action played and reward obtained at time t
    # in order to subtract it from S and F when t is no longer in the window
    action_played = []
    reward_obtained = []

    random = np.random.RandomState(seed)

    for t in range(T):
        # draw random samples
        samples = np.zeros(K)
        for k in range(K):
            samples[k] = random.beta(alpha + S[k], beta + F[k])

        # log stats
        stats[:,t] = samples

        # play the action with highest sample
        k_t = np.argmax(samples)
        r_t = bandit.play(k_t, t)
        action_played.append(k_t)
        reward_obtained.append(r_t)

        # update posterior parameters
        if t >= tau:
            k = action_played[t-tau]
            r = reward_obtained[t-tau]
            S[k] -= r
            F[k] -= 1-r
        S[k_t] += r_t
        F[k_t] += 1-r_t

    return stats

def f_dsw(bandit, T, alpha, beta, gamma, tau, f, seed=None):
    '''Play the given non-stationary bandit over T rounds using the
    f-dsw TS strategy for Bernoulli bandits with given priors alpha
    and beta, discount factor gamma, window size tau, function f,
    and (optional) random seed.'''
    K = bandit.get_K()

    # NEW : return statistics (samples) to plot stuff
    stats = np.zeros((K,T))
    # to compute the posterior beta distribution for each action
    # i.e : Beta(alpha + S_k, beta + F_k)
    S_d = np.zeros(K)
    F_d = np.zeros(K)
    S_sw = np.zeros(K)
    F_sw = np.zeros(K)

    # NEW : to track the action played and reward obtained at time t
    # in order to subtract it from S and F when t is no longer in the window
    action_played = []
    reward_obtained = []

    random = np.random.RandomState(seed)

    for t in range(T):
        # draw random samples
        samples_d = np.zeros(K)
        samples_sw = np.zeros(K)
        samples_f_dsw = np.zeros(K)
        for k in range(K):
            samples_d[k] = random.beta(alpha + S_d[k], beta + F_d[k])
            samples_sw[k] = random.beta(alpha + S_sw[k], beta + F_sw[k])
            samples_f_dsw[k] = f([samples_d[k], samples_sw[k]])

        # log stats
        stats[:,t] = samples_f_dsw

        # play the action with highest sample
        k_t = np.argmax(samples_f_dsw)
        r_t = bandit.play(k_t, t)
        action_played.append(k_t)
        reward_obtained.append(r_t)

        # update posterior parameters (discounted)
        for k in range(K):
            S_d[k] = gamma*S_d[k]
            F_d[k] = gamma*F_d[k]
        S_d[k_t] += r_t
        F_d[k_t] += 1-r_t

        # update posterior parameters (sliding window)
        if t >= tau:
            k = action_played[t-tau]
            r = reward_obtained[t-tau]
            S_sw[k] -= r
            F_sw[k] -= 1-r
        S_sw[k_t] += r_t
        F_sw[k_t] += 1-r_t

    return stats