import numpy as np

def discounted_ts(bandit, T, alpha, beta, gamma, seed=None):
    '''Play the given non-stationary bandit over T rounds using the
    discounted TS strategy for Bernoulli bandits with given priors
    alpha and beta, discount factor gamma, and (optional) random seed.'''
    K = bandit.get_K()

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

        # play the action with highest sample
        k_t = np.argmax(samples)
        r_t = bandit.play(k_t, t)

        # update posterior parameters
        for k in range(K):
            S[k] = gamma*S[k]
            F[k] = gamma*F[k]
        S[k_t] += r_t
        F[k_t] += 1-r_t

def sliding_window_ts(bandit, T, alpha, beta, tau, seed=None):
    '''Play the given non-stationary bandit over T rounds using the
    sliding window TS strategy for Bernoulli bandits with given priors
    alpha and beta, window size tau, and (optional) random seed.'''
    K = bandit.get_K()

    # to compute the posterior beta distribution for each action
    # i.e : Beta(alpha + S_k, beta + F_k)
    S = np.zeros(K)
    F = np.zeros(K)

    # NEW : to track the action played and reward obtained at time t
    # in order to subtract it from S and F when t is no longer in the window
    action_played = []
    reward_obtained = []

    random = np.random.RandomState(seed)

    for t in range(T):
        # draw random samples
        samples = np.zeros(K)
        for k in range(K):
            samples[k] = random.beta(alpha + S[k], beta + F[k])

        # play the action with highest sample
        k_t = np.argmax(samples)
        r_t = bandit.play(k_t, t)
        action_played.append(k_t)
        reward_obtained.append(r_t)

        # update posterior parameters
        if t > tau:
            k = action_played[t-tau-1]
            r = reward_obtained[t-tau-1]
            S[k] -= r
            F[k] -= 1-r
        S[k_t] += r_t
        F[k_t] += 1-r_t
