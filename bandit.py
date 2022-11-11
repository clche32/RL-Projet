import numpy as np

class NonStationaryBandit:
    def __init__(self, means, seed=None):
        self.means = means ## Ici on a un tableau de fonctions
        self.random = np.random.RandomState(seed)
        
        # for tracking regret
        self.regret = []
        # for tracking actions played
        self.action_played = []

    def get_K(self):
        '''Return the number of actions.'''
        return len(self.means)

    def mu_and_a_star(self, t):
        '''Return \mu_*(t) and a_*(t).'''
        max = 0
        a_star = 0
        for a in range(self.get_K()) :
            if self.means[a](t) > max:
                max = self.means[a](t)
                a_star = a
        return max, a_star

    def play(self, a, t):
        '''Play action a and log regret.'''
        samples = self.random.rand(self.get_K())
        reward = int(samples[a] < self.means[a](t))

        mu_star, _ = self.mu_and_a_star(t)
        self.regret.append(mu_star - self.means[a](t))
        self.action_played.append(a)

        return reward
    
    def get_cumulative_regret(self):
        '''Return an array of the cumulative sum of pseudo-regret per round.'''
        return np.cumsum(self.regret)

    def get_N(self):
        K = self.get_K()
        T = len(self.action_played)
        N = np.zeros((K,T))
        for k in range(K):
            N[k,:] = np.cumsum(np.array(self.action_played) == k)
        return N