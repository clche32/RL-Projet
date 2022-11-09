import numpy as np

import math

class NonStationaryBandit:
    def __init__(self, means, seed=None):
        self.means = means ## Ici on a un tableau de fonctions
        self.random = np.random.RandomState(seed)
        
        # for tracking regret
        self.regret = []

    def get_K(self):
        '''Return the number of actions.'''
        return len(self.means)

    def mu_and_a_star(self, t):
        max = 0
        a_star = 0
        for a in range(self.get_K()) :
            if self.means[a](t) > max:
                max = self.means[a](t)
                a_star = a
        return max, a_star

    def play(self, a, t):
        samples = self.random.rand(self.get_K())
        reward = int(samples[a] < self.means[a](t))

        mu_star, _ = self.mu_and_a_star(t)
        self.regret.append(mu_star - self.means[a](t))
        return reward
    
    def get_cumulative_regret(self):
        '''Return an array of the cumulative sum of pseudo-regret per round.'''
        return np.cumsum(self.regret)

def sliding_window(bandit, T, to, B, xi) :
    K = bandit.get_K()

    X = np.zeros((K, to))
    nbs = np.zeros((K, to))

    for t in range (K) : 
        r = bandit.play(t, t)
        if t >= K-to :  
            X[ :, t%to] = np.zeros(K)
            X[t][t%to]= r
            nbs[: , t%to] = np.zeros(K)
            nbs[t][t%to] = 1
        

    for t in range(K, T) :

        #Compute the ucbs
        means = (1/(np.sum(nbs,1)+1))*np.sum(X, 1)
        # TODO Pourquoi N avec to ou gamma ? Erreur dans le papier
        trusts = B*np.sqrt(xi*np.log(min(t, to))/(np.sum(nbs,1)+1))
        ucbs = means + trusts
        
        #Choose and play
        I_t = np.argmax(ucbs)
        r = bandit.play(I_t, t)

        #Update
        X[:, t%to] = np.zeros(K)
        X[I_t][t%to] = r
        nbs[:, t%to] = np.zeros(K)
        nbs[I_t][t%to] = 1


def mu_0(t) :
    return 0.5

def mu_1(t) :
    return 0.3

def mu_2(t) :
    if t < 3000 or t >= 5000 :
        return 0.4
    else :
        return 0.9

means = [mu_0, mu_1, mu_2]
bandit = NonStationaryBandit(means)

sliding_window(bandit, 10000, int(4*np.sqrt(np.log(10000))), 2, 1/2)
print(bandit.get_cumulative_regret())