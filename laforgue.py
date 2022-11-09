import numpy as np

import math

class NonStationaryBandit:
    def __init__(self, means, seed=None):
        self.means = means ## Ici on a un tableau de fonctions
        self.random = np.random.RandomState(seed)
        
        # for tracking regret
        self.regret = []
        self.to = np.ones(self.get_K()) # Initialisation (1) page 2

    def delta(self, a, to, a_bis):
        if a == a_bis :
            if to <= 0 :
                return to - 1
            else :
                return -1
        else : 
            if to <= 0 :
                return 1
            else :
                return to + 1
    
    def get_K(self):
        '''Return the number of actions.'''
        return len(self.means)

    def mu_and_a_star(self):
        max = 0
        a_star = 0
        for a in range(self.get_K()) :
            if self.means[a](self.to[a]) > max:
                max = self.means[a](self.to[a])
                a_star = a
        return max, a_star

    def play(self, a):
        samples = self.random.rand(self.get_K())
        reward = int(samples[a] < self.means[a](self.to[a]))
        for a_bis in range(self.get_K()) :
            self.to[a_bis] = self.delta(a_bis, self.to[a], a)

        mu_star, _ = self.mu_and_a_star()
        self.regret.append(mu_star - self.means[a](self.to[a]))
        return reward
    
    def get_cumulative_regret(self):
        '''Return an array of the cumulative sum of pseudo-regret per round.'''
        return np.cumsum(self.regret)


def isi_combucb1(bandit, T, d):
    K = bandit.get_K()
    To = np.zeros((K, d))
    X_bar = np.zeros((T, K, d)) #Il faut garder en mémoire pour la toute fin de l'algorithme
    U = np.full((K, d), np.inf)

    all_tos = np.zeros(d) # TODO : comment avoir tous les to de t=0 à d dès t=1 ?

    for t in range (T//d) :
        B_hat = np.zeros(d)
        
        #Play 
        # On crée un tableau de toutes les combinaisons possibles (TODO : à améliorer, ça semble bizzare de faire autant de combinaisons)
        all_As = np.array(np.meshgrid([i for i in range(K)] for _ in range(d))).T.reshape(-1,3)
        max = 0
        for A in all_As :
            sum = 0
            for s in range(d) :
                if A[s] in A[0:s-1] :
                    sum += U[A[s]][all_tos[s][A[s]]]
            if sum > max :
                max = sum
                B_hat = np.copy(A)

        X = np.zeros((K, d))
        for s in range(d) : 
            X[B_hat[s]][all_tos[B_hat[s]][s]] = bandit.play(B_hat[s])
        

        for i in range(K) :
            for j in range(d) :
                if (B_hat[j] == i) : # Si i est joué en j-ième
                    X_bar[t+1][i][j] = (To[i][j]*X_bar[t][i][j] + X[i][j])/(To[i][j]+1)
                    To[i][j] += 1
                else :
                    X_bar[t+1][i][j] = X_bar[t][i][j]

                U[i][j] = X_bar[To[i][j]][i][j] + np.sqrt(1.5*np.log(t+1)/To[i][j]) # α = 1.5 dans les théorèmes

def mu_0(to) :
    if to == 3 :
        return 0.95
    else :
        return 0

def mu_1(to) :
    if to == 6 :
        return 0.6
    elif to >= 9 :
        return 0.96
    else : 
        return 0.14

def mu_2(to) :
    return 0.15

means = [mu_0, mu_1, mu_2, mu_2, mu_2]
bandit = NonStationaryBandit(means)

print(bandit.play(0)) # mu_0 à 0 donc on a 0 comme récompense
print(bandit.play(1))