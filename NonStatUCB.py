import numpy as np

import matplotlib.pyplot as plt

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

def discounted_ucb(bandit, T, gamma, B, xi) :
    K = bandit.get_K()

    X = np.zeros((K, T))
    nbs = np.zeros((K, T))
    N_t = np.zeros(K)
    X_t = np.zeros(K)

    for t in range (K) : 
        r = bandit.play(t, t) 
        X[ :, t] = np.zeros(K)
        X[t][t]= r
        nbs[: , t] = np.zeros(K)
        nbs[t][t] = 1
        N_t = gamma*N_t + nbs[:, t]
        X_t = gamma*X_t + X[:, t]
    
    for t in range(K, T) :

        #Compute the ucbs
        means = (1/(N_t+1))*X_t
        trusts = 2*B*np.sqrt(xi*np.log(np.sum(N_t))/(N_t+1))
        ucbs = means + trusts
        
        #Choose and play
        I_t = np.argmax(ucbs)
        r = bandit.play(I_t, t)

        #Update
        X[:, t] = np.zeros(K)
        X[I_t][t] = r
        nbs[:, t] = np.zeros(K)
        nbs[I_t][t] = 1

        N_t = gamma*N_t + nbs[:, t]
        X_t = gamma*X_t + X[:, t]

# TODO : à revoir, il semble qu'il y ait une coquille quelque part qui le rend linéaire
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
        means = (1/(np.sum(nbs,1)+1))*np.sum(X, 1) # +1 Pour éviter les valeurs nulles
        # TODO Pourquoi N avec to ou gamma ? Erreur dans le papier ?
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


################ Test unitaire ###############
# means=[mu_0, mu_1, mu_2]
# T=10000

# bandit = NonStationaryBandit(means)
# sliding_window(bandit, T, int(4*np.sqrt(np.log(T))), 2, 1/2)
# discounted_ucb(bandit, T, 1, 2, 1/2) #UCB_1

################ Affichage du graphe du regret en fonction du temps ###############
means=[mu_0, mu_1, mu_2]
N=10
T=10000

bandit = NonStationaryBandit(means)

functions = [sliding_window, discounted_ucb]
params = [ [T, int(4*np.sqrt(np.log(T))), 2, 1/2], [T, 1, 2, 1/2] ]
for func, params in zip(functions, params):
    cumul_regret = []
    print(func.__name__)
    for i in range(N) :
        bandit.regret = []
        func(bandit, *params)
        cumul_regret.append(bandit.get_cumulative_regret())

    avg_cumul_regret = np.mean(cumul_regret, axis=0)
    std_cumul_regret = np.std(cumul_regret, axis=0)

    plt.plot(avg_cumul_regret, label="func = " + str(func.__name__))
    plt.fill_between(np.arange(T), avg_cumul_regret, avg_cumul_regret+std_cumul_regret, alpha=0.4)

    plt.xlabel("Time steps")
    plt.ylabel("Cumulative pseudo-regret")

    plt.legend()
plt.show()