import numpy as np

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

def sliding_window(bandit, T, tau, B, xi) :
    K = bandit.get_K()

    X = np.zeros((K, tau))
    nbs = np.zeros((K, tau))

    for t in range (K) : 
        r = bandit.play(t, t)
        if t >= K-tau :  
            X[ :, t%tau] = np.zeros(K)
            X[t][t%tau]= r
            nbs[: , t%tau] = np.zeros(K)
            nbs[t][t%tau] = 1
        

    for t in range(K, T) :

        #Compute the ucbs
        means = (1/(np.sum(nbs,1)+1))*np.sum(X, 1) # +1 Pour éviter les valeurs nulles
        trusts = B*np.sqrt(xi*np.log(min(t, tau))/(np.sum(nbs,1)+1))
        ucbs = means + trusts
        
        #Choose and play
        I_t = np.argmax(ucbs)
        r = bandit.play(I_t, t)

        #Update
        X[:, t%tau] = np.zeros(K)
        X[I_t][t%tau] = r
        nbs[:, t%tau] = np.zeros(K)
        nbs[I_t][t%tau] = 1


def f_dsw(bandit, T, tau, gamma, xi, B, f) : 
    K = bandit.get_K()

    X = np.zeros((K, T))
    nbs = np.zeros((K, T))
    N_t = np.zeros(K)
    X_t = np.zeros(K)
    X_hat = np.zeros((K, tau))
    nbs_hat = np.zeros((K, tau))

    for t in range (K) : 
        r = bandit.play(t, t)
        X[ :, t] = np.zeros(K)
        X[t][t]= r
        nbs[: , t] = np.zeros(K)
        nbs[t][t] = 1
        N_t = gamma*N_t + nbs[:, t]
        X_t = gamma*X_t + X[:, t]
        if t >= K-tau :  
            X_hat[ :, t%tau] = np.zeros(K)
            X_hat[t][t%tau]= r
            nbs_hat[: , t%tau] = np.zeros(K)
            nbs_hat[t][t%tau] = 1
        

    for t in range(K, T) :

        #Compute the ucbs
        means_hat = (1/(np.sum(nbs_hat,1)+1))*np.sum(X_hat, 1) # +1 Pour éviter les valeurs nulles
        trusts_hat = B*np.sqrt(xi*np.log(min(t, tau))/(np.sum(nbs_hat,1)+1))
        ucbs_hat = means_hat + trusts_hat
        
        means = (1/(N_t+1))*X_t
        trusts = 2*B*np.sqrt(xi*np.log(np.sum(N_t))/(N_t+1))
        ucbs = means + trusts

        #Choose and play
        new = np.zeros(K)
        for i in range (K) :
            new[i] = f([ucbs[i], ucbs_hat[i]]) 
        I_t = np.argmax(new)
        r = bandit.play(I_t, t)

        #Update
        X_hat[:, t%tau] = np.zeros(K)
        X_hat[I_t][t%tau] = r
        nbs_hat[:, t%tau] = np.zeros(K)
        nbs_hat[I_t][t%tau] = 1

        X[:, t] = np.zeros(K)
        X[I_t][t] = r
        nbs[:, t] = np.zeros(K)
        nbs[I_t][t] = 1

        N_t = gamma*N_t + nbs[:, t]
        X_t = gamma*X_t + X[:, t]