import numpy as np

def discounted(bandit, T, gamma, xi) :
    K = bandit.get_K()

    # NEW : return statistics (means and trusts) to plot UCBs
    stats = np.zeros((2,K,T-K))
    N_t = np.zeros(K)
    X_t = np.zeros(K)

    for t in range (K) : 
        r = bandit.play(t, t) 
        new_N = np.zeros(K)
        new_N[t] = 1

        new_X = np.zeros(K)
        new_X[t] = r

        N_t = gamma*N_t + new_N
        X_t = gamma*X_t + new_X
    
    for t in range(K, T) :

        #Compute the ucbs
        means = (1/N_t)*X_t # Pas besoin de +1 car N_t positif tant que gamma est non nul
        trusts = np.sqrt(xi*np.log(np.sum(N_t))/N_t)
        ucbs = means + trusts
        stats[0,:,t-K] = means
        stats[1,:,t-K] = trusts
        
        #Choose and play
        I_t = np.argmax(ucbs)
        r = bandit.play(I_t, t)

        #Update
        new_N = np.zeros(K)
        new_N[I_t] = 1

        new_X = np.zeros(K)
        new_X[I_t] = r

        N_t = gamma*N_t + new_N
        X_t = gamma*X_t + new_X
    
    return stats

def sliding_window(bandit, T, tau, xi) :
    K = bandit.get_K()

    # NEW : return statistics (means and trusts) to plot UCBs
    stats = np.zeros((2,K,T-K))
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
        means = (1/(np.sum(nbs,1)+0.01))*np.sum(X, 1) # +0.01 pour éviter division par zéro
        trusts = np.sqrt(xi*np.log(min(t, tau))/(np.sum(nbs,1)+0.01))
        ucbs = means + trusts
        stats[0,:,t-K] = means
        stats[1,:,t-K] = trusts
        
        #Choose and play
        I_t = np.argmax(ucbs)
        r = bandit.play(I_t, t)

        #Update
        X[:, t%tau] = np.zeros(K)
        X[I_t][t%tau] = r
        nbs[:, t%tau] = np.zeros(K)
        nbs[I_t][t%tau] = 1

    return stats

def f_dsw(bandit, T, tau, gamma, xi, f) : 
    K = bandit.get_K()

    N_t = np.zeros(K)
    X_t = np.zeros(K)
    X_hat = np.zeros((K, tau))
    nbs_hat = np.zeros((K, tau))

    for t in range (K) : 
        r = bandit.play(t, t)
        
        new_N = np.zeros(K)
        new_N[t] = 1

        new_X = np.zeros(K)
        new_X[t] = r
    
        N_t = gamma*N_t + new_N
        X_t = gamma*X_t + new_X
        if t >= K-tau :  
            X_hat[ :, t%tau] = np.zeros(K)
            X_hat[t][t%tau]= r
            nbs_hat[: , t%tau] = np.zeros(K)
            nbs_hat[t][t%tau] = 1
        

    for t in range(K, T) :

        #Compute the ucbs
        means_hat = (1/(np.sum(nbs_hat,1)+0.01))*np.sum(X_hat, 1)
        trusts_hat = np.sqrt(xi*np.log(min(t, tau))/(np.sum(nbs_hat,1)+0.01))
        ucbs_hat = means_hat + trusts_hat
        
        means = (1/N_t)*X_t
        trusts = np.sqrt(xi*np.log(np.sum(N_t))/N_t)
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

        new_N = np.zeros(K)
        new_N[I_t] = 1

        new_X = np.zeros(K)
        new_X[I_t] = r

        N_t = gamma*N_t + new_N
        X_t = gamma*X_t + new_X