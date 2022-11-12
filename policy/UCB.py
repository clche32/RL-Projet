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