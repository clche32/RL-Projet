import numpy as np

def mu_stable_0(t) :
    return 0.5

def mu_stable_1(t) :
    return 0.3

def mu_abrupt(t) :
    if t < 200 or t >= 600:
        return 0.3
    else:
        return 0.7

def mu_abrupt_decreasing(t) :
    if t < 500:
        return 0.7
    else:
        return 0.3

def mu_abrupt_increasing(t) :
    if t < 500:
        return 0.3
    else:
        return 0.7

def mu_abrupt_thin(t) :
    if t < 300 or t >= 1000 :
        return 0.1
    else :
        return 0.9

def mu_abrupt_large(t) :
    if t < 3000 or t >= 5000 :
        return 0.4
    else :
        return 0.9

def mu_reocurring_0(t) :
    if (not ( (t/1000) %2) ) : # Si le chiffre des milliers est pair
        return 1
    else :
        return 0.1

def mu_incremental(t) :
    if (t>200) :
        return 0.8 - 0.6/(t-200) # vaut 0.2 à t = 201 puis tend vers 0.8
    else :
        return 0.2

def mu_linear(t) :
    if (t<1000):
        return t/1000
    else:
        return 1

class RandomAbruptFunction:
    def __init__(self, d, seed=None):
        self.d = d # probability that a change occurs
        self.random = np.random.RandomState(seed)
        self.mu = [] # past history : mu(0), ... mu(t)
        self.t = -1 # current time step, see above

    def __call__(self, t):
        if t == self.t+1: # mu(t) can only be evaluated if mu(t-1) has been evaluated
            self.t = t
            change = self.random.rand() < self.d   
            if change or t == 0:
                self.mu.append(self.random.rand())
            else:
                self.mu.append(self.mu[t-1])
        return self.mu[t] # only exists for t in [0,...self.t]

# With such a random function f, we cannot call f(t) without having previously called f(t-1).