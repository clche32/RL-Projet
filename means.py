def mu_stable_0(t) :
    return 0.5

def mu_stable_1(t) :
    return 0.3

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